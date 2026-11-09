"""Resumable offline search and measurement for the per-token FP8 heuristic."""

import argparse
import cProfile
import hashlib
import importlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import pstats
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm-fp8-heuristic-refine"))

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import RowWiseTorchFP8ScaledMMLinearKernel
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
from vllm.platforms import current_platform

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "plan.json").read_text())
LINEAR = importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.triton")
SHARED = importlib.import_module("vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm")


def snapshot(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = snapshot("fp8_coverage_shared_baseline", ROOT / "baseline/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def append(path, value):
    with path.open("a") as f:
        f.write(json.dumps(value) + "\n")


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def inputs(m, n, k, bias=False, dtype=None):
    torch.manual_seed(18347 + m + n + k)
    dtype = dtype or current_platform.fp8_dtype()
    a = (torch.randn(m, k, device="cuda") * 0.2).to(dtype)
    w = (torch.randn(n, k, device="cuda") * 0.2).to(dtype)
    sa = torch.rand(m, 1, device="cuda") + 0.5
    sb = torch.rand(n, 1, device="cuda") + 0.5
    bias = (torch.randn(n, device="cuda") * 0.2).bfloat16() if bias else None
    return a, w.t(), sa, sb, bias


def reference_check(out, x):
    a, b, sa, sb, bias = x
    rows = torch.linspace(0, a.shape[0] - 1, min(a.shape[0], 16), device="cuda").long().unique()
    cols = torch.linspace(0, b.shape[1] - 1, min(b.shape[1], 32), device="cuda").long().unique()
    ref = (a[rows].float() @ b[:, cols].float()) * sa.reshape(-1, 1)[rows] * sb.reshape(-1)[cols]
    ref = ref.to(out.dtype)
    if bias is not None:
        ref = ref + bias[cols]
    torch.testing.assert_close(out[rows[:, None], cols[None, :]], ref, rtol=1e-2, atol=1e-2)
    assert torch.isfinite(out).all().item()


def graph_time(fn, m, n, k, repeats=3):
    out = fn()
    calls = 8 if 2*m*n*k < 2e9 else 1
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            out = fn()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(samples)


def config(bm, bn, bk, warps=4, stages=2, group=1, stripe=0, unmasked=False):
    return dict(block_size_m=bm, block_size_n=bn, block_size_k=bk,
                num_warps=warps, num_stages=stages, group_m=group,
                n_stripe=stripe, unmasked_loads=unmasked, use_heuristic=False)


def candidates(m):
    tiles = [
        (16,32,128),(16,64,128),(16,64,256),(16,128,256),
        (32,32,128),(32,64,128),(32,64,256),(32,128,128),
        (64,32,128),(64,64,64),(64,64,128),(64,64,256),
        (64,128,64),(64,128,128),(64,128,256),(64,256,128),
        (128,32,128),(128,64,64),(128,64,128),(128,128,64),
        (128,128,128),(128,256,64),(128,256,128),
    ]
    return [config(*tile) for tile in tiles]


def untuned():
    empty = ROOT / "empty-config-root"
    (empty / "configs").mkdir(parents=True, exist_ok=True)
    fp8_utils.__file__ = str(empty / "fp8_utils.py")
    fp8_utils.get_w8a8_per_token_fp8_configs.cache_clear()


def search(args):
    outpath = ROOT / "search.jsonl"
    done = {(r['M'],r['N'],r['K']) for r in records(outpath)}
    jobs = [(m,n,k) for n,k in PLAN['discovery_n_k'] for m in PLAN['discovery_m']]
    for index,(m,n,k) in enumerate(jobs):
        if (m,n,k) in done: continue
        x = inputs(m,n,k)
        baseline = lambda: BASE.triton_scaled_mm(*x[:4], torch.bfloat16, x[4])
        ref = baseline()
        reference_check(ref,x)
        ranked, skipped = [], []
        configs = candidates(m)
        random.Random(m+n+k).shuffle(configs)
        def test(c):
            fn=lambda: SHARED.triton_scaled_mm(*x[:4],torch.bfloat16,x[4],**c)
            try:
                y=fn()
                reference_check(y,x)
                del y
                return graph_time(fn,m,n,k)
            except (SHARED.triton.CompilationError, SHARED.triton.runtime.errors.OutOfResources) as exc:
                skipped.append({'config':c,'error':str(exc)[-600:]})
                return None
        for c in configs:
            us=test(c)
            if us is not None:ranked.append({'config':c,'gpu_us':us})
        ranked.sort(key=lambda r:r['gpu_us'])
        # Refine pipeline, warps and scheduling around independent tile finalists.
        refinements=[]
        for r in ranked[:3]:
            for warps,stages in itertools.product([4,8],[1,2,3]):
                refinements.append(r['config']|dict(num_warps=warps,num_stages=stages))
        for c in refinements:
            if any(c==r['config'] for r in ranked):continue
            us=test(c)
            if us is not None:ranked.append({'config':c,'gpu_us':us})
        ranked.sort(key=lambda r:r['gpu_us'])
        refinements=[]
        for r in ranked[:2]:
            for group,stripe in [(2,0),(4,0),(8,0),(1,2),(1,4),(1,8)]:
                refinements.append(r['config']|dict(group_m=group,n_stripe=stripe))
            c=r['config']
            if m%c['block_size_m']==n%c['block_size_n']==k%c['block_size_k']==0:
                refinements.append(c|dict(unmasked_loads=True))
        for c in refinements:
            us=test(c)
            if us is not None:ranked.append({'config':c,'gpu_us':us})
        ranked.sort(key=lambda r:r['gpu_us'])
        # Repeat finalists and the original heuristic in shuffled order.
        finalists=ranked[:4]
        samples={'baseline':[]}|{str(j):[] for j in range(len(finalists))}
        for round_id in range(5):
            order=list(samples)
            random.Random(round_id+m+n+k).shuffle(order)
            for name in order:
                fn=baseline if name=='baseline' else lambda c=finalists[int(name)]['config']:SHARED.triton_scaled_mm(*x[:4],torch.bfloat16,x[4],**c)
                samples[name].append(graph_time(fn,m,n,k))
        winner=min(range(len(finalists)),key=lambda j:statistics.median(samples[str(j)]))
        record=dict(M=m,N=n,K=k,config=finalists[winner]['config'],gpu_us=statistics.median(samples[str(winner)]),baseline_us=statistics.median(samples['baseline']),samples=samples,ranked=ranked,skipped=skipped)
        append(outpath,record)
        print(f"search {index+1}/{len(jobs)} {(m,n,k)} old={record['baseline_us']:.2f} best={record['gpu_us']:.2f}us config={record['config']}",flush=True)
        del ref,x,baseline,fn,finalists,ranked
        torch.cuda.empty_cache()


def profile(args):
    untuned()
    result=[]
    for m,n,k in [(1,1024,1024),(256,1024,1024),(1,3840,4096)]:
        x=inputs(m,n,k)
        cfg=FP8ScaledMMLinearLayerConfig(weight_quant_key=kFp8StaticChannelSym,activation_quant_key=kFp8DynamicTokenSym,weight_shape=(n,k),input_dtype=torch.bfloat16,out_dtype=torch.bfloat16)
        names=['weight','weight_scale','input_scale','input_scale_ub']
        tk=LINEAR.TritonFp8PerTokenScaledMMKernel(cfg,names)
        pk=RowWiseTorchFP8ScaledMMLinearKernel(cfg,names)
        kw=dict(A=x[0],B=x[1],As=x[2],Bs=x[3],out_dtype=torch.bfloat16,bias=None,output_shape=[m,n])
        methods={'torch':lambda:pk.apply_scaled_mm(**kw),'triton_backend':lambda:tk.apply_scaled_mm(**kw),'triton_direct':lambda:LINEAR._w8a8_triton_per_token_scaled_mm_func(*x[:4],torch.bfloat16,None,False,args.label!='baseline'),'shared':lambda:SHARED.triton_scaled_mm(*x[:4],torch.bfloat16)}
        measured={}
        for name,fn in methods.items():
            for _ in range(100):fn()
            torch.cuda.synchronize()
            times=[]
            for _ in range(21):
                start=time.thread_time_ns()
                for _ in range(32):fn()
                times.append((time.thread_time_ns()-start)/32000)
                torch.cuda.synchronize()
            profiler=cProfile.Profile()
            profiler.enable()
            for _ in range(300):fn()
            profiler.disable()
            torch.cuda.synchronize()
            stream=io.StringIO()
            pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35)
            (ROOT/f"profile-{args.label}-{m}-{n}-{k}-{name}.txt").write_text(stream.getvalue())
            measured[name]=dict(median_cpu_us=statistics.median(times),samples=times)
        result.append(dict(M=m,N=n,K=k,methods=measured))
        print(result[-1],flush=True)
    dump(ROOT/f'profile-{args.label}.json',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['search','profile'])
    parser.add_argument('--label',default='baseline')
    args=parser.parse_args()
    assert os.environ['HIP_VISIBLE_DEVICES']=='1'
    assert torch.cuda.device_count()==1
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[-1]})
    dump(ROOT/f'environment-{args.phase}-{args.label}.json',dict(torch=torch.__version__,hip=torch.version.hip,gpu=str(torch.cuda.get_device_properties(0)),physical_gpu=1,cpu=affinity[-1],source=str(LINEAR.__file__),started=time.time()))
    with set_current_vllm_config(VllmConfig()),torch.inference_mode():
        globals()[args.phase](args)


if __name__=='__main__':main()
