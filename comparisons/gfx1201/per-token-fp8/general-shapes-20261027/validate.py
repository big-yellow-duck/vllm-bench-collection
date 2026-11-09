"""Paired production-backend validation with frozen discovery/holdout shapes."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import time

import experiment as e
from vllm.model_executor.layers.quantization.utils import fp8_per_token_heuristic as h


def baseline_class():
    import ast
    old_rules=e.snapshot('fp8_refine_before_rules',e.ROOT/'before/vllm/model_executor/layers/quantization/utils/fp8_per_token_heuristic_rules.py')
    old_launch=e.snapshot('fp8_refine_before_launch',e.ROOT/'before/vllm/model_executor/layers/quantization/utils/fp8_per_token_heuristic.py')
    old_launch.get_heuristic_config=old_rules.get_heuristic_config
    path=e.ROOT/'before/vllm/model_executor/kernels/linear/scaled_mm/triton.py'
    class Rewrite(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if node.attr=='w8a8_triton_per_token_scaled_mm_func':node.attr='benchmark_fp8_refine_previous'
            return self.generic_visit(node)
        def visit_ImportFrom(self,node):
            if any(a.name=='fp8_per_token_heuristic' for a in node.names):
                return ast.Import(names=[ast.alias(name='fp8_refine_before_launch',asname='heuristic')])
            return node
    names={'TritonFp8PerTokenScaledMMKernel','_w8a8_triton_per_token_scaled_mm_func','_w8a8_triton_per_token_scaled_mm_fake'}
    nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    tree=ast.fix_missing_locations(Rewrite().visit(ast.Module(body=nodes,type_ignores=[])))
    scope=dict(e.LINEAR.__dict__);exec(compile(tree,str(path),'exec'),scope)
    e.LINEAR.direct_register_custom_op('benchmark_fp8_refine_previous',scope['_w8a8_triton_per_token_scaled_mm_func'],fake_impl=scope['_w8a8_triton_per_token_scaled_mm_fake'])
    return scope['TritonFp8PerTokenScaledMMKernel']


OLD=baseline_class()
REAL_UTILS=e.fp8_utils.__file__


def kernels(n,k):
    cfg=e.FP8ScaledMMLinearLayerConfig(weight_quant_key=e.kFp8StaticChannelSym,
        activation_quant_key=e.kFp8DynamicTokenSym,weight_shape=(n,k),
        input_dtype=e.torch.bfloat16,out_dtype=e.torch.bfloat16)
    names=['weight','weight_scale','input_scale','input_scale_ub']
    return {'torch':e.RowWiseTorchFP8ScaledMMLinearKernel(cfg,names),
            'before':OLD(cfg,names),'after':e.LINEAR.TritonFp8PerTokenScaledMMKernel(cfg,names)}


def check_full(out,ref):
    for start in range(0,out.shape[0],256):
        e.torch.testing.assert_close(out[start:start+256],ref[start:start+256],rtol=1e-2,atol=1e-2)


def measure(fn,m,n,k,eager=True):
    if not eager:return dict(graph_gpu_us=e.graph_time(fn,m,n,k))
    for _ in range(3):out=fn()
    e.torch.cuda.synchronize()
    sync=[]
    for _ in range(8):
        t=time.perf_counter_ns();out=fn();e.torch.cuda.synchronize()
        sync.append((time.perf_counter_ns()-t)/1000)
    t=time.perf_counter_ns();cpu=time.thread_time_ns()
    for _ in range(16):out=fn()
    enqueue=(time.perf_counter_ns()-t)/16000
    cpu=(time.thread_time_ns()-cpu)/16000
    e.torch.cuda.synchronize()
    batch=(time.perf_counter_ns()-t)/16000
    return dict(sync_wall_us=statistics.median(sync),batch_wall_us=batch,
                enqueue_us=enqueue,cpu_us=cpu,graph_gpu_us=e.graph_time(fn,m,n,k))


def run_case(m,n,k,objs,rounds=3,eager=True):
    x=e.inputs(m,n,k)
    kw=dict(A=x[0],B=x[1],As=x[2],Bs=x[3],out_dtype=e.torch.bfloat16,bias=None,output_shape=[m,n])
    methods={name:lambda obj=obj:obj.apply_scaled_mm(**kw) for name,obj in objs.items()}
    outputs={name:fn() for name,fn in methods.items()}
    for name,out in outputs.items():
        e.reference_check(out,x)
        check_full(out,outputs['torch'])
    del outputs
    samples={name:[] for name in methods}
    for repeat in range(rounds):
        order=list(methods);random.Random(m+n+k+repeat).shuffle(order)
        for name in order:samples[name].append(measure(methods[name],m,n,k,eager))
    medians={name:{metric:statistics.median(v[metric] for v in values)
                   for metric in values[0]} for name,values in samples.items()}
    return dict(M=m,N=n,K=k,median_us=medians,samples_us=samples,
                config=h.get_heuristic_config(m,n,k) if hasattr(h,'get_heuristic_config') else None,
                has_tuned_configs=objs['after'].has_tuned_configs,
                correctness='full cross-backend, finite, sampled FP32 reference passed')


def matrix(args):
    selected=None
    if args.suite=='tuned':
        e.fp8_utils.__file__=REAL_UTILS;e.fp8_utils.get_w8a8_per_token_fp8_configs.cache_clear()
        paths=(e.Path(REAL_UTILS).parent/'configs').glob('*fp8_w8a8_per_token*float8_e4m3fn*out_dtype=bfloat16.json')
        pairs=sorted({tuple(map(int,re.match(r'N=(\d+),K=(\d+),',p.name).groups())) for p in paths})
        ms=[1,8,33,64,256,2048,8192,32768]
    else:
        e.untuned()
        pairs=e.PLAN['discovery_n_k']+e.PLAN['holdout_n_k']+e.PLAN.get('model_holdout_n_k',[])+e.PLAN.get('supplemental_discovery_n_k',[])+e.PLAN.get('fresh_holdout_n_k',[])
        ms=e.PLAN['validation_m']
        if args.suite=='smoke':
            pairs=e.PLAN['discovery_n_k'][:5];ms=[1,16,64,256,4096]
        if args.suite=='blind':pairs=e.PLAN['new_blind_n_k'];ms=e.PLAN['new_blind_m']
        if args.suite=='prospective2':pairs=e.PLAN['prospective2_n_k'];ms=e.PLAN['prospective2_m']
        if args.suite=='boundaries':
            pairs=[[32768,256],[32768,1024],[24560,4096],[24576,6144],[18944,3584],[30720,3840],[3584,18944],[3840,15360]]
            ms=[127,128,129,191,192,193,511,512,513,1023,1024,1025,4095,4096,4097]
        if args.suite=='regressions':
            broad=e.records(e.ROOT/'validation-broad.jsonl')
            assert len(broad)==e.PLAN['broad_expected_cases']
            selected={}
            for r in broad:
                a,b=r['median_us']['after'],r['median_us']['torch']
                if any(a[metric]/b[metric]>ratio and a[metric]-b[metric]>absolute
                       for metric,ratio,absolute in [('graph_gpu_us',1.03,0.5),('batch_wall_us',1.05,1),('sync_wall_us',1.05,1)]):
                    selected.setdefault((r['N'],r['K']),[]).append(r['M'])
            pairs=sorted(selected)
    out=e.ROOT/f'validation-{args.suite}.jsonl'
    done={(r['M'],r['N'],r['K']) for r in e.records(out)}
    expected=sum(map(len,selected.values())) if selected is not None else len(pairs)*len(ms)
    for n,k in pairs:
        objs=kernels(n,k)
        if args.suite=='tuned':
            e.untuned();objs['heuristic']=kernels(n,k)['after']
            e.fp8_utils.__file__=REAL_UTILS;e.fp8_utils.get_w8a8_per_token_fp8_configs.cache_clear()
            assert objs['after'].has_tuned_configs and not objs['heuristic'].has_tuned_configs
        for m in selected[(n,k)] if selected is not None else ms:
            if (m,n,k) in done:continue
            try:
                record=run_case(m,n,k,objs,rounds=7 if args.suite=='regressions' else 5 if args.suite=='smoke' else 3)
            except (RuntimeError,AssertionError) as exc:
                e.append(e.ROOT/f'validation-{args.suite}-errors.jsonl',dict(M=m,N=n,K=k,error=str(exc)))
                raise
            record['split']='holdout' if [n,k] in e.PLAN['holdout_n_k'] else 'discovery'
            if [n,k] in e.PLAN.get('model_holdout_n_k',[]):record['split']='model_holdout'
            if [n,k] in e.PLAN.get('supplemental_discovery_n_k',[]):record['split']='supplemental_discovery'
            if [n,k] in e.PLAN.get('fresh_holdout_n_k',[]):record['split']='fresh_holdout'
            e.append(out,record);done.add((m,n,k))
            a=record['median_us'];print(f"{args.suite} {len(done)}/{expected} {(m,n,k)} graph={ {name:round(v['graph_gpu_us'],2) for name,v in a.items()} }",flush=True)
            gc.collect()
        del objs
        e.torch.cuda.empty_cache()


def controls(args):
    e.untuned()
    rows=[]
    for m,n,k in [(1,1024,1024),(33,512,512),(129,2048,1536),(257,3840,4112),(24,3584,18944),(33,3584,18944),(193,32768,256),(129,24560,4096),(65,19968,5632),(128,18944,3584),(257,24544,6656)]:
        obj=kernels(n,k)['after']
        for dtype in [e.torch.bfloat16,e.torch.float16]:
            for bias in [False,True]:
                for style in ['2d','1d','strided','offset']:
                    x=list(e.inputs(m,n,k,bias))
                    if bias:x[4]=x[4].to(dtype)
                    if style=='1d':x[2]=x[2].flatten();x[3]=x[3].flatten()
                    if style=='strided':
                        a=e.torch.zeros((m,2),device='cuda');a[:,0]=x[2].flatten();x[2]=a[:,0:1]
                        b=e.torch.zeros((n,2),device='cuda');b[:,0]=x[3].flatten();x[3]=b[:,0:1]
                    if style=='offset':
                        a=e.torch.empty(m+1,device='cuda');a[1:]=x[2].flatten();x[2]=a[1:]
                        b=e.torch.empty(n+1,device='cuda');b[1:]=x[3].flatten();x[3]=b[1:]
                    kw=dict(A=x[0],B=x[1],As=x[2],Bs=x[3],bias=x[4],out_dtype=dtype,output_shape=[m,n])
                    for repeat in range(3):
                        actual=obj.apply_scaled_mm(**kw);e.reference_check(actual,x)
                    rows.append(dict(M=m,N=n,K=k,dtype=str(dtype),bias=bias,style=style,status='pass'))
    # Real compiler graph capture, changing token counts and pointer allocations.
    for tuned in [False,True]:
        if tuned:e.fp8_utils.__file__=REAL_UTILS;e.fp8_utils.get_w8a8_per_token_fp8_configs.cache_clear()
        obj=kernels(3840,2048)['after']
        def call(a,b,sa,sb):
            return obj.apply_scaled_mm(A=a,B=b,As=sa,Bs=sb,bias=None,out_dtype=e.torch.bfloat16,output_shape=[a.shape[0],b.shape[1]])
        compiled=e.torch.compile(call,backend='inductor',dynamic=True,fullgraph=True)
        for m in [1,33,64,257,33]:
            x=e.inputs(m,3840,2048)
            out=compiled(*x[:4]);e.reference_check(out,x)
            check_full(out,call(*x[:4]))
        rows.append(dict(compiled=True,tuned=tuned,status='pass'))
    e.untuned()
    variants=[('gemm',(129,1024,1024),e.config(32,64,128)),
              ('gemv',(1,1024,1024),e.config(16,8,256)|dict(variant='gemv')),
              ('split_k',(1,1024,16384),e.config(16,64,128)|dict(split_k=16)),
              ('persistent',(4096,1024,1024),e.config(128,128,128,group=4)|dict(variant='persistent',programs=32)),
              ('persistent_64_small_tile',(1025,512,512),e.config(64,128,128,group=4)|dict(variant='persistent',programs=64)),
              ('persistent_64_large_tile',(4097,30720,3840),e.config(128,128,128,warps=8,group=4)|dict(variant='persistent',programs=64)),
              ('selected_decode',(1,1024,16384),None),
              ('selected_prefill',(1024,6144,4096),None)]
    for name,(m,n,k),choice in variants:
        x=e.inputs(m,n,k)
        if choice is None:
            obj=kernels(n,k)['after']
            def fn():return obj.apply_scaled_mm(A=x[0],B=x[1],As=x[2],Bs=x[3],bias=None,out_dtype=e.torch.bfloat16,output_shape=[m,n])
        else:
            def fn():return h.launch_fp8_per_token(*x[:4],e.torch.bfloat16,None,choice)
        for _ in range(3):fn()
        e.torch.cuda.synchronize()
        graph=e.torch.cuda.CUDAGraph()
        with e.torch.cuda.graph(graph):captured=fn()
        for _ in range(3):
            x[0].copy_((0.2*e.torch.randn(m,k,device='cuda')).to(x[0].dtype))
            x[2].mul_(1.1);x[3].mul_(.9)
            graph.replay();e.torch.cuda.synchronize()
            e.reference_check(captured,x)
            check_full(captured,e.BASE.triton_scaled_mm(*x[:4],e.torch.bfloat16))
        rows.append(dict(variant=name,graph_replay=True,status='pass'))
    e.dump(e.ROOT/'validation-controls.json',rows)
    print('PASS controls',len(rows),flush=True)


def cold(args):
    e.untuned()
    rows=[]
    for m,n,k in [(1,1024,1024),(256,1024,1024),(1,3840,4096),(256,3840,4096),(1,6144,4096),(1024,6144,4096)]:
        objs=kernels(n,k)
        count=max(4,min(32,math.ceil(32*1024**2/(n*k))))
        operands=[e.inputs(m,n,k) for _ in range(count)]
        methods={}
        for name,obj in objs.items():
            def fn(obj=obj):
                for x in operands:
                    obj.apply_scaled_mm(A=x[0],B=x[1],As=x[2],Bs=x[3],bias=None,out_dtype=e.torch.bfloat16,output_shape=[m,n])
            methods[name]=fn
        samples={name:[] for name in methods}
        for repeat in range(5):
            order=list(methods);random.Random(m+n+k+repeat).shuffle(order)
            for name in order:samples[name].append(e.graph_time(methods[name],m*count,n,k)/count)
        rows.append(dict(M=m,N=n,K=k,weight_buffers=count,total_weight_bytes=count*n*k,samples_us=samples,median_us={name:statistics.median(v) for name,v in samples.items()}))
        print('rotating',m,n,k,rows[-1]['median_us'],flush=True)
    e.dump(e.ROOT/'validation-rotating.json',rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('suite',choices=['smoke','broad','tuned','controls','cold','regressions','blind','boundaries','prospective2'])
    args=parser.parse_args()
    assert e.os.environ['HIP_VISIBLE_DEVICES']=='1' and e.torch.cuda.device_count()==1
    e.torch.set_num_threads(1);e.torch.backends.cuda.matmul.allow_tf32=False
    e.os.sched_setaffinity(0,{max(e.os.sched_getaffinity(0))})
    rules=e.Path(h.__file__).with_name('fp8_per_token_heuristic_rules.py')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [e.Path(e.LINEAR.__file__),e.Path(h.__file__),rules,e.Path(__file__),e.Path(e.SHARED.__file__)]}
    manifest_path=e.ROOT/f'validation-{args.suite}-manifest.json'
    if (e.ROOT/f'validation-{args.suite}.jsonl').exists() and manifest_path.exists():
        previous=e.json.loads(manifest_path.read_text())
        assert previous['source_sha256']==sources, 'Cannot mix results from different sources; archive the old run first'
    maps={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (e.Path(REAL_UTILS).parent/'configs').glob('*fp8_w8a8_per_token*')}
    e.dump(manifest_path,dict(source_sha256=sources,
        plan_sha256=hashlib.sha256((e.ROOT/'plan.json').read_bytes()).hexdigest(),
        tuning_maps_sha256=maps,torch=e.torch.__version__,hip=e.torch.version.hip,
        triton=e.SHARED.triton.__version__,gpu=str(e.torch.cuda.get_device_properties(0)),
        physical_gpu=1,cpu_affinity=sorted(e.os.sched_getaffinity(0)),started=time.time()))
    with e.set_current_vllm_config(e.VllmConfig()),e.torch.inference_mode():
        if args.suite=='controls':controls(args)
        elif args.suite=='cold':cold(args)
        else:matrix(args)
    resources=[]
    for key,value in h._LAUNCH_CACHE.items():
        resources.append(dict(name=value.name,registers=value.n_regs,spills=value.n_spills,
            shared_bytes=value.metadata.shared,warps=value.metadata.num_warps,
            stages=value.metadata.num_stages,launch_key=[str(v) for v in key[3]]))
    if resources:e.dump(e.ROOT/f'compiled-resources-{args.suite}.json',resources)


if __name__=='__main__':main()
