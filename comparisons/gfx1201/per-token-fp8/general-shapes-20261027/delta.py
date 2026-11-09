"""Retime every changed launch choice; retain explicitly equivalent v3 samples."""
import copy,gc,hashlib,json,time
import validate as v
e=v.e;h=v.h
RULE='vllm/model_executor/layers/quantization/utils/fp8_per_token_heuristic_rules.py'
old=e.snapshot('fp8_streaming_v3_rules',e.ROOT/'screen-v3/source'/RULE)
source=e.ROOT.parents[1]/'vllm-fp8-heuristic-refine';frozen=e.json.loads((e.ROOT/'screen-v3/frozen-candidate.json').read_text())
for rel,digest in frozen['sha256'].items():
 if rel!=RULE:assert hashlib.sha256((source/rel).read_bytes()).hexdigest()==digest,rel
sources={str(source/rel):hashlib.sha256((source/rel).read_bytes()).hexdigest() for rel in frozen['sha256']}
sources[str(e.Path(__file__))]=hashlib.sha256(e.Path(__file__).read_bytes()).hexdigest()
for path in [e.Path(v.__file__),e.Path(e.__file__),e.Path(e.SHARED.__file__)]:sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
old_manifest=e.json.loads((e.ROOT/'screen-v3/validation-broad-manifest.json').read_text())
assert hashlib.sha256(e.Path(e.SHARED.__file__).read_bytes()).hexdigest()==old_manifest['source_sha256'][str(e.Path(e.SHARED.__file__))]
with e.set_current_vllm_config(e.VllmConfig()),e.torch.inference_mode():
 e.untuned();e.torch.set_num_threads(1);e.os.sched_setaffinity(0,{max(e.os.sched_getaffinity(0))})
 resource_rows=json.loads((e.ROOT/'screen-v3/compiled-resources-broad.json').read_text())
 for suite in ['broad','blind','boundaries','prospective2']:
  oldpath=e.ROOT/'screen-v3'/f'validation-{suite}.jsonl';oldrows=e.records(oldpath);out=e.ROOT/f'validation-{suite}.jsonl'
  assert not out.exists(),'Archive old final-stage data before rerunning'
  reused=retimed=0
  for r in oldrows:
   m,n,k=r['M'],r['N'],r['K'];choice=h.get_heuristic_config(m,n,k)
   assert r['config']==old.get_heuristic_config(m,n,k)
   if choice==r['config']:
    r=copy.deepcopy(r);r['timing_source']='v3_equivalent_choice';reused+=1
   else:
    previous=r;r=v.run_case(m,n,k,v.kernels(n,k),rounds=5);r['timing_source']='final_retimed';r['previous_choice']=previous['config'];retimed+=1
    if 'split' in previous:r['split']=previous['split']
    print('RETIME',suite,(m,n,k),r['config'],{name:{a:round(x,2) for a,x in val.items() if a in ['graph_gpu_us','batch_wall_us']} for name,val in r['median_us'].items()},flush=True)
    gc.collect();e.torch.cuda.empty_cache()
   r['equivalence_basis']='Backend and GPU launcher byte-identical to v3; exact launch-choice equality required for timing reuse'
   e.append(out,r)
  if suite=='boundaries':
   existing={(r['M'],r['N'],r['K']) for r in e.records(out)}
   for n,k in [(32768,256),(32768,1024),(24560,4096),(24576,6144),(18944,3584),(30720,3840),(3584,18944),(3840,15360),(19968,5632),(24544,6656)]:
    for m in [383,384,385]:
     if (m,n,k) in existing:continue
     r=v.run_case(m,n,k,v.kernels(n,k),rounds=5);r['timing_source']='final_new_boundary';e.append(out,r);retimed+=1
     print('NEW BOUNDARY',(m,n,k),flush=True)
  e.dump(e.ROOT/f'validation-{suite}-manifest.json',dict(source_sha256=sources,reused_source_sha256=frozen['sha256'],reused_from=str(oldpath),reused_records=reused,retimed_records=retimed,policy='Only exact launch-choice matches reuse paired measurements; full final-source correctness runs separately',torch=e.torch.__version__,hip=e.torch.version.hip,triton=e.SHARED.triton.__version__,physical_gpu=1))
  print('DELTA COMPLETE',suite,'reused',reused,'retimed',retimed,flush=True)
 for key,value in h._LAUNCH_CACHE.items():resource_rows.append(dict(name=value.name,registers=value.n_regs,spills=value.n_spills,shared_bytes=value.metadata.shared,warps=value.metadata.num_warps,stages=value.metadata.num_stages,launch_key=[str(v) for v in key[3]],provenance='final_delta'))
 e.dump(e.ROOT/'compiled-resources-broad.json',resource_rows)
