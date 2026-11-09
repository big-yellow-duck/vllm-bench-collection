"""Full final-source output checks across every unique evaluation shape."""
import gc,time
import validate as v
e=v.e
with e.set_current_vllm_config(e.VllmConfig()),e.torch.inference_mode():
 e.untuned();e.torch.set_num_threads(1);e.os.sched_setaffinity(0,{max(e.os.sched_getaffinity(0))})
 cases=sorted({(r['N'],r['K'],r['M']) for suite in ['broad','blind','boundaries','prospective2'] for r in e.records(e.ROOT/f'validation-{suite}.jsonl')})
 out=e.ROOT/'full-correctness.jsonl';done={(r['N'],r['K'],r['M']) for r in e.records(out)};pair=None
 for i,(n,k,m) in enumerate(cases):
  if (n,k,m) in done:continue
  if pair!=(n,k):objs=v.kernels(n,k);pair=(n,k);gc.collect();e.torch.cuda.empty_cache()
  x=e.inputs(m,n,k);kw=dict(A=x[0],B=x[1],As=x[2],Bs=x[3],bias=None,out_dtype=e.torch.bfloat16,output_shape=[m,n])
  ref=objs['torch'].apply_scaled_mm(**kw);actual=objs['after'].apply_scaled_mm(**kw)
  v.check_full(actual,ref);e.reference_check(actual,x)
  e.append(out,dict(M=m,N=n,K=k,status='pass',check='Full Torch comparison, finite, sampled FP32',config=v.h.get_heuristic_config(m,n,k)))
  if (i+1)%50==0:print('CORRECTNESS',i+1,len(cases),flush=True)
  del x,kw,ref,actual
 e.dump(e.ROOT/'full-correctness-status.json',dict(status='pass',cases=len(cases),completed=time.time()))
