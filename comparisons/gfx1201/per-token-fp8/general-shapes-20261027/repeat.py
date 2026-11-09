"""Repeat the union of material regression screens with seven paired rounds."""
import gc,hashlib,time
import validate as v
e=v.e
with e.set_current_vllm_config(e.VllmConfig()),e.torch.inference_mode():
 e.untuned();e.torch.set_num_threads(1);e.os.sched_setaffinity(0,{max(e.os.sched_getaffinity(0))})
 selected={}
 for suite in ['broad','blind','boundaries','prospective2']:
  for r in e.records(e.ROOT/f'validation-{suite}.jsonl'):
   a,t=r['median_us']['after'],r['median_us']['torch']
   if any(a[key]/t[key]>ratio and a[key]-t[key]>absolute for key,ratio,absolute in [('graph_gpu_us',1.03,.5),('batch_wall_us',1.05,1),('sync_wall_us',1.05,1)]):selected.setdefault((r['M'],r['N'],r['K']),[]).append(suite)
 e.dump(e.ROOT/'residual-repeat-manifest.json',dict(selected=[dict(M=m,N=n,K=k,suites=suites) for (m,n,k),suites in selected.items()],started=time.time(),sources={str(e.Path(p)):hashlib.sha256(e.Path(p).read_bytes()).hexdigest() for p in [e.LINEAR.__file__,v.h.__file__,e.Path(v.h.__file__).with_name('fp8_per_token_heuristic_rules.py')]}))
 done={(r['M'],r['N'],r['K']) for r in e.records(e.ROOT/'residual-repeat.jsonl')}
 for i,((m,n,k),suites) in enumerate(sorted(selected.items())):
  if (m,n,k) in done:continue
  r=v.run_case(m,n,k,v.kernels(n,k),rounds=7);r['screen_suites']=suites;e.append(e.ROOT/'residual-repeat.jsonl',r)
  print(i+1,len(selected),(m,n,k),{name:{k:round(x,2) for k,x in v.items() if k in ['graph_gpu_us','batch_wall_us']} for name,v in r['median_us'].items()},flush=True)
  gc.collect();e.torch.cuda.empty_cache()
 if not selected:(e.ROOT/'residual-repeat.jsonl').touch()
