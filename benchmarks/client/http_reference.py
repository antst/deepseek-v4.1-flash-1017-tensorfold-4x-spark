#!/usr/bin/env python3
"""Finite direct-backend A/B workload. No model/proxy/service mutations."""
import argparse,concurrent.futures,hashlib,json,pathlib,threading,time,urllib.parse,urllib.request

def workload(model,maximum):
    waves=[]
    for concurrency in [n for n in (1,2,4,8,16) if n<=maximum]:
        for repeat in range(3):
            batch=[]
            for rank in range(concurrency):
                key=f'sc-ag-20261104-c{concurrency}-w{repeat}-r{rank}'
                task=('Explain how a CPU cache works, using a sequence of short numbered sentences.' if rank%2==0 else 'Write a Python merge-sort implementation with comments, then explain its time complexity.')
                body=dict(model=model,messages=[dict(role='user',content=key+'\n'+task)],temperature=0,top_p=1,seed=1234,max_tokens=128,stream=True,stream_options=dict(include_usage=True),chat_template_kwargs=dict(enable_thinking=False))
                batch.append(dict(id=key,body=body))
            for cache in ('first','repeat'):
                waves.append(dict(concurrency=concurrency,repeat=repeat,cache=cache,requests=batch))
    return waves

def request(base,case,barrier,timeout=60):
    body=json.dumps(case['body'],sort_keys=True,separators=(',',':')).encode()
    barrier.wait();start=time.perf_counter_ns();events=[];parts=[];reasoning=[];ids=[];probabilities=[];usage=None;finish=None;first=None;last=None;pending=[]
    req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        while True:
            line=response.readline()
            now=time.perf_counter_ns()
            if now-start>timeout*1e9:raise TimeoutError(case['id'])
            if not line:break
            text=line.decode().rstrip('\r\n')
            if text.startswith('data:'):pending.append(text[5:].lstrip())
            if text or not pending:continue
            data='\n'.join(pending);pending=[]
            if data=='[DONE]':break
            obj=json.loads(data);events.append(dict(ns=now-start,value=obj))
            if obj.get('usage') is not None:usage=obj['usage']
            for choice in obj.get('choices',[]):
                delta=choice.get('delta',{})
                content=delta.get('content') or ''
                thought=delta.get('reasoning_content') or delta.get('reasoning') or ''
                if content or thought:
                    if first is None:first=now
                    last=now
                parts.append(content);reasoning.append(thought)
                ids.extend(delta.get('token_ids') or [])
                lp=choice.get('logprobs') or delta.get('logprobs')
                if lp is not None:probabilities.append(lp)
                if choice.get('finish_reason') is not None:finish=choice['finish_reason']
    end=time.perf_counter_ns()
    if usage is None or not isinstance(usage.get('completion_tokens'),int):raise RuntimeError('missing completion usage: '+case['id'])
    parity=dict(text=''.join(parts),reasoning=''.join(reasoning),token_ids=ids,logprobs=probabilities,completion_tokens=usage['completion_tokens'],finish_reason=finish)
    return dict(id=case['id'],request_sha256=hashlib.sha256(body).hexdigest(),parity=parity,parity_sha256=hashlib.sha256(json.dumps(parity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),wall_ns=end-start,ttft_ns=None if first is None else first-start,content_window_ns=None if first is None else last-first,usage=usage,events=events)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--model',required=True);p.add_argument('--maximum-concurrency',type=int,choices=[8,16],required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    parsed=urllib.parse.urlparse(a.base)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost'):p.error('use a localhost forward to the private backend; central proxy is excluded')
    a.out.mkdir(parents=True,exist_ok=False)
    waves=workload(a.model,a.maximum_concurrency)
    (a.out/'workload.json').write_text(json.dumps(waves,indent=2)+'\n')
    if a.dry_run:
        print(json.dumps(dict(dry_run=True,requests=sum(len(w['requests']) for w in waves),waves=len(waves),max_tokens=128)));return
    deadline=time.monotonic()+360;results=[]
    for index,wave in enumerate(waves):
        if time.monotonic()>deadline:raise TimeoutError('six-minute client phase deadline')
        barrier=threading.Barrier(wave['concurrency']+1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=wave['concurrency']) as pool:
            pending=[pool.submit(request,a.base,r,barrier) for r in wave['requests']]
            start=time.perf_counter_ns();barrier.wait()
            rows=[f.result(timeout=70) for f in pending];elapsed=time.perf_counter_ns()-start
        result=dict(wave=index,concurrency=wave['concurrency'],repeat=wave['repeat'],cache=wave['cache'],elapsed_ns=elapsed,completion_tokens=sum(r['usage']['completion_tokens'] for r in rows),requests=rows)
        result['end_to_end_tokens_per_second']=result['completion_tokens']*1e9/elapsed
        (a.out/f'wave{index:02d}.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
        print(json.dumps({k:result[k] for k in ('wave','concurrency','cache','completion_tokens','end_to_end_tokens_per_second')}),flush=True)
    (a.out/'result.json').write_text(json.dumps(dict(complete=True,waves=results,counting='actual completion tokens / wave wall including TTFT; SSE content timestamps are chunk arrivals, not a token-by-token generation clock'),indent=2)+'\n')

if __name__=='__main__':main()
