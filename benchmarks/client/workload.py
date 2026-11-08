"""Direct private1017 transport comparison; derived pinned prose/code cases."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.parse

ROOT=Path(__file__).resolve().parent

def digest(data):
    return hashlib.sha256(data).hexdigest()

def inputs():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,row in manifest['files'].items():
        data=(ROOT/name).read_bytes()
        assert len(data)==row['bytes'] and digest(data)==row['sha256'],name
    waves=json.loads((ROOT/'requests.json').read_text())
    spec=importlib.util.spec_from_file_location('matched_http_reference',ROOT/'http_reference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return manifest,waves,module.request

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://127.0.0.1:19088')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--maximum-concurrency',type=int,choices=[8,16],default=16)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();u=urllib.parse.urlparse(a.base)
    if u.scheme!='http' or u.hostname not in ('127.0.0.1','localhost') or u.port not in (18888,19088) or u.path not in ('','/') or u.query or u.fragment:
        p.error('owned private1017 endpoint only')
    manifest,waves,request=inputs()
    waves=[w for w in waves if w['concurrency']<=a.maximum_concurrency]
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'requests.json').write_bytes((ROOT/'requests.json').read_bytes())
    (a.out/'manifest.json').write_bytes((ROOT/'manifest.json').read_bytes())
    if a.dry_run:
        print(json.dumps(dict(dry_run=True,waves=len(waves),
            requests=sum(len(w['requests']) for w in waves),
            measured_requests=sum(len(w['requests']) for w in waves if not w['warmup']),
            maximum_output_tokens=sum(r['body']['max_tokens'] for w in waves for r in w['requests']))))
        return
    admission=time.monotonic()+600;results=[]
    for index,wave in enumerate(waves):
        if time.monotonic()>admission:raise TimeoutError('ten-minute new-wave admission bound')
        barrier=threading.Barrier(wave['concurrency']+1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=wave['concurrency']) as pool:
            pending=[pool.submit(request,a.base,case,barrier,90) for case in wave['requests']]
            before=time.perf_counter_ns();wall_before=time.time_ns()
            barrier.wait(timeout=5)
            rows=[f.result(timeout=95) for f in pending]
            after=time.perf_counter_ns();wall_after=time.time_ns()
        for row,case in zip(rows,wave['requests']):
            assert row['parity']['finish_reason'] in ('stop','length'),row['id']
            assert 0<row['usage']['completion_tokens']<=case['body']['max_tokens'],row['id']
            assert not row['parity']['reasoning'],row['id']
            assert not row['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0),row['id']
            assert not any(event['value'].get('error') for event in row['events']),row['id']
        result={k:wave[k] for k in ('id','kind','concurrency','warmup','repeat')}
        result.update(wave=index,host_start_ns=before,host_end_ns=after,
            wall_start_ns=wall_before,wall_end_ns=wall_after,requests=rows,
            output_tokens=sum(r['usage']['completion_tokens'] for r in rows))
        result['end_to_end_tokens_per_second']=result['output_tokens']*1e9/(after-before)
        result['median_ttft_ms']=statistics.median(r['ttft_ns']/1e6 for r in rows)
        result['cached_tokens']=[r['usage'].get('prompt_tokens_details',{}).get('cached_tokens') for r in rows]
        (a.out/f'wave{index:02d}.json').write_text(json.dumps(result,indent=2)+'\n')
        results.append(result)
        print(json.dumps({k:result[k] for k in ('wave','kind','concurrency','warmup','output_tokens','end_to_end_tokens_per_second','median_ttft_ms','cached_tokens')}),flush=True)
    cells=[]
    for kind in ('prose-t0','prose-t1','code-t0','code-t1'):
        for concurrency in (c for c in (1,4,8) if c<=a.maximum_concurrency):
            measured=[w for w in results if w['kind']==kind and w['concurrency']==concurrency and not w['warmup']]
            assert len(measured)==(6 if concurrency==8 and kind.startswith('code-') else 3)
            cells.append(dict(kind=kind,concurrency=concurrency,
                median_end_to_end_tokens_per_second=statistics.median(w['end_to_end_tokens_per_second'] for w in measured),
                median_wave_ttft_ms=statistics.median(w['median_ttft_ms'] for w in measured),
                waves=[w['wave'] for w in measured]))
    output=dict(complete=True,waves=results,cells=cells,
        counting='Actual output tokens / concurrent-wave wall including TTFT and final usage; TTFT measured to first content chunk; warmups excluded from cell summaries.',
        limits=['SSE arrivals are chunks, not individual GPU token clocks.',
                'Cached-token counts are recorded; short prompts are not called cold without cached_tokens=0.',
                'Returned payload/usage parity is checked across arms separately; actual native token SHA256 is compared; full logits are not exposed.',
                'No model, transport, proxy, resource or fabric changes are made by this client.'])
    (a.out/'result.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':main()
