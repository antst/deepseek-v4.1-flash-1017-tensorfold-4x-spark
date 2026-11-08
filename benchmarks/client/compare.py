"""Descriptive TF/SC/TF bracket; returned payload/usage parity is mandatory."""
import argparse,json,math,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('tf_a',type=pathlib.Path);p.add_argument('sc',type=pathlib.Path);p.add_argument('tf_b',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
phases={name:json.load(open(path/'result.json')) for name,path in [('tf_a',a.tf_a),('sc',a.sc),('tf_b',a.tf_b)]}
assert all(v['complete'] for v in phases.values()),'incomplete bracket'
requests={name:{row['id']:row for wave in value['waves'] for row in wave['requests']} for name,value in phases.items()}
ids=set(requests['tf_a']);assert all(set(v)==ids for v in requests.values()),'request sets differ'
differences=[]
for case in sorted(ids):
 rows={k:v[case] for k,v in requests.items()};base=rows['tf_a']
 for name,row in rows.items():
  for field in ['request_sha256','parity_sha256','usage']:
   if row[field]!=base[field]:differences.append(dict(case=case,phase=name,field=field))
cells={name:{(c['kind'],c['concurrency']):c for c in phase['cells']} for name,phase in phases.items()}
keys=set(cells['tf_a']);assert all(set(v)==keys for v in cells.values())
comparisons=[]
for kind,C in sorted(keys):
 values={k:v[(kind,C)]['median_end_to_end_tokens_per_second'] for k,v in cells.items()};fast=max(values['tf_a'],values['tf_b']);gain=(values['sc']/fast-1)*100;drift=abs(values['tf_b']/values['tf_a']-1)*100
 ttft={}
 for name,phase in phases.items():
  ms=sorted(r['ttft_ns']/1e6 for w in phase['waves'] if w['kind']==kind and w['concurrency']==C and not w['warmup'] for r in w['requests'])
  ttft[name]=dict(p50_ms=statistics.median(ms),p99_ms=ms[max(0,math.ceil(len(ms)*.99)-1)],n=len(ms))
 comparisons.append(dict(kind=kind,concurrency=C,median_e2e_tokens_per_second=values,sc_vs_fastest_tf_percent=gain,tf_bracket_drift_percent=drift,request_ttft=ttft,interpretation='drift-limited' if drift>=abs(gain) else 'below2percent-visible-floor' if gain<2 else 'descriptive-visible-threshold-met'))
by_concurrency=[]
for C in sorted({k[1] for k in keys}):
 rows=[c for c in comparisons if c['concurrency']==C];assert len(rows)==2
 ratio=math.prod(1+c['sc_vs_fastest_tf_percent']/100 for c in rows)**.5
 by_concurrency.append(dict(concurrency=C,prose_code_geomean_sc_gain_percent=100*(ratio-1),threshold_met=ratio>=1.02 and all(c['sc_vs_fastest_tf_percent']>=-2 and c['tf_bracket_drift_percent']<abs(c['sc_vs_fastest_tf_percent']) for c in rows)))
report=dict(complete_bracket=True,requests_each_phase=len(ids),exact_request_returned_payload_usage_parity=not differences,differences=differences,cells=comparisons,concurrency=by_concurrency,token_ids_available=all(bool(r['parity']['token_ids']) for phase in requests.values() for r in phase.values()),cached_tokens_field_available=all(r['usage'].get('prompt_tokens_details',{}).get('cached_tokens') is not None for phase in requests.values() for r in phase.values()),limits=['Actual output tokens / wave wall includes TTFT; no per-token GPU clock claim.','TTFT p99 is descriptive at this sample count; three measured waves/cell.','Threshold checks are descriptive. Independent clean engine-close, activation and exact restoration proofs remain required.','No full-vocabulary-logits certificate or automatic adoption.'])
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(parity=not differences,concurrency=by_concurrency),indent=2));raise SystemExit(bool(differences))
