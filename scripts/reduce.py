"""Recompute all-cell timing medians and faster-original geometric means."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def reduce(data):
    cells = {}
    for name, arm in data['arms'].items():
        assert arm['complete']
        waves = [w for w in arm['waves'] if not w['warmup'] and w['kind'] != 'conformance']
        out = {}
        for kind in ('prose-t0', 'prose-t1', 'code-t0', 'code-t1'):
            for c in (1, 4, 8):
                rows = [w for w in waves if w['kind'] == kind and w['concurrency'] == c]
                assert len(rows) == (6 if kind.startswith('code-') and c == 8 else 3)
                ttft = [w['median_ttft_ms'] for w in rows]
                out[(kind, c)] = dict(tps=statistics.median(w['end_to_end_tokens_per_second'] for w in rows),
                                      ttft_ms=statistics.median(ttft), ttft_range_ms=max(ttft)-min(ttft))
        cells[name] = out
    result = []
    for key in sorted(cells['original_a']):
        original = max(cells[n][key]['tps'] for n in ('original_a', 'original_b'))
        r = dict(kind=key[0], concurrency=key[1], faster_original_tps=original,
                 arms={n:v[key] for n,v in cells.items()})
        r['gain_percent'] = {n:(cells[n][key]['tps']/original-1)*100 for n in ('optimized', 'gu_off')}
        result.append(r)
    geo = {n:(math.exp(statistics.mean(math.log(1+r['gain_percent'][n]/100) for r in result))-1)*100
           for n in ('optimized', 'gu_off')}
    return dict(geomean_percent=geo, cells=result)


if __name__ == '__main__':
    print(json.dumps(reduce(json.loads((ROOT/'results/confirmation.json').read_text())), indent=2))
