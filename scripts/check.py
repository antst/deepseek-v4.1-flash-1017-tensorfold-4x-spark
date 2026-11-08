"""Offline recipe integrity, numerical reduction and launch-render checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    release = json.loads((ROOT/'release-manifest.json').read_text())
    for name,digest in release.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    tracked = [p for p in ROOT.rglob('*') if p.is_file()
               and p.relative_to(ROOT).parts[0] not in ('local', 'state', '.git')
               and '__pycache__' not in p.parts]
    for p in tracked:
        if p.suffix == '.json':json.loads(p.read_text())
        if p.suffix == '.py':ast.parse(p.read_text(), filename=str(p))
    for p in [ROOT/'README.md', *list((ROOT/'docs').glob('*.md')), ROOT/'engine/README.md', ROOT/'benchmarks/README.md']:
        for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if '://' not in link and not link.startswith('#'):
                assert (p.parent/link.split('#')[0]).exists(), (p, link)
    prov = json.loads((ROOT/'provenance.json').read_text())
    assert hashlib.sha256((ROOT/prov['overlay_path']).read_bytes()).hexdigest() == prov['overlay_sha256']
    assert hashlib.sha256((ROOT/'engine/source-manifest.json').read_bytes()).hexdigest() == prov['source_manifest_sha256']
    assert len(json.loads((ROOT/'engine/fatbins-manifest.json').read_text())) == 50
    bench = ROOT/'benchmarks/client'
    manifest = json.loads((bench/'manifest.json').read_text())
    for name,row in manifest['files'].items():
        data = (bench/name).read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], name
    assert hashlib.sha256((bench/'requests.json').read_bytes()).hexdigest() == '907cc125ed2988868cfc0d7e9e88c589624c09f7e40e0ebbcd1b632b2da09a45'
    waves = json.loads((bench/'requests.json').read_text())
    assert len(waves) == 56 and sum(len(w['requests']) for w in waves) == 258
    assert sum(len(w['requests']) for w in waves if not w['warmup']) == 204
    reduced = module('recipe_reduce', ROOT/'scripts/reduce.py').reduce(json.loads((ROOT/'results/confirmation.json').read_text()))
    assert abs(reduced['geomean_percent']['optimized']-2.3221666205771596) < 1e-10
    assert abs(reduced['geomean_percent']['gu_off']-1.359229874147272) < 1e-10
    assert all(r['gain_percent']['optimized'] > 0 for r in reduced['cells'])
    renderer = module('recipe_launch', ROOT/'scripts/launch-plan.py')
    c = json.loads((ROOT/'config.example.json').read_text())
    c.update(head_image='example.invalid/head:local', follower_image='example.invalid/follower:local')
    rendered = renderer.render(c)
    assert not rendered['executed'] and not rendered['network_or_gpu_actions']
    assert [n['rank'] for n in rendered['nodes']] == [0,1,2,3]
    assert all('SC_TF_ENABLE=0' in n['argv'] and 'TF_DS_DRAFT_GU_COLUMNS=1' in n['argv'] for n in rendered['nodes'])
    c['nodes'][0]['poster_cpus'] = [1]*6
    try:
        renderer.render(c)
    except AssertionError:
        pass
    else:
        raise AssertionError('Duplicate poster cores were accepted')
    subprocess.run(['bash', '-n', str(ROOT/'scripts/build.sh')], check=True)
    # The copied client is tested only in its no-request mode here.
    with tempfile.TemporaryDirectory(prefix='ds1017-corpus-') as td:
        subprocess.run(['python3','-B',str(bench/'workload.py'),'--out',str(Path(td)/'dry'),'--dry-run'], check=True)
    print(json.dumps(dict(offline=True, json_and_python=True, links=True,
                         source_pins=True, corpus=True, all12_timing_cells=True,
                         plan_only=True, duplicate_cpu_rejected=True, shell_syntax=True,
                         docker_executed=False, gpu_initialized=False)))


if __name__ == '__main__':
    main()
