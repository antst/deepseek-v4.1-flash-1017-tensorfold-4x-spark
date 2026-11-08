"""Reconstruct the qualified build inputs without compiling or loading CUDA."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROV = json.loads((ROOT/'provenance.json').read_text())


def inventory(root):
    rows = {}
    for name in PROV['compiled_input_roots']:
        p = root/name
        if not p.exists():
            continue
        for f in ([p] if p.is_file() else sorted(p.rglob('*'))):
            if f.is_symlink():
                raise ValueError('Symlink in build inputs: '+str(f))
            if f.is_file():
                rows[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return rows


def verify(root, manifest):
    actual = inventory(root)
    if actual != manifest:
        different = sorted(k for k in set(actual)|set(manifest) if actual.get(k) != manifest.get(k))
        raise ValueError('Build-input drift: '+repr(different[:20]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    source = ap.add_mutually_exclusive_group()
    source.add_argument('--base', type=Path, help='Extracted pinned public engine')
    source.add_argument('--fetch', action='store_true')
    ap.add_argument('--plan', action='store_true')
    a = ap.parse_args()
    patch = ROOT/PROV['overlay_path']
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == PROV['overlay_sha256']
    patch_bytes = gzip.decompress(patch.read_bytes())
    assert hashlib.sha256(patch_bytes).hexdigest() == PROV['overlay_uncompressed_sha256']
    assert hashlib.sha256((ROOT/'engine/source-manifest.json').read_bytes()).hexdigest() == PROV['source_manifest_sha256']
    if a.output.exists():
        ap.error('Output must not already exist')
    if a.plan:
        print(json.dumps(dict(output=str(a.output), base=PROV['base_engine'],
                             patch_sha256=PROV['overlay_sha256'],
                             compiled_input_roots=PROV['compiled_input_roots'],
                             network=bool(a.fetch), compile=False, gpu=False), indent=2))
        return
    if not a.base and not a.fetch:
        ap.error('Supply --base or explicit --fetch')
    original = json.loads((ROOT/'engine/base-manifest.json').read_text())
    final = json.loads((ROOT/'engine/source-manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='ds1017-source-') as td:
        temp = Path(td)
        expanded_patch = temp/'overlay.patch'
        expanded_patch.write_bytes(patch_bytes)
        if a.fetch:
            archive = temp/'base.tar.gz'
            with urllib.request.urlopen(PROV['base_engine']['archive_url'], timeout=120) as response, archive.open('wb') as out:
                shutil.copyfileobj(response, out)
            with tarfile.open(archive) as t:
                t.extractall(temp/'base', filter='data')
            entries = list((temp/'base').iterdir())
            if len(entries) != 1 or not entries[0].is_dir():
                raise ValueError('Unexpected upstream archive layout')
            base = entries[0]
        else:
            base = a.base.resolve()
        verify(base, original)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, a.output, ignore=shutil.ignore_patterns('.git', '__pycache__', '.zig-cache', 'zig-out'))
        # Do not let a parent recipe checkout make Git silently filter these
        # paths as outside its current subdirectory.
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(a.output.resolve().parent))
        subprocess.run(['git', 'apply', '--no-index', '--check', '--whitespace=nowarn', str(expanded_patch)], cwd=a.output, env=env, check=True)
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=nowarn', str(expanded_patch)], cwd=a.output, env=env, check=True)
        verify(a.output, final)
        receipt = dict(complete=True, commit=PROV['base_engine']['commit'],
                       files=len(final), source_manifest_sha256=PROV['source_manifest_sha256'],
                       overlay_sha256=PROV['overlay_sha256'], compiled=False, gpu_initialized=False)
        (a.output/'recipe-source-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
