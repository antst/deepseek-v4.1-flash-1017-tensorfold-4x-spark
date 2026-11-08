#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: $0 BUILD_IMAGE SOURCE_DIR FATBINS_DIR NEW_OUTPUT_DIR" >&2
  exit 2
fi
recipe_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
build_image=$1
source_dir=$(realpath -- "$2")
fatbins_dir=$(realpath -- "$3")
output_dir=$(realpath -m -- "$4")
[[ ! -e "$output_dir" ]] || { echo 'Output already exists' >&2; exit 2; }
python3 - "$recipe_root" "$source_dir" "$fatbins_dir" <<'PY'
from pathlib import Path
import hashlib,json,sys
r,s,f=map(Path,sys.argv[1:])
p=json.loads((r/'provenance.json').read_text())
receipt=json.loads((s/'recipe-source-receipt.json').read_text())
assert receipt['complete'] and receipt['source_manifest_sha256']==p['source_manifest_sha256']
for name,digest in json.loads((r/'engine/source-manifest.json').read_text()).items():
    assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest,name
for name,digest in json.loads((r/'engine/fatbins-manifest.json').read_text()).items():
    assert hashlib.sha256((f/name).read_bytes()).hexdigest()==digest,name
PY
mkdir -p -- "$output_dir"
docker run --pull=never --rm --network none --cpus 2 --memory 6g --memory-swap 6g \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  --mount "type=bind,src=$source_dir,dst=/input,readonly" \
  --mount "type=bind,src=$fatbins_dir,dst=/fatbins,readonly" \
  --mount "type=bind,src=$output_dir,dst=/work" \
  --entrypoint /bin/bash "$build_image" -c '
set -euo pipefail
mkdir -p /tmp/private-source/source
cp -a /input/. /tmp/private-source/source/
cd /tmp/private-source/source
/opt/zig/zig build test -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dfatbins=/fatbins -Dsm=121 -j1 --summary all
/opt/zig/zig build native nvme-lanes --prefix /work/build-out -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dfatbins=/fatbins -Dsm=121 -j1 --summary all
nvcc --fatbin -O3 -std=c++17 -arch=sm_121 zig/kernels/cuda/dsv41/local_head_pack.cu -o /work/local_head_pack.fatbin
nvcc --fatbin -O3 -std=c++17 -arch=sm_121 zig/kernels/cuda/dsv41/exl3_dn_prefetch.cu -o /work/exl3_dn_prefetch.fatbin
'
python3 - "$recipe_root" "$output_dir" <<'PY'
from pathlib import Path
import hashlib,json,sys
r,out=map(Path,sys.argv[1:])
p=json.loads((r/'provenance.json').read_text())
files={'head':'build-out/native/bin/tensorfold-native','follower':'build-out/bin/tf-dsv41-lanes',
       'local_head_pack.fatbin':'local_head_pack.fatbin','exl3_dn_prefetch.fatbin':'exl3_dn_prefetch.fatbin'}
actual={k:hashlib.sha256((out/v).read_bytes()).hexdigest() for k,v in files.items()}
expected=dict(p['qualified_binaries'],**p['qualified_extra_kernels'])
record={'compile_complete':True,'sha256':actual,'matches_published_artifacts':actual==expected,
        'gpu_initialized':False,'inference_qualified_by_this_build':False}
(out/'build-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
