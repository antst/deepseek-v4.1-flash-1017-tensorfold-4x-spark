# Exact source and qualified build inputs

`prepare.py` reconstructs **2,627 build-input files** from public TensorFold
commit `59402086381a42c74d53053443224eec2790b4c5`. It verifies the preimage,
applies `overlay.patch.gz`, then verifies the final compiled-input inventory.
The overlay includes the inherited merged engine and local optimizations; it is
not merely the last 48/48/2 profile change. It includes the original copyright
and attribution files. See `../provenance.json` and `changed-files.json`.

```sh
python3 -B engine/prepare.py --output local/source --plan
python3 -B engine/prepare.py --output local/source --fetch
# Or use an already extracted matching public commit without network access:
python3 -B engine/prepare.py --output local/source --base /path/to/pinned/TensorFold
```

Requires Python 3.12+ and Git. Outputs must be new. No Docker, compiler, model
import or CUDA initialization occurs in reconstruction. A partial output from
a failed run is not a qualified source tree; retain it and choose a new path.

## Inputs not redistributed

1. An ARM64 Linux toolchain with Zig **0.17.0**, CUDA `nvcc`, native build/link
   dependencies and the compatible TensorFold runtime Python dependencies.
2. The **50 original fatbins** in `fatbins-manifest.json`. These are the exact
   inputs retained by the qualified build, not substitutes compiled under
   changed flags. They include kernels unused by this particular model because
   the head binary embeds the wider engine's kernel set.
3. The original model-specific kit: EXL3 dense/grouped/expert cubins, Triton AOT
   assets, grammar support, vision weights and input preparation tools. Start
   from the pinned [original recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10)
   and the model revision in provenance. Keep the kit with its checkpoint;
   do not substitute another checkpoint's Engram or vision-routing bias.
4. The original RDMA fatbin and NCCL shared library matching provenance. Fabric
   setup and library qualification remain separate from building this source.

The qualified compiler image's local ID and the qualified executable hashes
are in provenance. A Docker **local image ID is not a public pull reference**.
This repository does not claim that the private runtime image or complete
binary kernel set can be fetched from a public registry. Supply matching retained
inputs or rebuild from the credited upstream projects and qualify the resulting
new artifact identities on your own hardware. A source match does not establish
binary or numerical identity for changed compiler/kernel inputs.

## CPU-only build

The retained engine build used these commands inside the ARM64 build image,
with source at `/tmp/private-source/source` and pinned inputs at `/fatbins`:

```sh
/opt/zig/zig build test -Dtarget=aarch64-linux-gnu.2.28 \
  -Doptimize=ReleaseFast -Dfatbins=/fatbins -Dsm=121 -j1 --summary all
/opt/zig/zig build native nvme-lanes --prefix /work/build-out \
  -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast \
  -Dfatbins=/fatbins -Dsm=121 -j1 --summary all
```

The retained result was **1,975/1,985 tests passed, 10 skipped**. This publication
checks reconstruction and recipe tooling; it does not rerun the full ARM build.

Compile the two extra, source-provided kernels with the qualified `nvcc`:

```sh
nvcc --fatbin -O3 -std=c++17 -arch=sm_121 \
  zig/kernels/cuda/dsv41/local_head_pack.cu -o /work/local_head_pack.fatbin
nvcc --fatbin -O3 -std=c++17 -arch=sm_121 \
  zig/kernels/cuda/dsv41/exl3_dn_prefetch.cu -o /work/exl3_dn_prefetch.fatbin
```

`bash scripts/build.sh IMAGE SOURCE FATBINS NEW_OUTPUT` wraps this in a bounded CPU-only container with no GPU devices
and no network. It verifies supplied fatbin hashes before invoking Docker.
It does not pull images or perform any serving/fabric operation. A different
toolchain can produce different executable hashes; do not label those artifacts
qualified merely because compilation passes.

The source contains an optional experimental SparkComm adapter inherited from
the comparison build. The deployed profile sets `SC_TF_ENABLE=0`; it neither
requires nor loads a SparkComm shared library. No SparkComm binary is published.
