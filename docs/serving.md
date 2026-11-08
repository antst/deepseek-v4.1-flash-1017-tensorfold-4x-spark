# Prepare, qualify, serve and roll back

The profile is physical TP4 with logical TP2 × two column pairs. Rank order is
part of the recipe: `g=2*p+r`, row partner `g^1`, column partner `g^2`. Use the
four-node layout from the original recipe. The measured site used prepared
switched f0 paths, both Socket Direct roots, GID index 3 and traffic class 104;
this is not a switchless-ring bandwidth claim. This repository changes no
interfaces, routes, drivers, clocks, switches or fabric settings.

## Site inputs

Copy `config.example.json` into ignored `local/config.json`. Supply:

- Your ARM64 runtime images. The head also needs the original Python helpers
  for multimodal input; compatible native/CUDA, grammar and kit dependencies
  remain required. See the credited upstream build instructions.
- Read-only EXL3 model/served view, original-checkpoint Engram inputs, the matching
  vision-routing bias, token map, rank-cache directories and model-specific kit.
- Shared paths to the built head/follower and two extra kernels, and the pinned
  RDMA fatbin/NCCL library. These are inputs, not files shipped in this recipe.
- Rank-zero address, four SSH destinations and six **distinct** available poster
  CPU cores on each node. Choosing the same six core numbers on different hosts
  is fine; reusing a core for two posters on one host is not the measured profile.
- A **new empty host-local cache root** for this build. Keep previous roots and
  their engine identities for rollback. Each rank uses its own subdirectory.

The example addresses and paths are placeholders. Fill them in before rendering:

```sh
python3 -B scripts/launch-plan.py --config local/config.json --output local/plan.json
# Where all artifact paths are visible, add --verify-artifacts to check the
# qualified executable/kernel/library hashes before producing another new plan.
```

The output contains Docker argv arrays, not shell commands. It reports whether
artifact hashes were checked. Review it on each destination and verify actual
mounts and library hashes there. The renderer does not grant hardware admission
or prove rank-cache, runtime, network or model qualification.

## Qualification

Run one model stack at a time. Drain existing model and image requests using
their actual service owners, stop the old stack cleanly, and verify that its GPU
contexts and fabric users have exited before a private load. Keep the original
containers, configuration and cache roots; do not remove them for a test.

Use a private API port, e.g. 18888, while retaining the same source, weights,
pool, lanes, runtime environment, ranks and libraries. Start followers before the
head. The native profile has `SC_TF_ENABLE=0`; no SparkComm DSO is required.
The renderer disables site-specific central registration. Operators with a
registry can bind their own existing registration configuration separately.

The retained synthetic corpus can be inspected without inference:

```sh
python3 -B benchmarks/client/workload.py --out local/corpus-check --dry-run
```

For actual inference, forward the private backend to localhost:19088 and run the
same client against each arm, using **new output directories** and the same
prompt-cache starting condition. Do not point a private benchmark at a busy
production proxy. The published client accepts localhost/private ports only.
Use `benchmarks/client/compare.py --help` for response comparison. The latest
qualification also ran separate functional C16, structured-output, sampling and
long-prompt/cache checks; the timing corpus alone does not reproduce those
additional receipts.

Before activation, verify every rank's loaded binary, environment, rank order,
model/kit and context/pool settings; whole-token/output/usage comparisons;
concurrent and long-prefix operation; healthy close and final context absence.
Do not infer correctness merely from a faster kernel or throughput result.

## Activation and recovery

The recorded activation first genuinely restored and checked the original
deployment after qualification. It then staged the selected artifacts at a
permanent path, prepared new empty caches, withdrew admission, drained requests,
stopped the old deployment, and started the new followers and head. Central API,
seeded token hash, long retrieval and cache reuse, process/configuration identity
and unchanged image workers passed before the new serving record was published.

The selected build was activated at **2026-11-08 13:14:54 UTC**. Old containers
and old cache roots remain available for rollback. Site-specific container IDs,
registry leases and controller scripts are intentionally not published.

On failure: stop the failed new stack, verify all its GPU/fabric users have
exited, restart the exact old followers and head with their original mounts and
cache roots, verify health and registration, and resume admission. Never reload
the old model over a still-running candidate. This recipe does not perform that
transaction automatically.

The configured 1M context limit, 16 lanes and 10,485,760-row shared pool are
distinct quantities. They do not promise sixteen independent 1M histories.
