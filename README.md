# DeepSeek-V4.1-Flash-1017 on four DGX Sparks

The native TensorFold recipe deployed on **2026-11-08**, serving the original
[EXL3 4.0 bpw checkpoint](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw)
on four NVIDIA GB10 nodes. The profile uses native TensorFold RDMA, with NCCL
fallbacks, DSpark drafting, a 1,048,576-token per-request context limit and
16 admitted lanes sharing a 10,485,760-row pool.

The latest matched original / optimized / GU-off / original confirmation measured
**+2.32%** geometric-mean end-to-end throughput against the **faster original in
each of 12 cells**. All 12 optimized medians were faster. A previous full-cell
comparison measured +2.69%; those are separate runs, not additive gains.
All **870 reference comparisons** in the latest confirmation matched the native
generated-token hash, response content and usage under their recorded gates.

## Measured throughput

Aggregate end-to-end output tokens/s, **original → optimized**. The denominator
includes first-token latency and completion of the concurrent wave. These are
not decode-only or per-request rates at C4/C8. Thinking is off; T is temperature.

| Workload | C1 | C4 | C8 |
|---|---:|---:|---:|
| Prose, T0 | 85.2 → **87.5** | 165.5 → **166.7** | 199.3 → **206.0** |
| Prose, T1 | 80.4 → **83.2** | 162.9 → **165.8** | 204.4 → **205.9** |
| Code, T0 | 133.4 → **136.6** | 210.9 → **214.4** | 241.8 → **248.5** |
| Code, T1 | 130.6 → **134.2** | 204.2 → **210.1** | 240.5 → **247.4** |

C is concurrent requests. Code-C8 has six measured waves per arm; the other
cells have three. Warmups are retained but excluded from medians. Control drift
and every measured wave remain in the [results](results/confirmation.json).
Small cell gains are descriptive, not individually established significance.
See [method, TTFT and limits](docs/validation.md).

## What this repository provides

- A hash-checked [source reconstruction](engine/README.md) from pinned public
  TensorFold source, including the merged engine and deployed host changes.
- [Exact profile settings](serving/profile.json), portable site configuration
  and a **plan-only** Docker command renderer.
- Original GPU-kernel input hashes and build commands. Weights, native binaries,
  image layers, NCCL libraries and site/cache data are not redistributed.
- The synthetic benchmark corpus/client, complete measured-wave summaries and
  an offline reducer for the published figures.
- [Serving and rollback instructions](docs/serving.md), attribution and explicit
  boundaries between deployed changes and unsuccessful experiments.

This is a source-and-measurement recipe, not a prebuilt image or automatic fleet
installer. Matching kernel/compiler/runtime inputs must be supplied separately;
the local image IDs in provenance are evidence, not downloadable registry names.
Source reconstruction alone does not qualify a different runtime or cluster.

## Deployed changes

The combination retains original expert packing, represented weights, target
arithmetic and embedded math assets. It adds exact local head/candidate handling,
paced routed-down warming, and column partitioning of duplicated drafter work:

- `main_proj`: vocabulary-independent projection split into four output-column
  parts, followed by one all-rank gather in decode absorb. Prompt-fill absorb
  stays on its original route.
- Drafter attention `wo_b` and expert down: output-column halves across pairs;
  existing pair gathers become all-rank gathers. Down output is compacted before
  the gather.
- Drafter gate/up: five/four 128-column blocks across pairs, an existing column
  exchange, then the original down arithmetic. This path is admitted only for
  **20 ≤ R < 64**; smaller and prompt-style windows retain their original route.
- Native TF stage/collect/grain **48/48/2**, four data slots, six distinct host
  poster-core leases per node, and `CUDA_DEVICE_MAX_CONNECTIONS=8`.

These are a measured combination. The percentages of separate experiments must
not be summed. SparkComm was compared separately and is **disabled** in this
serving profile. Its tuning screen found no additional setting worth adopting.
Lazy verification, target `wo_b` local-K overlap, draft-head row compaction,
attention-view splitting and a larger pair-ring capacity are not enabled.

## Reproduce the source and inspect a launch plan

Use a management checkout and a shared build output visible at the same path on
all ranks. Compilation runs on ARM64. Start with no service or fabric changes:

```sh
git clone https://github.com/antst/deepseek-v4.1-flash-1017-tensorfold-4x-spark.git
cd deepseek-v4.1-flash-1017-tensorfold-4x-spark
python3 -B engine/prepare.py --output local/source --plan
python3 -B engine/prepare.py --output local/source --fetch
python3 -B scripts/check.py
python3 -B scripts/reduce.py
cp config.example.json local/config.json
# Fill in your paths, images, SSH hosts, addresses, CPU leases and fabric inputs.
python3 -B scripts/launch-plan.py --config local/config.json --output local/plan.json
```

Read [build inputs and commands](engine/README.md) before compiling. The launch
renderer creates commands only; it does not stop a service, initialize a GPU,
change the fabric, create a container or open a connection to a rank.
The benchmark client makes inference requests only when explicitly run.

## Evidence and boundaries

The confirmation completed 258 benchmark plus 32 functional outputs per arm,
1,160 outputs in total. Functional coverage includes temperature 0/1, 16-way
requests, structured output and long-prompt/cache reuse. Grammar probes are
recorded separately. This establishes preservation of the qualified original
EXL3 recipe on that corpus, not a new general quality score, an AR-equivalence
theorem, a C16 throughput result or sixteen simultaneous 1M histories.

The previous C8 code-T1 TTFT excess was checked with the six-wave confirmation.
Optimized TTFT was **1152.872 ms**, inside the two originals' **1142.668–1158.734
ms** interval. The largest optimized TTFT excess above the slower original over
all 12 cells was 2.355 ms, at code-T1 C4. All spreads are retained; this is not
an assertion that latency effects are absent.

An allocation warning during the private original-B load is retained in the
validation record; its cause and harmlessness were not established. The final
activation passed central API, seeded-output, long retrieval/cache, configuration
and process checks, with no new kernel messages during that activation.

## Credits and license

TensorFold is by [Ash Hart and contributors](https://github.com/ashhart/TensorFold).
The DeepSeek family, exact 2D layout and original recipe are by
[BertholomusAI](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10).
The merged engine includes the credited
[MiaAI-Lab patches](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold).
DeepSeek supplies the model; ExLlamaV3 supplies the EXL3 format/conversion basis.
See [CREDITS.md](CREDITS.md), [NOTICE](NOTICE) and upstream notices in `engine/`.
Recipe and engine modifications are Apache-2.0. Model and third-party assets
retain their own licenses.
