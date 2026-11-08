# Measurement and validation record

The final confirmation ran four fresh private processes in order:

1. Exact original production binary/environment/CPU placement.
2. Optimized native TF, gate/up column split on (`optimized`).
3. Same optimized binary and environment, only gate/up column split off (`gu_off`).
4. Exact original repeat.

The two candidate arms kept six unique poster-core leases, four native TF data
slots, 48 stage CTAs / 48 collect cap / grain 2, and eight CUDA connections.
Both originals retained their actual original settings. This is a combination
versus the actual original, not an isolated attribution to any one knob.

## Rates and latency

Each wave rate is actual total output tokens divided by the concurrent wave's
host wall time, including TTFT and final usage. A cell uses the median wave rate.
The baseline is the **maximum of the two original cell medians**, chosen
independently in each cell. The overall result is the geometric mean of the 12
candidate/baseline rate ratios. It is not a pooled-token or decode-only rate.

Run `python3 -B scripts/reduce.py` to reproduce **+2.3221666205771596% optimized**
and **+1.359229874147272% GU-off** from published wave summaries. Optimized gains
by concurrency are C1 +2.783%, C4 +1.756%, C8 +2.431%; every optimized cell is
positive, from +0.700% to +3.417%. Several small individual gains are within the
observed drift scale and should not be treated as separately proven speedups.

Code-C8 uses six measured waves per arm; other cells use three. The synthetic
258-request corpus includes warmups and conformance; 204 requests are measured.
Thinking is off, temperatures are 0/1, and ordinary completions are capped at
256 tokens. No wave or outlier was removed. SSE arrivals are chunks, not GPU
token clocks. All measured prompts reported zero cached tokens; cache reuse was
tested separately rather than inferred from these timing rows.

TTFT is HTTP origin to first content. Each cell uses the median of wave-median
TTFTs. Sorted request TTFT ranks, where inspected, are descriptive ranks and
must not be called service or arrival order. Per-job timers with different
origins were not subtracted to invent a GPU or scheduling explanation.

An earlier three-wave run showed about +4.95% code-T1 C8 TTFT for the candidate.
The focused native stage/grain screen did not isolate a consistent fix. The
final six-wave confirmation placed optimized TTFT at 1152.872 ms within the
original 1142.668–1158.734 ms bracket; GU-off was 1161.762 ms. This does not prove
that a latency effect is absent. All wave values and ranges remain available.

## Exactness and coverage

Four arms × (258 benchmark +32 functional outputs) = **1,160 outputs**;
three arms compared to original A = **870 exact reference comparisons**.
Native generated-token SHA, returned content and usage were checked; the
benchmark cache-difference exception was disabled. Separate functional choices
covered seeded sampling, C16, mixed requests and long prefill/cache reuse.
Grammar API probes were present in addition to those counts.

The head retained **50 original embedded fatbins**, the follower **8**, all
byte-identical to the original numerical control. The candidate includes two
additional externally loaded head/warm kernels; those have separate hashes.
The new build ran 1,975/1,985 tests with 10 skips before qualification.

These checks preserve the qualified EXL3 numerical recipe on the retained
corpus. They are not a new model-quality score, full-logit comparison,
AR-equivalence theorem, general width-invariance claim, C16 throughput test or
full-capacity 1M-context campaign.

## Diagnostics and alternatives

One rank-zero `mem_desc.c:1359 NV_ERR_NO_MEMORY` warning occurred during the
private original-B startup at 13:04:35 UTC. Every arm loaded, completed its
requests and closed successfully. The warning's cause and harmlessness were
not established; no driver reset or retry was performed. It remains in the
published outcome. The final activation's kernel delta was empty on all ranks.

SparkComm with the same engine combination did not beat native TF in the prior
full-cell test. Later exact-shape tuning retained the current 48/48/2/K4 policy:
collect-8, stage-8 and grain-4 changes had mixed or negative loaded results,
including variations in inactive A/A controls. No isolated microbenchmark win
was promoted or extrapolated into an inference gain.

The proposed 6MiB pair-ring capacity, original-H attention views, lazy verifier,
and optional target local-K overlap are outside this serving profile. No gain
from those proposals is included here.
