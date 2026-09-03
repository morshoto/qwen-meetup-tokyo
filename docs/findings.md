# Findings log

This file contains **measured project findings only**.

The current log contains only bounded harness/pilot observations. A result is
not promoted to a model capability finding until its backend, scorer, task
catalog, sample denominator, and raw hash are explicit.

Do not copy hypotheses, expected plots, vendor benchmarks, community anecdotes, or illustrative numbers into this file as if they were our results.

## Status

```text
exp_001: calibrated real-model reduced baseline plus bounded Q8 feasibility probe measured; full context matrix pending
exp_002: calibrated real-model capability matrix measured (Q8_0/Q6_K/Q5_K_M/Q4_K_M x 8,192/32,768 x 30 tasks); separate timing matrix incomplete
exp_003: calibrated matched pilot measured (Q8_0/Q4_K_M x 8,192/32,768 x p50 x 30 tasks); full matrix pending
exp_004: fixed-policy real-model recheck measured (Q8_0/Q4_K_M x trajectory 1/4/8/16/32 x p50 x 10 tasks x 3 repeats); full position matrix pending
exp_005: not yet measured
```

The measured exp_002 pilot is documented in
[`pilot-v002-report.md`](../experiments/exp_002-quantization_llama_cpp_gguf/results/processed/pilot-v002-report.md).

## 2026-09-03 — exp_001 bounded feasibility: 64K useful, larger probes time out

Experiments:
- exp_001 feasibility follow-up

Result artifacts:
- raw: `experiments/exp_001-context_measurement/results/raw/feasibility-trials.jsonl`
- manifest: `experiments/exp_001-context_measurement/results/manifests/feasibility.json`
- processed: `experiments/exp_001-context_measurement/results/processed/feasibility-summary.csv`, `experiments/exp_001-context_measurement/results/processed/feasibility-aggregate.csv`
- raw SHA-256: `7b58322e06d149bd438e7d9a0dbaf78ffe5a4163f90db4c9006e01cc50af551e`

Conditions:
- model/runtime: Qwen/Qwen3.8-27B, pinned Q8_0 GGUF through llama.cpp on the recorded arm64/macOS environment
- contexts: 65,536, 131,072, and 262,144 target tokens; evidence position 50%
- task catalog: `data/tasks/core.v002.jsonl`, three explicitly selected task IDs (one per family)
- one child process per task, one greedy run, fixed 900-second hard timeout; all 9 attempts retained
- scorer: `calibrated.v1`; timeout records carry no correctness score, but retain scorer/version and provenance metadata

Measured result (conservative all-task classification):

| target context | attempted | completed | answer-bearing | classification |
| ---: | ---: | ---: | ---: | --- |
| 65,536 | 3 | 3 | 3 | `accepted_and_useful` |
| 131,072 | 3 | 0 | 0 | `operational_failure` (3 timeouts) |
| 262,144 | 3 | 0 | 0 | `operational_failure` (3 timeouts) |

The completed 65,536-token trials had TTFT of 782.8–786.2 seconds and peak
RSS of approximately 33.3 GB. The timeout trials reached approximately 37.6 GB
RSS at 131,072 and 46.2 GB at 262,144. “Useful” here means all three selected
outputs were answer-bearing under the calibrated scorer; exact and format-valid
correctness were lower for two of the three outputs.

Interpretation:

In this pinned environment and fixed protocol, 65,536 tokens completed with
answer-bearing results, while 131,072 and 262,144 did not complete within the
900-second wall-clock budget. This is evidence about an environment- and
protocol-bounded feasibility boundary, not a general effective-context limit,
model capability limit, or position-bias result. The probe uses one position,
one repeat, and three tasks, so it cannot estimate task or position variance.

Next check:

Treat the result as a bounded systems/capability probe in the presentation.
Run the full declared position matrix or repository pilot only with a separately
budgeted measurement plan; do not label the timeout boundary as “262K impossible”.

## 2026-09-02 — exp_002 full capability matrix: footprint shrinks, quality remains task-shaped

Experiments:
- exp_002

Inputs:
- raw: `experiments/exp_002-quantization_llama_cpp_gguf/results/raw/full-capability.jsonl`
- processed: `experiments/exp_002-quantization_llama_cpp_gguf/results/processed/summary.csv`
- control manifest: `experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.full.json`
- raw SHA-256: `0d222dfe0ff801c93d39f8d24d367dd013c02a860f982bf711dab0b77107c092`

Conditions:
- model/runtime: Qwen/Qwen3.8-27B through the same recorded llama.cpp/GGUF path
- quantization: Q8_0, Q6_K, Q5_K_M, and Q4_K_M
- contexts: 8,192 and 32,768 input tokens; evidence position 50%
- task catalog: `data/tasks/core.v002.jsonl`, 30 independent tasks
- one greedy capability run per task/cell; 240/240 attempted trials completed with zero runtime errors
- scorer: `calibrated.v1`

Measured result (end-to-end success over 60 attempted trials per variant):

| quantization | artifact size | end-to-end success | answer-bearing | format-valid |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 29.05 GB | 32/60 (53.3%) | 60/60 | 42/60 |
| Q6_K | 22.43 GB | 32/60 (53.3%) | 60/60 | 42/60 |
| Q5_K_M | 19.54 GB | 32/60 (53.3%) | 60/60 | 42/60 |
| Q4_K_M | 16.81 GB | 27/60 (45.0%) | 59/60 | 37/60 |

Interpretation:

The full capability matrix shows a 42.1% Q8_0-to-Q4_K_M artifact-size
reduction. In this catalog and protocol, Q8_0/Q6_K/Q5_K_M have the same
aggregate end-to-end count, while Q4_K_M is lower; this is a bounded
descriptive observation, not a general quantization-quality ranking.

Alternative explanations / limitations:

The capability matrix uses one greedy run per independent task and a synthetic
retrieval catalog. Answer-bearing outputs remain much higher than format-valid
end-to-end success, so scorer/output-format behavior contributes to the gap.
The separate timing matrix has only 474/1,200 attempted probes; stream TTFT and
throughput values are proxies, and sequential sampled RSS is not a definitive
cross-variant memory comparison.

Next check:

Complete the separate 3–5-repeat timing matrix only if systems-cost claims are
needed; keep capability and timing conclusions separate.

## 2026-09-03 — exp_002 Q4/Q8 practical equivalence is metric-specific

Experiments:
- exp_002 full capability matrix

Result artifacts:
- analysis: `experiments/exp_002-quantization_llama_cpp_gguf/equivalence.py`
- report: `experiments/exp_002-quantization_llama_cpp_gguf/results/processed/q4-q8-equivalence.json`
- CSV: `experiments/exp_002-quantization_llama_cpp_gguf/results/processed/q4-q8-equivalence.csv`
- source raw SHA-256: `0d222dfe0ff801c93d39f8d24d367dd013c02a860f982bf711dab0b77107c092`

Method:
- 60 matched Q8_0/Q4_K_M pairs (30 task IDs × 2 context lengths), joined by
  task ID, target context, and evidence position;
- candidate-minus-reference paired bootstrap, 95% confidence, 100,000 resamples
  in the committed report, and a pre-registered ±10 percentage-point practical
  equivalence margin;
- exact, answer-bearing, format-valid, and end-to-end outcomes analyzed
  separately; all selected trials were completed.

Measured result:

| metric | Q8_0 | Q4_K_M | observed Q4−Q8 | 95% paired CI | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| answer-bearing | 60/60 | 59/60 | −1.7pp | −5.0pp to 0.0pp | equivalent within ±10pp |
| exact / end-to-end | 32/60 | 27/60 | −8.3pp | −16.7pp to −1.7pp | inconclusive |
| format-valid | 42/60 | 37/60 | −8.3pp | −16.7pp to −1.7pp | inconclusive |

Interpretation:

The data support practical equivalence only for answer-bearing correctness
under the declared ±10pp margin. They do not support “Q4 and Q8 have fully
equal quality”: the exact/end-to-end and format-valid intervals cross the
practical margin. This is a bounded equivalence result, not a general model
ranking or a claim about other prompts, tasks, sampling policies, or contexts.

Next check:

If a stronger quality claim is needed, expand the independent task catalog and
pre-register the same metric-specific equivalence test before collecting new
measurements. Do not reinterpret the current one-greedy-run matrix as a test
of sampling variance.

## 2026-09-02 — exp_001 reduced real-model baseline: task-specific baseline limits

Experiments:
- exp_001

Result manifest:
- `experiments/exp_001-context_measurement/results/manifests/pilot-matched.json`
- raw SHA-256: `9f3c567e94051b4bed96716f4fedc4b0a242af961014162f6e34b6a44f0ebb59`

Conditions:
- model/runtime: Qwen/Qwen3.8-27B through llama.cpp/GGUF, Q8_0, with `n_ctx=33088`
- task catalog: `data/tasks/core.v002.jsonl`, 30 independent tasks (10 per family)
- contexts: 8,192 and 32,768 input tokens
- evidence position: 50% only
- capability runs: one per independent task/cell; 60 attempted trials, all completed
- scorer: `calibrated.v1`

Measured result (end-to-end success, `n=10` per task family/cell):

| task family | 8K | 32K |
| --- | ---: | ---: |
| literal | 2/10 | 6/10 |
| semantic | 4/10 | 3/10 |
| multi-hop | 8/10 | 9/10 |

Answer-bearing correctness was 10/10 in every cell. Format validity was 2/10
and 6/10 for literal, 9/10 and 8/10 for semantic, and 8/10 and 9/10 for
multi-hop at 8K and 32K respectively. Median stream-derived TTFT was 75.0 s
at 8K and 339.5 s at 32K.

Interpretation:

The 8K baseline is below the configured 0.80 gate for literal and semantic
retrieval, so their effective-context values are `baseline_limited`. Multi-hop
passes the gate and remains `right_censored` at 32K; no finite effective-context
number is established. The measured input-processing cost increases sharply at
32K under the recorded stream-TTFT proxy.

Alternative explanations / limitations:

This is a reduced baseline: one evidence position, two context lengths, one
greedy run per independent task, and no 64K/128K cells. The position-gap output
is explicitly `insufficient_data`, and answer-bearing, format-valid, exact, and
end-to-end metrics remain separate. The first 131K-allocation run with 59/60
decode failures is preserved separately as an instrumentation failure, not a
model finding.

Next check:

Run the declared positions and 64K/128K cells only after the baseline task set
and runtime budget are confirmed; do not promote this reduced pilot to a full
effective-context or position-bias conclusion.

## 2026-09-03 — exp_004 fixed-policy recheck: output failures do not recur

Experiments:
- exp_004

Result artifacts:
- raw: `experiments/exp_004-agent_context_growth/results/raw/recheck-trials.jsonl`
- processed: `experiments/exp_004-agent_context_growth/results/processed/recheck-summary.csv`
- manifest: `experiments/exp_004-agent_context_growth/results/manifests/recheck.json`
- figures: `experiments/exp_004-agent_context_growth/results/figures/reliability-vs-trajectory-length-recheck.png`, `experiments/exp_004-agent_context_growth/results/figures/reliability-vs-critical-position-recheck.png`
- raw SHA-256: `0f11903a95f55e5714b1a8f218fcf07da7bc123bf7b7abe030c4e93bc618aa24`

Conditions:
- model/runtime: Qwen/Qwen3.8-27B through llama.cpp/GGUF on the recorded arm64/macOS environment
- quantization: Q8_0 and Q4_K_M
- agent catalog: `data/tasks/agent.v002.jsonl`, 10 independent tasks
- trajectory lengths: 1, 4, 8, 16, and 32 tool observations; length 1 is the zero-distractor one-turn control
- critical-information position: 50% requested position (actual position is 0.0 for the length-1 control)
- fixed output policy: one JSON object, 128 maximum new tokens, markdown disallowed
- fixed retry policy: three action attempts, zero backoff
- three greedy repeats per task/cell; 300/300 attempted trials completed

Measured result (final task success; `n=30` per cell):

| quantization | traj 1 control | traj 4 | traj 8 | traj 16 | traj 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| Q4_K_M | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |

Critical-fact reuse and final-task success were 300/300. Tool-call validity was
300/300, with zero planning errors, retries, invalid outputs, or runtime errors.
All 30 variant × length cells have three repeats and identical 100% success.
In the preceding reduced pilot, all 30 `invalid_output` trials reached the old
64-token completion limit; in this recheck the largest final response was 109
tokens under the fixed 128-token budget.

Interpretation:

Under the fixed 128-token JSON/retry protocol, this run shows no trajectory-length
degradation and no Q8/Q4 difference for this synthetic task catalog. The earlier
pilot's parse failures do not recur, so those failures are consistent with an
output-budget/protocol effect; this comparison is not a causal ablation because
the earlier run used a different protocol.

Alternative explanations / limitations:

The recheck uses one requested position, greedy decoding, and a synthetic
state-tracking catalog. It does not establish a general absence of Lost-in-the-Agent,
position effects, or repository-task transfer. The three repeats are deterministic
greedy repeats, so they demonstrate protocol stability rather than sampling variance.

Next check:

If position effects are required, run the declared 5-position matrix under this same
fixed policy. Otherwise use this recheck as the bounded agent result and retain the
earlier pilot as a protocol-diagnostic comparison.

## 2026-09-02 — exp_004 reduced agent pilot: runtime stable, tool planning is not

Experiments:
- exp_004

Result manifest:
- `experiments/exp_004-agent_context_growth/results/manifests/fast-matched.json`
- raw SHA-256: `0075408def4b86a2b94fbd80f7a5a9c4034cc80e2e6169376a2695ef2532012d`

Conditions:
- model/runtime: Qwen/Qwen3.8-27B through llama.cpp/GGUF on the recorded arm64/macOS environment
- quantization: Q8_0 and Q4_K_M
- agent catalog: `data/tasks/agent.v002.jsonl`, 10 independent tasks
- trajectory lengths: 4, 8, 16, and 32 tool observations
- critical-information position: 50% only
- one greedy run per task/cell; 80 attempted trials, all cells have `n=10`
  (50 `completed`, 30 `invalid_output`)
- source manifest: the committed exp_003 matched pilot, SHA-256 recorded in the manifest

Measured result (final task success; `n=10` per cell):

| quantization | traj 4 | traj 8 | traj 16 | traj 32 |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 6/10 | 8/10 | 5/10 | 9/10 |
| Q4_K_M | 7/10 | 7/10 | 4/10 | 4/10 |

Critical-fact reuse matched final-task success in every cell. Tool-call validity
was 100% for calls that were emitted, while 30/80 attempted trials ended as
`invalid_output` after `ActionParseError` retries and were classified as
`tool_planning`; 50/80 ended successfully. No runtime errors were observed.

Interpretation:

The pilot separates runtime stability from agent completion: a valid tool call
and a discovered fact do not guarantee a successful final state. Neither
quantization showed a monotonic trajectory-length trend in this run (Q8:
60%/80%/50%/90%; Q4: 70%/70%/40%/40%).

Alternative explanations / limitations:

This is a reduced pilot with one critical-information position, one greedy run
per independent task, and no repeated positions or 5/20-repeat variance
estimate. The 30 planning failures are model/output observations, not runtime
failures. The result does not establish a general Lost-in-the-Agent law,
position bias, or repository-task transfer.

Next check:

Repeat the same task and environment across declared critical positions and
additional independent runs before making a trajectory or quantization claim.

## 2026-09-02 — exp_003 matched pilot: a descriptive context-dependent gap

Experiments:
- exp_003

Result manifests:
- `experiments/exp_003-context_x_quantization/results/manifests/fast-matched.json`
- raw SHA-256: `2bcc89b3e7f14d3bb71947cef4c2f1cd1a992ed6df6ba815a62dc6491cba9212`

Conditions:
- model/runtime: Qwen/Qwen3.8-27B through llama.cpp/GGUF on the recorded arm64/macOS environment
- quantization: Q8_0 and Q4_K_M
- task catalog: `data/tasks/core.v002.jsonl`, 30 independent tasks (10 per family)
- contexts: 8,192 and 32,768 input tokens
- evidence position: 50%
- capability runs: one per task/cell; 120 attempted trials, all completed
- scorer: `calibrated.v1`

Measured result (end-to-end success, `n=10` per task family/cell):

| task family | Q8 8K | Q8 32K | Q4 8K | Q4 32K |
| --- | ---: | ---: | ---: | ---: |
| literal | 2/10 | 6/10 | 2/10 | 4/10 |
| semantic | 4/10 | 3/10 | 3/10 | 3/10 |
| multi-hop | 8/10 | 9/10 | 7/10 | 8/10 |

Answer-bearing correctness was 10/10 in every cell except Q4 multi-hop at 32K
(9/10). Format validity ranged from 2/10 to 9/10 across these cells. The
descriptive Q4-minus-Q8 end-to-end gap was classified as `context_dependent`
for literal retrieval (0.0 at 8K to 0.2 at 32K), and
`approximately_constant` for semantic retrieval (0.1 to 0.0) and multi-hop
(0.1 to 0.1) under the configured 0.10 tolerance.

Interpretation:

In this matched pilot, quantization did not produce one uniform capability
ordering. The literal gap widened at 32K, while the other two task families
showed no larger-than-tolerance change in the Q4/Q8 gap.

Alternative explanations / limitations:

This is a reduced pilot: one evidence position, two context lengths, one greedy
run per independent task, and no 64K/128K cells. The interaction labels are
descriptive rather than significance tests. Exact, answer-bearing, and format
metrics remain separate, and the synthetic retrieval catalog does not establish
repository-task transfer.

Next check:

Repeat the calibrated matched design across all declared context lengths and
positions before promoting the result to a full interaction conclusion.

## 2026-08-31 — exp_002 pilot: answer-bearing output exceeds exact output

Experiments:
- exp_002

Result manifests:
- `experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json`
- raw SHA-256: `84eab3da1656d15df100e3fd7382ca3ab44cfaf83e32f1cd1a123b061a62ade6`

Conditions:
- model/runtime: Qwen/Qwen3.8-27B through llama.cpp/GGUF
- quantization: Q8_0
- task subset: 30 independent synthetic retrieval tasks (10 per family)
- context: 8,192 input tokens
- sample size: 30 capability runs

Measured result:

- exact calibrated answers: 14/30 (46.7%);
- answer-bearing outputs: 30/30 (100%);
- format-valid outputs: 19/30 (63.3%);
- median stream-derived TTFT: 71.1740 s.

The resolved GGUF artifacts are Q8_0 29.05 GB, Q6_K 22.43 GB, Q5_K_M
19.54 GB, and Q4_K_M 16.81 GB; these are artifact-footprint measurements, not
accuracy results. Peak-RSS values from the historical sequential run are not
used for a per-quantization memory claim.

Interpretation:

The pilot demonstrates why exact correctness, answer-bearing correctness, and
format validity must remain separate. It does not establish Q8 quality, a
quantization ranking, or an effective context limit.

Alternative explanations / limitations:

The catalog is a controlled synthetic retrieval stress test, not a general
reasoning benchmark. Greedy repeats are not independent samples; the capability
denominator is the independent task count. Timing values are stream proxies,
and the full Q8/Q6/Q5/Q4 matrix has not run.

Next check:

Run the calibrated task catalog across all declared variants and contexts, with
the same scorer and manifest, before making a cross-quantization claim.

## 2026-08-31 — exp_001 and exp_003 boundary audit

The committed exp_001 result is 180/180 fixture trials across 18 cells and is
harness-only (raw SHA-256
`e493461a7bf8f552b7a73792578daee1dffeea1a314dfc5a699c012e82847585`; the
artifact was regenerated after pinning the sampling seed and output budget).
The
committed exp_003 result is a 48-trial legacy `expected.v1` llama.cpp smoke
artifact (raw SHA-256
`f97e23332b01039210c83af6e0dd61956ad0af656723a83436e65a720ed5c95d`); its flat
scores are an exploratory audit and are classified as `insufficient_data`, not
as a context × quantization interaction.
Neither artifact is Qwen capability evidence.

## Finding template

Use one section per meaningful observation.

```markdown
## YYYY-MM-DD — Short finding title

Experiments:
- exp_XXX

Result manifests:
- path / hash / durable reference

Conditions:
- model artifact:
- runtime:
- quantization:
- task subset:
- context range:
- sample size:

Measured result:
State the observation numerically and neutrally.

Evidence:
Link to notebook section, processed table, and figure.

Interpretation:
What we think the result means.

Alternative explanations / limitations:
What else could explain it or where it may not generalize.

Next check:
What experiment/analysis would increase confidence.
```

## Rules

1. A finding must be traceable to committed or durably referenced result data.
2. Include sample count.
3. Distinguish measured value from interpretation.
4. If a finding changes after a bug fix, do not silently rewrite history. Note the correction and why.
5. Keep null results when they resolve an important hypothesis.
6. Do not generalize beyond the tested model/runtime/hardware without evidence.

## Example wording style

Good:

> In exp_003, Q4 accuracy on semantic-retrieval tasks decreased by X percentage points between 8K and 128K, compared with Y points for Q8 (`n=...` per cell). The matched-cell gap widened with context.

Bad:

> Q4 destroys long-context reasoning.

Good:

> Under the tested runtime and task set, median prefill time increased from X to Y seconds between 32K and 128K while decode throughput changed by only Z%.

Bad:

> Long context is too slow locally.
