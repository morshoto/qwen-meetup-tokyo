# Findings log

This file contains **measured project findings only**.

The current log contains only bounded harness/pilot observations. A result is
not promoted to a model capability finding until its backend, scorer, task
catalog, sample denominator, and raw hash are explicit.

Do not copy hypotheses, expected plots, vendor benchmarks, community anecdotes, or illustrative numbers into this file as if they were our results.

## Status

```text
exp_001: calibrated real-model reduced baseline measured (Q8_0 x 8,192/32,768 x p50 x 30 tasks); full matrix pending
exp_002: calibrated real-model pilot measured (Q8_0 x 8,192 x 30 tasks x 1 capability run); full matrix pending
exp_003: calibrated matched pilot measured (Q8_0/Q4_K_M x 8,192/32,768 x p50 x 30 tasks); full matrix pending
exp_004: not yet measured
exp_005: not yet measured
```

The measured exp_002 pilot is documented in
[`pilot-v002-report.md`](../experiments/exp_002-quantization_llama_cpp_gguf/results/processed/pilot-v002-report.md).

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
