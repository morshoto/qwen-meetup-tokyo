# Findings log

This file contains **measured project findings only**.

The current log contains only bounded harness/pilot observations. A result is
not promoted to a model capability finding until its backend, scorer, task
catalog, sample denominator, and raw hash are explicit.

Do not copy hypotheses, expected plots, vendor benchmarks, community anecdotes, or illustrative numbers into this file as if they were our results.

## Status

```text
exp_001: fixture smoke measured (harness-only); real-model baseline pending
exp_002: calibrated real-model pilot measured (Q8_0 x 8,192 x 30 tasks x 1 capability run); full matrix pending
exp_003: legacy expected.v1 smoke audit only; calibrated interaction pending
exp_004: not yet measured
exp_005: not yet measured
```

The measured exp_002 pilot is documented in
[`pilot-v002-report.md`](../experiments/exp_002-quantization_llama_cpp_gguf/results/processed/pilot-v002-report.md).

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
