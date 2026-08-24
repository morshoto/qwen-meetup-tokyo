# Findings log

This file contains **measured project findings only**.

At project initialization there are no measured findings yet.

Do not copy hypotheses, expected plots, vendor benchmarks, community anecdotes, or illustrative numbers into this file as if they were our results.

## Status

```text
exp_001: not yet measured
exp_002: not yet measured
exp_003: not yet measured
exp_004: not yet measured
exp_005: not yet measured
```

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
