# Analysis plan

**Issue:** #12 — Build cross-experiment error taxonomy and analysis views

This document defines the required analytical outputs so each experiment notebook produces compatible evidence rather than unrelated charts.

## 1. Principles

1. Analysis must preserve trial-level traceability.
2. Plot measured data only; hypothetical plots belong in design docs, not results notebooks.
3. Show sample count and uncertainty where relevant.
4. Distinguish execution failures from incorrect model answers.
5. Use paired comparisons when the same task instances run across model/quantization conditions.
6. Negative/null findings remain in the report.

## 2. Notebook structure

Every `analysis.ipynb` should follow approximately:

```text
0. Experiment metadata / manifest
1. Load data
2. Validate schema and coverage
3. Missing/error cell audit
4. Primary metrics
5. Required plots
6. Statistical/paired comparisons
7. Failure examples
8. Candidate findings
9. Limitations
10. Export processed tables/figures
```

The notebook should fail loudly if required columns or expected cells are missing rather than silently plotting incomplete data.

## 3. exp_001 required analysis

### Coverage table

Rows:

- task type;
- context length;
- evidence position.

Report:

- planned `n`;
- completed `n`;
- OOM/runtime failures;
- scored `n`.

### Figure A — position curve

X:

```text
normalized evidence position
```

Y:

```text
accuracy
```

Separate line/facet by context length and task type.

Question:

Does performance vary systematically with position, and does that dependence strengthen at longer contexts?

### Figure B — context degradation

X: actual/target context tokens.  
Y: accuracy.

Separate series by task type and optionally selected positions.

### Figure C — effective context

Show effective context by task type under the agreed alpha threshold.

Include sensitivity for alternative alpha values if conclusions are threshold-sensitive.

### Figure D — systems cost vs context

At minimum plot:

- TTFT or prefill time vs context tokens;
- peak memory vs context tokens;
- decode throughput vs context if it changes materially.

## 4. exp_002 required analysis

### Figure A — capability vs memory

Scatter or Pareto plot:

```text
x = peak memory or artifact size
y = task accuracy
point = quantization artifact
```

### Figure B — throughput vs memory

Separate prefill and decode. Never combine them into one unnamed `tok/s` metric.

### Table — quantization provenance

For each artifact:

- method/format;
- source/revision;
- artifact size;
- peak memory;
- task score;
- TTFT;
- prefill t/s;
- decode t/s.

### Recommendation

Choose which variants continue to exp_003 using measured trade-offs. Preserve the rationale in the experiment README.

## 5. exp_003 required analysis

This is the central interaction experiment.

### Figure A — context × quantization heatmap

Rows: context lengths.  
Columns: quantization variants.  
Cell: task accuracy or relative degradation.

Produce separate heatmaps by task type if combining them would hide effects.

### Figure B — position × context heatmap per quantization

Rows: context lengths.  
Columns: evidence positions.  
Cell: accuracy.

The presentation can place Q8 and Q4 side-by-side if this reveals a pattern.

### Figure C — quantization gap vs context

For matched task instances:

```text
gap(C) = accuracy_high_precision(C) - accuracy_low_precision(C)
```

Plot the gap over context length with uncertainty.

This directly tests H3.

### Figure D — effective context by quantization/task

A compact summary chart/table.

## 6. exp_004 required analysis

### Primary outcome

Final task success as a function of:

- accumulated input/history tokens;
- number of agent turns/tool calls;
- relative age/position of critical observation.

### Lost in the Agent figure

X:

```text
relative position/age of critical observation
```

Y:

```text
probability of correctly reusing it / final success
```

Separate by history length and/or quantization where sample size permits.

### Trajectory efficiency

Report:

- model calls;
- tool calls;
- repeated calls;
- input tokens processed cumulatively;
- wall-clock time.

### State consistency

Track explicit contradictions of previously discovered critical facts where a deterministic reference is available.

## 7. exp_005 required analysis

Compare context strategies on matched repository tasks:

```text
curated
neighborhood
broad
```

### Outcomes

- correct target file/module;
- correct diagnosis;
- test-passing completion;
- total tokens processed;
- tool calls/files opened;
- task time.

### Main question

Does increasing available repository context improve success enough to justify its systems/attention cost?

## 8. Cross-experiment effective-context table

The final report should aim to produce a table resembling this structure, populated only with measured values:

| Configuration | Literal | Semantic | Multi-hop | Agent | Repository |
|---|---:|---:|---:|---:|---:|
| Higher precision | TBD | TBD | TBD | TBD | TBD |
| Mid precision | TBD | TBD | TBD | TBD | TBD |
| Lower precision | TBD | TBD | TBD | TBD | TBD |

Not every cell must be measurable. Mark unavailable rather than extrapolating.

## 9. Error taxonomy

Initial shared taxonomy:

### Retrieval

Relevant information is present but the model does not surface/use it.

### Recognition

The evidence is surfaced but the model does not recognize its relevance/meaning.

### Synthesis / reasoning

Required evidence is available but combined incorrectly.

### Planning

The model chooses an unproductive high-level sequence of actions.

### Tool selection

Wrong tool for the current objective.

### Tool arguments

Correct tool but invalid/wrong arguments.

### State tracking

The model forgets, contradicts, or fails to reuse an earlier observation.

### Verification

The model stops/claims success without checking the machine-checkable condition.

### Repetition / loop

The model repeats substantially equivalent actions without progress.

### Context overload

The task succeeds under a smaller/curated context but fails when irrelevant context is added, without another obvious cause.

### Runtime/system

OOM, timeout, backend error, malformed runtime output. Keep separate from cognitive/model errors.

## 10. Classification procedure

Each classification record should contain:

```text
trial_id
primary_category
secondary_categories (optional)
classification_method = automatic | manual
reviewer/version
notes
```

Do not force ambiguous cases into one category. Allow:

```text
primary_category = unclear
```

Manual error analysis should use a blinded condition label where practical so reviewers do not know which quantization produced the trajectory.

## 11. Statistical guidance

### Binary success

Use Wilson confidence intervals.

### Pairwise quantization comparisons

Because the same task instances should be reused, report paired outcomes such as:

```text
both correct
high only correct
low only correct
both wrong
```

A McNemar-style paired test may be useful if sample size supports it, but effect size and raw counts matter more than chasing a p-value.

### Multiple cells

Do not perform dozens of unplanned significance tests and highlight whichever crosses 0.05. The primary comparisons should be stated in the experiment design.

### Runtime metrics

Use median/IQR or percentile bands. Keep model-load/cold-start metrics separate from warm generation.

## 12. Figure standards

Every presentation figure should show or make available:

- experiment ID;
- metric definition;
- sample count;
- uncertainty where appropriate;
- condition labels with full quantization meaning in caption/notes;
- source notebook/result manifest.

Avoid:

- truncated axes that exaggerate tiny differences without justification;
- mixing percentages and proportions;
- unlabeled hypothetical values;
- smoothing that hides sparse context checkpoints.

## 13. Candidate cross-experiment findings template

When adding a finding to `docs/findings.md`, include:

```markdown
## YYYY-MM-DD — Finding title

Experiments:

Evidence:

Measured result:

Interpretation:

Alternative explanations / caveats:

Next check:
```

Keep “measured result” and “interpretation” separate.
