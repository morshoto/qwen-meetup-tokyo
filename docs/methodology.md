# Methodology

**Issue:** #3 — Define methodology, controls, metrics, and reproducibility rules

This document defines shared evaluation rules across experiments. Values marked **proposed default** may be revised after smoke/pilot runs, but revisions must be recorded and should not differ silently across quantizations or context conditions.

## 1. Experimental unit

The default experimental unit is one **trial**:

```text
(model artifact + runtime config)
×
(task instance)
×
(context construction)
×
(sampling config)
→
(response + telemetry + score)
```

Every plotted point must be traceable to underlying trial records.

## 2. General controlled variables

Hold these constant inside a matched comparison unless the experiment explicitly studies them:

- model family/checkpoint;
- tokenizer revision;
- inference runtime and version;
- runtime kernel/backend options;
- chat template/system prompt;
- prompt/task wording;
- context filler/corpus generation procedure;
- target evidence placement algorithm;
- generation temperature and sampling parameters;
- maximum output tokens;
- reasoning-effort mode if exposed;
- tool schema for agent tasks;
- machine and OS;
- warm/cold-cache policy;
- background workload policy.

If any controlled variable differs because a runtime cannot support the same behavior, record the difference and avoid attributing the entire effect to the nominal independent variable.

## 3. Sampling policy

### Proposed default

For tasks with a deterministic expected answer:

- temperature: `0.0` where the runtime supports deterministic greedy decoding;
- top-p/top-k: runtime defaults compatible with greedy decoding;
- maximum generation length: task-specific, kept constant across matched cells.

Deterministic decoding does **not** remove the need for multiple task instances. Variation should come from independently generated benchmark instances rather than repeatedly asking the exact same prompt where possible.

For agent tasks, if deterministic decoding leads to brittle single-trajectory behavior, introduce a controlled nonzero sampling condition only as a separate analysis.

### Seed policy

Keep these seeds distinct in the run manifest:

- **fixture seed:** generates or selects the task, filler, corpus, and initial
  environment state;
- **generation seed:** controls stochastic decoding when sampling is enabled;
- **orchestration seed:** controls randomized tool scheduling or retry behavior,
  if the harness has any.

The primary deterministic run uses `temperature: 0.0` and still records all
three seed fields. A repeated stochastic run must use a declared list of
generation seeds and keep fixture/task IDs fixed so variation is attributable
to decoding rather than a different problem. Never silently reseed a failed
trial during analysis.

## 4. Trial count / phases

Do not launch the full long-context matrix before validating the runner and task construction.

### Phase A — smoke

Purpose: correctness only.

- 1–2 instances per selected cell;
- short and one long context;
- one beginning, middle, and end evidence position;
- verify token counts, evidence placement, scoring, logging, and runtime stability.

Smoke results are not presentation findings.

### Phase B — pilot

Purpose: estimate runtime, failure modes, and variance.

**Proposed default:** at least 5 independent task instances per representative cell.

Use the pilot to decide whether target context lengths are practical and whether the benchmark is too easy/hard.

### Phase C — main run

**Proposed default:** target at least 20 independent task instances per cell for binary accuracy where practical.

For extremely expensive 128K–262K conditions, a smaller sample may be acceptable if resource constraints are documented and uncertainty is shown. Do not hide unequal sample counts.

## 5. Context lengths

Initial target grid from the project plan:

```text
8K
32K
64K
128K
262K (or highest practical native-context condition)
```

Possible additional diagnostic point:

```text
16K
```

Rules:

- define length in **tokenizer tokens**, not characters/words;
- store actual final prompt token count in every trial;
- aim for a narrow tolerance around target length;
- do not count generated output in the input-context target;
- include chat-template/system/tool-schema overhead in the recorded actual token count;
- for agent trajectories, report context length at each model invocation.

If 262K is infeasible on the selected local runtime/hardware, record the highest completed condition and the reason for exclusion.

## 6. Evidence positions

Initial normalized evidence positions:

```text
0.05
0.25
0.50
0.75
0.95
```

Position means the target location of the **relevant evidence span** within the controlled context body.

Implementation requirements:

- place evidence by token offset, not string/character offset;
- record requested and actual normalized positions;
- do not let prompt wrappers systematically move “5%” to a different effective location without accounting for them;
- for multi-hop tasks, record each evidence span's position separately.

For the position-bias analysis, define `A_edge(C)` as the mean of the matched
beginning and end conditions at context length `C`, then calculate:

```text
Delta_pos(C) = A_edge(C) - A_middle(C)
```

Bootstrap matched task instances to estimate `Delta_pos(C)` at each length and
the predeclared difference-in-differences
`Delta_pos(C_long) - Delta_pos(C_short)`, or fit an equivalent context-length ×
position interaction. The first contrast tests a position effect; the second
tests whether that effect grows with context length. Do not use overlapping
marginal confidence intervals as the decision rule.

## 7. Task ladder

The core benchmark increases semantic difficulty deliberately.

### T1 — literal retrieval

Directly retrieve a fact with high lexical overlap between evidence and question.

Purpose: verify basic accessibility of context.

### T2 — semantic retrieval

Answer requires recognizing a concept/paraphrase rather than locating an identical key string.

Purpose: separate lexical lookup from meaningful interpretation.

### T3 — multi-hop reasoning

Answer requires combining multiple evidence spans distributed through context.

Purpose: test composition and weakest-link effects.

### T4 — repository reasoning

Answer or action depends on information across files/modules.

Purpose: realism validation.

### T5 — agent trajectory

The model must use information discovered during previous tool interactions.

Purpose: test state tracking and useful accumulated context.

## 8. Effective context window

The project distinguishes **maximum accepted context** from **effective context**.

For task family `T`, define a short-context baseline accuracy:

```text
A_baseline(T) = accuracy at baseline context condition
```

### Baseline validity and absolute reporting

The relative breakpoint is meaningful only when the task family is already
usable at the short-context reference. The primary gate is at least `0.80`
accuracy at the 8,192-token reference-precision baseline, measured over the
predeclared scored instances. Declare a different gate before collecting a
phase if the task family requires it.

If a family misses the gate, classify it as **baseline-limited** and do not
report or rank a relative effective-context breakpoint for that family. Still
report its absolute accuracy curve, successes/attempted scored trials, and
runtime-failure status at every tested length. Relative breakpoints never
replace the absolute curves.

For the primary report, order the tested context lengths from shortest to
longest and define the first sustained threshold crossing:

```text
C_break(T, q, alpha) = smallest tested C_i where
                       A(T, q, C_i) < alpha * A_baseline(T, q)
                       and the next tested length is also below the threshold
```

The effective context is the tested length immediately before `C_break`. If
the first tested length crosses the threshold, effective context is reported as
`< first tested length`. If no crossing is observed, report the result as
right-censored (`>= largest tested length`) rather than claiming an unlimited
window. If there is only one tested length after the crossing, mark the
crossing as provisional and show the non-sustained drop separately.

Here `q` is the complete model/runtime/quantization configuration, not just a
bit-width label. Use the same task instances and scorer for the baseline and
each compared condition.

### Proposed default

`alpha = 0.90`.

This threshold must be sensitivity-checked. Report whether conclusions change for reasonable alternatives such as `0.85` or `0.95`.

Because positional bias exists, compute at least two versions:

1. **mean-position effective context** — aggregate across tested positions;
2. **worst-position effective context** — uses the lowest position-conditioned performance.

The latter is more conservative and may be more relevant to uncontrolled real prompts.

## 9. Accuracy / task metrics

### Exact/task accuracy

Binary correctness under a deterministic scorer where possible. Store each
trial as `0` or `1`; report `successes / attempted scored trials`, `n`, and the
percentage in `[0, 100]`.

### Semantic score

Use only when exact matching is inappropriate. Prefer explicit answer
normalization or structured expected concepts before using an LLM judge. Record
the scorer name/version and score range; do not mix a 0–1 normalized score with
an unnormalized judge score in one summary.

### Completion rate

For agent/coding tasks:

```text
successful final states / attempted tasks
```

Report as a rate in `[0, 1]` and percentage in `[0, 100]`. An attempted task
whose runtime fails or times out remains in the denominator and has its status
recorded separately.

A “successful” implementation task should preferably be machine-checkable by tests.

### Tool-call validity

Fraction of tool calls that are syntactically/schema-valid:

```text
valid tool calls / all emitted tool calls
```

Report the numerator, denominator, and rate in `[0, 1]`. A task with no
emitted tool call is not silently treated as 100% valid; report it as
`not_applicable` for this metric.

### Useful tool-call rate

Optional/manual metric: whether a valid call plausibly advances the task. Keep
separate from syntactic validity, document the rubric, and report the number of
calls judged.

### Recovery rate

Among trajectories containing a defined error or failed action, fraction that
return to a productive path and ultimately satisfy the task success condition.

```text
recovered error trajectories / trajectories with a defined recoverable error
```

Report the denominator and use `not_applicable` when no qualifying error was
observed.

### Steps to solution

Count model turns and tool calls separately. Report both as integer counts per
trajectory; do not call a tool call a model turn unless the harness defines
those events as identical.

## 10. Systems metrics

Every runtime should expose as many of these as reliably possible.

### Time to first token (TTFT)

Wall-clock time in seconds from generation request start until the first
generated token becomes available.

### Prefill throughput

```text
input tokens processed / prefill seconds
```

Report the result in input tokens per second.

If the runtime cannot expose prefill independently, report prompt-evaluation timing or clearly mark unavailable.

### Decode throughput

```text
generated tokens / decode seconds
```

Report the result in generated tokens per second.

Do not report “tok/s” without labeling input/prefill vs output/decode.

### Total model-call time

End-to-end generation call duration in seconds, excluding model load unless the
run explicitly labels it as a cold-start measurement.

### Total task time

For agents:

```text
sum(model inference + tools + orchestration overhead)
```

Report wall-clock seconds from task start until the success/failure terminal
state. Include tool and orchestration time for agent tasks; report model-call
time separately.

### Peak memory

Record peak bytes using the best available process/system metric and state
exactly how it is measured. On unified-memory systems, “VRAM” may be
misleading; use peak resident/unified memory terminology appropriate to the
measurement source. Do not report a number without the measurement source and
sampling interval.

### Model artifact size

Record bytes on disk separately from peak memory. Include the artifact format
and whether auxiliary files such as a vision projector are included.

### Energy / power

Stretch metric. Only report if the measurement method is stable and documented.
Prefer task-level energy in joules or watt-hours over a single instantaneous
watt reading; record the sampling interval, device scope, and whether idle
power is subtracted.

## 11. Quantization comparisons

Nominal labels such as Q4/Q6/Q8 are insufficient by themselves.

Record:

- quantization algorithm/format;
- effective bits if known;
- group/block size if relevant;
- which tensors remain higher precision;
- runtime/backend;
- source artifact or quantization command;
- cache/KV precision;
- model revision.

Do not compare two artifacts from different methods and call the difference “the effect of 4 vs 8 bits” without qualification.

## 12. Warm-up and repetition

### Proposed default

For latency measurements:

- run one warm-up request that is excluded from summary statistics;
- define whether model load time is included separately;
- repeat timing measurements enough to estimate dispersion;
- avoid mixing cold-load and warm-generation measurements.

Agent task total time should include real tool time; low-level decode benchmarks may exclude model load.

## 13. Statistical treatment

For binary task success:

- report `n`, successes, and percentage;
- include a binomial confidence interval (Wilson preferred over naive normal intervals at small n).

For latency/throughput:

- report median and a spread metric (IQR or percentile range);
- mean may be included but should not be the only statistic if distributions are skewed.

For matched generated task instances across quantizations:

- preserve instance IDs so paired comparisons are possible;
- prefer paired analysis over comparing unrelated sample sets.

Do not overclaim small percentage differences when intervals overlap heavily or sample counts are tiny.

## 14. Failure recording

A failed runtime call is data.

Record statuses such as:

```text
success
wrong_answer
invalid_output
runtime_error
out_of_memory
timeout
scorer_error
cancelled
```

Do not silently drop OOM or timeout cells. They define part of the practical frontier.

## 15. Raw versus processed result provenance

Raw trial records are the immutable evidence layer. Processed summaries,
figures, and notebooks are derived layers and must never overwrite raw data.

### Raw records

Store one JSONL record per trial, including the complete status, model output
or a durable reference to it, scores, telemetry, and required run metadata.
Keep failed calls, OOMs, timeouts, and scorer errors as records. A rerun gets a
new `run_id` and must not replace an earlier record with the same task/config
combination.

### Processed records

Every processed table or figure must identify:

- the input raw file(s) and their SHA-256 hashes;
- the processing script/notebook and repository commit;
- scorer and normalization versions;
- filters, exclusions, and aggregation rules;
- generation timestamp; and
- the output artifact path.

Processed data may add derived columns or summaries, but it must preserve the
link back to `trial_id` and must not silently discard non-success statuses.

### Required run manifest

Each batch needs a small manifest containing at least:

- `run_id`, experiment ID, configuration ID, and config hash;
- model ID, revision/checksum, tokenizer revision, quantization artifact, and
  KV/cache settings;
- runtime/backend names and versions;
- prompt, task, corpus, and generator revisions;
- fixture, generation, and orchestration seeds;
- target and actual token lengths/positions;
- hardware/OS/environment identifiers;
- repository Git SHA and command invocation;
- UTC start/end timestamps and timezone;
- raw record path and processed-output paths; and
- operator notes for deviations, missing cells, OOMs, or timeouts.

Do not put API keys, personal machine identifiers, or other secrets in a
manifest. A redacted descriptive hardware label is sufficient.

## 16. Hardware/environment record

Every run batch should record:

- machine identifier (non-secret descriptive label);
- chip/CPU/GPU;
- total memory;
- OS version;
- Python version;
- relevant runtime package versions;
- model artifact identifier/hash/revision;
- git commit of this repository;
- date/time;
- optional thermal/power mode notes.

Avoid committing personal machine identifiers or secrets.

## 17. Experiment freeze rule

Once a main experiment starts, changes to:

- prompts;
- task generation;
- scoring;
- runtime settings;
- context construction;

must trigger either:

1. a new experiment revision documented in the README, or
2. a rerun of affected cells.

Never merge incompatible runs into one plot without labeling them.

## 18. Interpretation rules

- “Supports 262K” means accepted by the declared model/runtime configuration, not necessarily reliable reasoning at 262K.
- A synthetic retrieval success does not imply repository/agent success.
- A quantization difference is only causal if other major variables are controlled.
- One impressive or ridiculous agent trajectory is a case study, not a rate.
- Null results should be preserved.
