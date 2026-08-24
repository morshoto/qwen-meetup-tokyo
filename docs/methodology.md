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

Then define:

```text
C_effective(T, q) = largest tested context C such that
                    A(T, q, C) >= alpha * A_baseline(T, q)
```

where `q` is quantization/configuration.

### Proposed default

`alpha = 0.90`.

This threshold must be sensitivity-checked. Report whether conclusions change for reasonable alternatives such as `0.85` or `0.95`.

Because positional bias exists, compute at least two versions:

1. **mean-position effective context** — aggregate across tested positions;
2. **worst-position effective context** — uses the lowest position-conditioned performance.

The latter is more conservative and may be more relevant to uncontrolled real prompts.

## 9. Accuracy / task metrics

### Exact/task accuracy

Binary correctness under a deterministic scorer where possible.

### Semantic score

Use only when exact matching is inappropriate. Prefer explicit answer normalization or structured expected concepts before using an LLM judge.

### Completion rate

For agent/coding tasks:

```text
successful final state / attempted tasks
```

A “successful” implementation task should preferably be machine-checkable by tests.

### Tool-call validity

Fraction of tool calls that are syntactically/schema-valid.

### Useful tool-call rate

Optional/manual metric: whether a valid call plausibly advances the task. Keep separate from syntactic validity.

### Recovery rate

Among trajectories containing a defined error or failed action, fraction that return to a productive path and ultimately satisfy the task success condition.

### Steps to solution

Count model turns and tool calls separately.

## 10. Systems metrics

Every runtime should expose as many of these as reliably possible.

### Time to first token (TTFT)

Wall-clock time from generation request start until first generated token becomes available.

### Prefill throughput

```text
input tokens processed / prefill seconds
```

If the runtime cannot expose prefill independently, report prompt-evaluation timing or clearly mark unavailable.

### Decode throughput

```text
generated tokens / decode seconds
```

Do not report “tok/s” without labeling input/prefill vs output/decode.

### Total model-call time

End-to-end generation call duration.

### Total task time

For agents:

```text
sum(model inference + tools + orchestration overhead)
```

### Peak memory

Record the best available process/system metric and state exactly how it is measured. On unified-memory systems, “VRAM” may be misleading; use peak resident/unified memory terminology appropriate to the measurement source.

### Model artifact size

Size on disk is not the same as peak memory. Record separately.

### Energy / power

Stretch metric. Only report if the measurement method is stable and documented. Prefer task-level energy (Wh or joules) over a single instantaneous watt reading.

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

## 15. Hardware/environment record

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

## 16. Experiment freeze rule

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

## 17. Interpretation rules

- “Supports 262K” means accepted by the declared model/runtime configuration, not necessarily reliable reasoning at 262K.
- A synthetic retrieval success does not imply repository/agent success.
- A quantization difference is only causal if other major variables are controlled.
- One impressive or ridiculous agent trajectory is a case study, not a rate.
- Null results should be preserved.
