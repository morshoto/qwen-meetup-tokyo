# Research plan

**Issue:** #1 — Define research questions, hypotheses, and project scope

**Working title:** When Does a Local LLM Start to Break?

**Target model:** `Qwen/Qwen3.8-27B`

**Plan date:** 2026-08-25

## 1. Motivation and central question

Quantized open-weight models can now fit on consumer systems and participate
in retrieval, coding, tool use, and multi-turn agent workflows. The practical
question is no longer only whether a model can load. It is how much useful
capability survives under the constraints that make local deployment practical:

- reduced weight precision;
- very large prompts and growing histories;
- limited memory bandwidth and KV-cache capacity;
- prompt/prefill cost;
- runtime/backend behavior; and
- tool use and state tracking over many turns.

The primary research question is:

> **When does a local 27B-class model stop behaving reliably as we increase
> context, reduce precision, and turn static prompts into long-running agent
> trajectories?**

The project uses Qwen3.8-27B as an experimental subject, not as evidence that
one model represents all local systems. We will measure three kinds of
breakpoint:

1. **Capability breakpoint:** task correctness falls below the short-context,
   reference-precision baseline.
2. **Systems breakpoint:** the request no longer fits available memory, or
   latency/throughput becomes impractical for the declared setup.
3. **Trajectory breakpoint:** an agent loses state, violates a tool contract,
   or fails to reach the required end state as history grows.

The useful model is a high-dimensional function:

```text
reliability = f(
    context_length,
    information_position,
    task_difficulty,
    quantization,
    trajectory_length,
    runtime_and_system_constraints
)
```

The core study isolates selected interactions rather than trying to identify
the whole function.

## 2. Research questions

### RQ1 — Effective context

Does useful context depend on the task? We distinguish the maximum context a
runtime accepts from the length at which literal retrieval, semantic
retrieval, multi-hop reasoning, repository reasoning, and agent history still
work reliably.

### RQ2 — Position bias / Lost in the Middle

When equivalent evidence appears at different relative positions inside a
controlled context, does task success change as total context grows?

### RQ3 — Quantization × context

Is the capability gap between precision variants approximately constant, or
does it grow with context length and semantic difficulty?

### RQ4 — Agent-history reliability

As an agent accumulates observations, code snippets, failures, test output, and
earlier conclusions, does it continue to reuse relevant discoveries reliably?
This is the **Lost in the Agent** extension of the static context question.

### RQ5 — More context versus better context

For repository-level tasks, does a broader repository view help, plateau, or
hurt compared with a curated relevant subset?

### RQ6 — Local systems trade-off

What is the Pareto frontier among task success, memory, time to first token,
prefill throughput, decode throughput, total task time, and optional energy?

## 3. Core hypotheses

These are hypotheses, not findings. Each has a falsifiable prediction and a
measurable dependent variable.

| ID | Falsifiable prediction | Primary dependent variable | Minimum falsification test |
| --- | --- | --- | --- |
| **H1** | At a fixed long context, evidence near the middle has lower correctness than equivalent evidence near the beginning or end, and the gap grows with length. | The position gap `Δ_pos(C) = A_edge(C) - A_middle(C)` at each length and its context-length × position interaction. | Bootstrap the paired position gap at each length and a direct difference-in-differences or fitted interaction across lengths. The fixed-length contrast tests position bias; the across-length contrast tests whether it grows. Do not use overlapping marginal confidence intervals as the decision rule. |
| **H2** | Useful context differs by task: literal retrieval retains performance longer than semantic retrieval, which retains performance longer than multi-hop or repository reasoning. | Task-specific effective context length and absolute accuracy-vs-length curve. | Apply the baseline-validity gate and the same breakpoint rule to all task families; report baseline-limited families without a relative breakpoint and reject the ordering if differences are not directionally stable across repeated instances. |
| **H3** | Lower-bit weights have a larger capability/reliability penalty at long context than at short context while improving memory or throughput. | Accuracy loss, context × precision interaction, peak memory, TTFT, prefill tok/s, and decode tok/s. | Compare identical prompts and lengths across precision variants; test whether the long-minus-short loss is larger for lower precision. |
| **H4** | A model can pass literal retrieval while failing semantically distributed or multi-hop evidence at the same nominal context length. | Literal, semantic, and multi-hop accuracy plus per-hop failure attribution. | Hold length and distractor budget fixed and compare matched task families. |
| **H5** | End-state success and valid tool-call rate decline as retained agent history grows, even when individual turns remain locally plausible. | End-state success, valid tool calls, first unrecoverable failure step, retry count, prompt tokens, and `pass^k`. | Replay the same tasks at 1, 4, 8, and 16 tool turns with a fixed sandbox. |
| **H6** | For repository tasks, a curated relevant subset can outperform a broad repository context on success and systems cost. | Test-passing success, diagnosis accuracy, input tokens, memory, latency, and cost per successful task. | Compare curated files, a local neighborhood, and broad context on the same pinned task and repository revision. |
| **H7** | On long local agent workloads, prompt processing and repeated context ingestion can dominate decode speed. | TTFT, prefill tok/s, decode tok/s, total task time, and fraction of time spent in prefill. | Measure the same task across context lengths and report prefill and decode separately. |

### Operational definitions

- **Short-context reference:** the 8,192-token, reference-precision condition in
  `exp_001`, with the same task prompt and generation settings.
- **90% breakpoint:** the first sustained tested length whose point estimate is
  below 90% of the corresponding short-context reference, using the crossing
  rule in [`methodology.md`](methodology.md). Also report an 80% sensitivity
  threshold; no threshold is universal.
- **Position interaction:** for each tested length `C`, define `A_edge(C)` as
  the mean of the matched beginning and end conditions and calculate
  `Δ_pos(C) = A_edge(C) - A_middle(C)`. Resample matched task instances to
  bootstrap both each `Δ_pos(C)` and the predeclared contrast
  `Δ_pos(C_long) - Δ_pos(C_short)`, or fit an equivalent context-length ×
  position interaction. A positive long-minus-short contrast is evidence for
  the “grows with length” component of H1; a fixed-length positive gap is the
  separate evidence for positional bias.
- **Baseline-validity gate:** report a relative effective-context breakpoint for
  a task family only when its 8,192-token reference-precision baseline reaches
  at least 80% end-to-end success over the predeclared attempted instances.
  Runtime and invalid-output failures remain in this denominator. If it misses
  that gate, classify the family as **baseline-limited** and do not rank its
  relative breakpoint against other families. Always show the absolute
  accuracy curve, end-to-end successes/attempted trials, and runtime-failure
  status for baseline-limited families.
- **Task correctness:** exact match or a deterministic structured grader for
  retrieval; task-specific answer keys for semantic and multi-hop tasks.
  Free-form answers are not judged by an uncalibrated LLM alone.
- **Agent success:** the environment reaches the required end state and all
  mandatory policy/tool constraints pass. A plausible final message without a
  correct state change is a failure.
- **Uncertainty:** use repeated task instances and paired bootstrap intervals
  for accuracy and success rates, reporting the number of instances.

## 4. Core experiment sequence

The sequence isolates one variable before introducing the next. Experiment
numbering follows the repository backlog and existing `experiments/README.md`.

| Experiment | Status | Main question | Conditions | Hypotheses |
| --- | --- | --- | --- | --- |
| `exp_001-context_measurement` | Scaffolded | How does useful context change with length, evidence position, and task type? | 8k, 32k, 64k, 128k, 256k; positions 0.05, 0.25, 0.50, 0.75, 0.95; literal, semantic, multi-hop; reference precision. | H1, H2, H4 |
| `exp_002-quantization_llama_cpp_gguf` | Scaffolded | What capability and systems trade-off comes from one reproducible quantization family? | GGUF F16 reference plus Q8_0, Q6_K, Q5_K_M, and Q4_K_M under llama.cpp; short and medium contexts; same task suite. | H3, H7 |
| `exp_003-context_x_quantization` | Planned core | Does the context breakpoint move as precision is reduced? | A smaller factorial subset of `exp_001` × `exp_002`, expanded only after the interaction is observable. | H3; cross-check H1/H2 |
| `exp_004-agent_context_growth` | Planned core | How does retained trajectory history affect reliable tool use? | Fixed local tool sandbox; 1, 4, 8, and 16 tool turns; repeated tasks; reference and selected quantized configurations. | H5, H7 |
| `exp_005-repository_reasoning` | Planned core | Do synthetic context findings transfer to repository-level coding tasks? | Pinned repositories/tasks; curated files, local neighborhood, and broad context; machine-checkable tests; selected precision variants. | H6; cross-check H2/H3/H5 |

`exp_005` is deliberately the repository-level validation experiment from
Issue #11. The curated-versus-broad comparison is a factor inside that
experiment, not a new `exp_005` name.

The first three experiments are the minimum research core. `exp_004` is the
strongest agent novelty angle and `exp_005` is the realism check.

## 5. Measurement design

### Fixed reference setup

Before collecting comparisons, record the exact model revision, tokenizer,
inference runtime and version, quantizer, hardware, operating-system power
mode, maximum output tokens, prompt template, and seed. The floating model
name is not enough: every result must include a resolved revision or artifact
checksum.

The core matrix is text-only and uses explicit non-thinking/instruct mode so
hidden reasoning-token growth is not an uncontrolled second context variable.
Thinking mode can be added as a labeled extension. Primary comparisons use
greedy decoding (`temperature: 0.0`); repeated-sampling analyses get their own
configuration.

### Context matrix

`exp_001` starts with the existing configuration:

- context lengths: 8,192; 32,768; 65,536; 131,072; 262,144 tokens;
- evidence positions: 5%, 25%, 50%, 75%, and 95% of the input;
- task families: literal retrieval, semantic retrieval, and multi-hop
  reasoning;
- same evidence, distractor budget, prompt template, and answer key across
  positions;
- at least five independent task instances per pilot cell, increasing to ten
  for any condition used in the presentation.

For multi-hop instances, store the token position of every required evidence
span rather than reducing the task to one average position. A failed 256k
allocation is a systems outcome, not a missing data point.

### Quantization matrix

`exp_002` must name the actual quantizer and storage/runtime format before a
result is treated as a comparison. “q4” alone is not a method. Record weight
precision, activation precision, KV-cache precision, group size, calibration
corpus, outlier policy, packing format, and runtime kernel path.

Report:

- task accuracy and per-task failure cases;
- peak host/device memory and model load time;
- time to first token and total latency;
- prefill tokens/second and decode tokens/second;
- completion, OOM, timeout, or invalid-response status.

### Agent trajectory matrix

Use a small deterministic tool sandbox. Each task has a machine-checkable end
state, explicit schemas, a bounded set of valid action paths, and controlled
tool responses. Vary retained history while holding task, tools, and initial
state fixed. Preserve model input, tool call, tool result, validation result,
and timestamp/latency in every trajectory log.

Classify failures as:

1. a bad decision on the current turn;
2. a stale or lost fact from earlier history;
3. a tool/schema violation;
4. a recoverable error followed by success; or
5. the first unrecoverable failure.

### Repository reasoning and context controls

`exp_005-repository_reasoning` uses pinned repositories and revisions. Initial
tasks should include locating an implementation, identifying a seeded
regression, tracing state mutation, diagnosing a failing test, and making a
small test-verified fix.

The context comparison uses the same underlying repository, task, and
answer-bearing evidence:

- **curated:** tightly selected relevant files;
- **neighborhood:** relevant files plus their local imports/callers;
- **broad:** a larger repository view where practical.

This is intentionally not described as the same information or token budget:
the input length is one systems variable of interest. If a capability-only
comparison is needed, add a matched-token distractor/random-selection control
with the same input length. An oracle-curated condition may be used as an
upper-bound diagnostic, but it is not a deployable baseline.

## 6. Stretch scope

Only add these after the core matrix is working:

- MTP/speculative decoding on/off;
- alternate local runtimes;
- KV-cache precision/compression;
- power/energy per completed task;
- other 20–30B local models;
- cloud frontier comparison;
- image/video agent tasks; and
- million-token extrapolation or YaRN experiments.

Stretch work must not delay the core answer.

## 7. Explicit non-goals for v1

The first talk is not intended to:

- prove a universal ranking of Qwen against all frontier models;
- benchmark every Qwen variant or quantization algorithm;
- claim that one machine represents all local hardware;
- measure training-time properties;
- reproduce all LongBench, RULER, or HELMET tasks;
- optimize the inference engine itself; or
- claim architectural causality without direct evidence.

## 8. Issue #1 planning completion criteria

Issue #1 is complete when:

1. `docs/research-plan.md` exists and is reviewable on its own.
2. Every planned core experiment maps to at least one hypothesis.
3. Each hypothesis is falsifiable and names a measurable dependent variable.
4. Core scope is small enough to execute before presentation synthesis.

## 9. Project / presentation success criteria

The full project is ready for presentation synthesis when:

1. RQ1–RQ6 and the hypotheses map to measured results.
2. `exp_001` reports paired position effects, the H1 interaction test, and
   task-specific breakpoints with baseline-validity status.
3. `exp_002` reports an accuracy/performance Pareto comparison and separates
   model failures from runtime failures.
4. `exp_003` tests the context × precision interaction.
5. `exp_004` reports end-state success, tool validity, and first unrecoverable
   failure as history grows.
6. `exp_005` connects synthetic findings to pinned repository tasks and
   machine-checkable outcomes.
7. Every figure can be regenerated from committed results, and every headline
   claim names its experiment, task set, and uncertainty.

The talk should answer “where does it break?” with a small table of measured
breakpoints and representative failure traces, rather than one unqualified
maximum-context number. A null result is useful evidence, not a failed study.
