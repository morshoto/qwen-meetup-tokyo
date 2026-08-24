# Research plan

**Issue:** #1 — Define research questions, hypotheses, and project scope

## 1. Working title

**When Does a Local LLM Start to Break?**  
*Quantization, Long Context, and Agent Reliability in Qwen3.8-27B*

Alternative framing for the final talk:

- **The Local Intelligence Frontier** — capability, memory, and context on one machine.
- **Lost in the Agent** — what happens when a local AI forgets its own work?
- **How Much Context Does a Local LLM Really Have?**

The working title should not determine the result. Final naming happens after the strongest measured finding is known.

## 2. Motivation

The interesting local-LLM question is no longer simply whether a capable model can load on one computer. Quantized open-weight models can already fit on consumer systems and can participate in coding, retrieval, tool use, and multi-turn agent workflows.

The unresolved practical question is:

> **How much of the model's useful capability survives under the constraints that make local deployment practical?**

Those constraints include:

- reduced weight precision;
- very large prompts and growing histories;
- limited memory bandwidth;
- KV/cache growth;
- prompt/prefill cost;
- runtime/backend implementation;
- tool-use and state-tracking over many turns.

The project uses Qwen3.8-27B as an experimental subject for that broader question.

## 3. Central research question

> **When does a local 27B-class model stop behaving reliably as we increase context, reduce precision, and turn static prompts into long-running agent trajectories?**

A useful conceptual model is:

```text
Reliability = f(
    context_length,
    information_position,
    task_semantic_difficulty,
    quantization,
    trajectory_length,
    runtime/system_constraints
)
```

The core study will not attempt to fully identify this high-dimensional function. Instead, the numbered experiments isolate important interactions.

## 4. Research questions

### RQ1 — Effective context

Does Qwen3.8-27B's useful context window depend on what the model is being asked to do?

We distinguish:

- accepted / maximum context;
- literal-retrieval context;
- semantic-retrieval context;
- multi-hop reasoning context;
- repository reasoning context;
- agent-history context.

The project should avoid saying a model “has N tokens of context” when the underlying claim is only that the runtime accepts N tokens.

### RQ2 — Position bias / Lost in the Middle

When the same evidence appears at different relative positions inside an otherwise controlled context, does task success change?

We care about whether performance at approximately the middle of context is lower than near the beginning or end, especially as total context grows.

### RQ3 — Quantization × context

Is the capability gap between higher- and lower-precision variants approximately constant, or does it grow with context length and semantic difficulty?

This is more important than a standalone “Q4 vs Q8” comparison.

### RQ4 — Agent-history reliability

As a tool-using agent accumulates its own observations, code snippets, failures, test output, and earlier conclusions, does it continue to reuse relevant discoveries reliably?

This motivates the **Lost in the Agent** experiment: bury a critical fact inside an accumulated trajectory rather than a static synthetic document.

### RQ5 — More context vs better context

For repository-level tasks, does supplying a broader portion of the repository help, plateau, or hurt compared with a curated relevant subset?

This tests whether context selection is more valuable than raw context capacity.

### RQ6 — Local systems trade-off

What is the Pareto frontier among:

- task success;
- memory;
- time to first token;
- prefill throughput;
- decode throughput;
- total task time;
- optional energy usage?

The final recommendation should be a configuration trade-off, not simply “use the highest precision.”

## 5. Hypotheses

These are **hypotheses, not findings**.

### H1 — Position bias increases with context length

At sufficiently long contexts, accuracy at mid-context evidence positions will be lower than accuracy near prompt boundaries.

Conceptually:

```text
P(correct | evidence_position ≈ 0.5)
<
P(correct | evidence_position ≈ 0.05 or 0.95)
```

We do not assume the curve must be perfectly U-shaped.

### H2 — Effective context depends on task type

Literal key/value retrieval will retain accuracy at longer contexts than semantic retrieval, multi-hop reasoning, repository reasoning, or long agent trajectories.

Possible ordering to test, not assume:

```text
literal retrieval
>= semantic retrieval
>= multi-hop reasoning
>= repository / agent tasks
```

### H3 — Quantization damage is context-dependent

Lower precision may be nearly indistinguishable from a higher-precision baseline at short context but diverge more strongly at long context.

The interaction term matters:

```text
loss(Q4 vs Q8 at long context)
>
loss(Q4 vs Q8 at short context)
```

A null result is equally valuable: if the gap is stable or negligible, that supports more aggressive local compression.

### H4 — Semantic difficulty exposes failures before literal retrieval does

A model may pass classic needle/key retrieval while failing semantic and multi-hop tasks at the same context length.

### H5 — Agent reliability degrades with accumulated history

A critical fact discovered early in an agent trajectory may become less likely to influence later actions after many tool calls and context growth.

### H6 — Curated repository context can outperform maximal context

A smaller, high-signal file set may outperform a much larger repository dump on diagnosis or implementation tasks.

### H7 — Prefill becomes a first-class bottleneck

For long-context local agent workloads, total latency may be dominated by prompt processing and repeated context ingestion rather than decode speed alone.

## 6. Core contribution we want

The strongest version of the work would contribute:

1. a reproducible definition and measurement of **effective context** for a local model;
2. evidence about whether quantization interacts with long-context capability;
3. an agent-oriented extension of positional-bias testing (**Lost in the Agent**);
4. local-system measurements linking capability to memory and latency;
5. a failure taxonomy that explains *how* long-running local behavior degrades;
6. a practical configuration recommendation for local users.

## 7. Core scope

The first complete version of the project should include:

- `exp_001` — context length × evidence position × task type;
- `exp_002` — one concrete quantization approach and precision sweep;
- `exp_003` — quantization × context interaction;
- `exp_004` — agent history growth / Lost in the Agent;
- `exp_005` — small repository-level validation.

The first three are the minimum research core. `exp_004` is the strongest novelty/agent angle. `exp_005` is the realism check.

## 8. Stretch scope

Only add these after the core matrix is working:

- MTP/speculative decoding on/off;
- alternate local runtimes;
- KV-cache precision/compression;
- power/energy per completed task;
- other ~20–30B local models;
- cloud frontier comparison;
- image/vision agent tasks;
- million-token extrapolation or YaRN experiments.

These are interesting, but each adds a new confounder.

## 9. Explicit non-goals for v1

The v1 talk is **not** intended to:

- prove a universal ranking of Qwen against all frontier models;
- benchmark every Qwen variant;
- benchmark every quantization algorithm;
- claim that one machine represents all local hardware;
- measure training-time properties;
- reproduce all LongBench/RULER/HELMET tasks;
- optimize the inference engine itself;
- claim causality for architectural mechanisms without direct evidence.

## 10. What would make the talk interesting?

Any of the following would be a strong result:

- low-bit variants match high precision at short context but diverge sharply at long context;
- quantization barely matters and context management dominates;
- classic Lost in the Middle is weak or absent in Qwen3.8-27B;
- literal retrieval remains strong while semantic/multi-hop performance collapses much earlier;
- broad repository context hurts compared with curated files;
- agent state tracking fails long before the formal context limit;
- prefill cost, not generation speed, becomes the practical local bottleneck.

A null result is not a failed experiment. If Q4 is robust, or position bias is minimal, that is useful evidence.

## 11. Decision rules

When trade-offs arise, prioritize in this order:

1. reproducibility;
2. interpretability / control of variables;
3. real-world relevance;
4. breadth of benchmark coverage;
5. spectacle/demo value.

## 12. Completion criteria for research planning

Issue #1 can be considered complete when:

- RQ1–RQ6 are accepted or revised explicitly;
- hypotheses are falsifiable and map to measurable metrics;
- core vs stretch scope is agreed;
- every experiment maps to at least one research question;
- no final result is implied in the wording of the experiment.
