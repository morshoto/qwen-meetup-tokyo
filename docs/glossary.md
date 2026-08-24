# Glossary

Project terminology should be used consistently across docs, code, notebooks, and slides.

## Accepted / maximum context

The maximum input length accepted by the model/runtime configuration without exceeding a declared or practical limit.

This is a **capacity property**, not proof that every token is used reliably.

## Effective context window

The largest tested context length at which a task family retains a specified fraction of its short-context baseline performance.

Project definition is in `methodology.md`.

Effective context is task- and configuration-dependent.

## Evidence position

The normalized token location of relevant evidence inside controlled context.

Typical tested values:

```text
0.05, 0.25, 0.50, 0.75, 0.95
```

## Lost in the Middle

Position-dependent long-context behavior in which relevant information around the middle of context is used less effectively than information nearer boundaries.

In this project it is a hypothesis to test, not an assumed property of Qwen3.8-27B.

## Lost in the Agent

Project term for a related agent-history phenomenon: a critical observation discovered earlier in a trajectory becomes buried in accumulated conversation/tool context and is not reliably reused later.

## Literal retrieval

Task where the answer can be obtained from directly matching/high-overlap evidence.

## Semantic retrieval

Task requiring recognition/paraphrase/concept mapping rather than simple string/key lookup.

## Multi-hop reasoning

Task requiring multiple separate evidence spans to produce the answer.

## Quantization

Reduction of numerical precision/storage used by model tensors (and potentially runtime state). A bit label alone is not a full specification; method/format and runtime details matter.

## Q4 / Q6 / Q8

Informal precision labels used in planning. The final experiment must replace/augment these with exact artifact/method metadata.

## KV cache

Inference-time cache of attention key/value state used to avoid recomputing previous tokens in autoregressive transformers/attention layers. Memory cost generally grows with context and implementation/model architecture.

## Prefill

Processing the input prompt/context before output-token generation. For long contexts this can be a major latency cost.

## Decode

Autoregressive output-token generation after prefill.

## TTFT

Time to first token: wall-clock latency between issuing a generation request and receiving the first generated token, under the measurement convention documented in methodology.

## Prefill throughput

Input tokens processed per second during prompt evaluation, when separable timing is available.

## Decode throughput

Generated output tokens per second during decode.

## Agent trajectory

Ordered sequence of model invocations, tool calls, tool results, observations, and state changes used to solve one agent task.

## Tool-call validity

Whether a tool call conforms to the declared function/schema contract. It does not imply the tool choice was useful.

## Recovery rate

How often an agent that encounters a defined failure/error returns to a productive path and ultimately reaches the success condition.

## Context strategy

Method for choosing what information to place in prompt context. In exp_005, examples include curated, neighborhood, and broad repository context.

## Trial

One execution of a specific model/runtime/configuration on one task/context instance.

## Batch

A related set of planned trials run under one resolved experiment configuration/environment snapshot.

## Raw result

Append-only trial-level generation, telemetry, execution status, and score record.

## Processed result

Derived aggregate/table computed from raw result records.

## Pareto frontier

Set of configurations for which no other measured configuration is simultaneously better on all considered dimensions (for example, higher accuracy and lower memory).

## Finding

A validated measured observation from this repository. Hypotheses and expected behavior are not findings.
