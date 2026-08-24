# Experiment program

This document translates the project research plan into the numbered experiment sequence under `experiments/`.

## 1. Dependency overview

```text
Research + methodology + data + runner
              |
              v
   exp_001 Context baseline
              |
              +------------------+
              |                  |
              v                  v
   exp_002 Quantization     context findings
              |                  |
              +--------+---------+
                       v
            exp_003 Context × quantization
                       |
              +--------+---------+
              |                  |
              v                  v
   exp_004 Agent growth    exp_005 Repo validation
              |                  |
              +--------+---------+
                       v
             Cross-experiment analysis
                       |
                       v
                  Presentation
```

Foundation issues #1–#6 should be sufficiently complete before expensive main sweeps.

---

# exp_001 — Context measurement

**Repository path:** `experiments/exp_001-context_measurement/`  
**Issue:** #7

## Research questions

- RQ1 — effective context by task type;
- RQ2 — position bias / Lost in the Middle.

## Hypotheses

- H1 — middle-position performance may degrade with context length;
- H2 — effective context differs by task type;
- H4 — literal retrieval remains robust longer than semantic/multi-hop tasks.

## Independent variables

1. context length;
2. evidence position;
3. task type.

## Proposed grid

Context:

```text
8K, 32K, 64K, 128K, 262K/highest practical
```

Position:

```text
5%, 25%, 50%, 75%, 95%
```

Task:

```text
literal retrieval
semantic retrieval
multi-hop reasoning
```

## Primary outputs

- per-cell accuracy;
- position curves by context length;
- context degradation curves by task type;
- effective context values;
- TTFT/prefill/decode and memory observations;
- representative failures.

## Implementation notes

- Generate contexts by token count using the same tokenizer as inference.
- Store task instance IDs and exact evidence offsets.
- For multi-hop, store every evidence span and required reasoning chain metadata.
- First run one precision/model configuration only. Do not mix quantization into this baseline before the test itself is stable.

## Exit criteria

- main matrix is complete enough to characterize position/context effects;
- notebook reruns from recorded results;
- effective-context metric can be calculated;
- benchmark is not saturated at every length/task.

---

# exp_002 — Quantization baseline

**Current repository path:** `experiments/exp_002-quantization_using_approach_a/`  
**Issue:** #8

The `approach_a` name is temporary. Rename the experiment once the implementation is selected.

## Research questions

- RQ3 — quantization capability trade-off;
- RQ6 — local systems Pareto frontier.

## Decision required before main run

Select **one quantization family/runtime approach** for the controlled first study.

Candidates may include runtime-native or published artifacts appropriate to the selected local inference backend. The goal is not to compare quantizers yet; it is to compare precision levels while minimizing other changes.

## Proposed variants

Where supported:

```text
Q8
Q6
Q5
Q4
```

A higher-precision reference may be added if practical, but do not make full-precision execution a hard dependency if it prevents the study from running locally.

## Tasks

Use short/medium-context versions of the shared task ladder. Avoid spending the exp_002 budget on huge context; exp_003 owns that interaction.

## Primary outputs

- artifact size;
- peak memory;
- TTFT;
- prefill throughput;
- decode throughput;
- task accuracy;
- accuracy-vs-memory Pareto plot;
- recommendation of two or three variants to carry into exp_003.

## Exit criteria

- quantization provenance is known;
- matched prompts/tasks are run across variants;
- the chosen baseline variants are practical and stable;
- exp_003 can reuse them without changing quantization method.

---

# exp_003 — Quantization × context interaction

**Planned path:** `experiments/exp_003-context_x_quantization/`  
**Issue:** #9

## Research question

Does lower precision disproportionately damage long-context behavior or position robustness?

## Hypothesis

H3 — quantization degradation may grow with context length.

## Design

Reuse exp_001 task instances, context generation, position definitions, and scoring. Reuse exp_002 model artifacts/runtime.

This is important: exp_003 should be an **interaction study**, not a fresh benchmark with new prompts.

## Practical grid strategy

A full grid can be expensive. Use a staged design:

1. run all selected quantizations at 8K, 64K, and 128K with all positions;
2. inspect whether an interaction appears;
3. add 32K and highest-practical context to locate thresholds;
4. increase sample count around transition regions rather than blindly expanding every cell.

Any adaptive expansion must be documented before final statistical interpretation.

## Primary outputs

- quantization × context heatmap;
- position × quantization plots;
- effective context per quantization/task;
- degradation relative to each quantization's short-context baseline;
- memory/performance/capability trade-off.

## Key possible conclusions

- quantization loss is roughly constant;
- quantization loss accelerates with context;
- only semantic/multi-hop tasks show interaction;
- positional bias, not precision, dominates;
- one low-bit variant lies on a strong Pareto frontier.

All are acceptable outcomes.

---

# exp_004 — Agent context growth / Lost in the Agent

**Planned path:** `experiments/exp_004-agent_context_growth/`  
**Issue:** #10

## Motivation

Static documents are not the only source of long context. Real agents construct their own context from:

- previous reasoning/output;
- tool calls;
- tool results;
- errors;
- repository snippets;
- test logs;
- interim decisions.

The experiment asks whether the model can still use something important it discovered earlier.

## Core controlled task

Build a deterministic tool environment in which the agent discovers a critical fact, then performs additional interactions before that fact becomes necessary.

Example pattern:

```text
turn 3: discover target module = middleware/auth.ts
turn 4..N: inspect unrelated but plausible files/tool outputs
turn N+1: task requires using the earlier discovery
```

Vary the relative location of the critical observation in the final accumulated history.

## Independent variables

- accumulated context length / number of turns;
- relative position/age of critical discovery;
- selected quantization;
- optionally history-management strategy in a later extension.

## Metrics

- final task success;
- correct reuse of prior critical fact;
- tool-call validity;
- repeated/redundant actions;
- state-tracking contradiction rate;
- recovery after errors;
- total input tokens across trajectory;
- total wall-clock time.

## Important control

Avoid letting the tool environment itself become nondeterministic. The first version should use fixed outputs/fixtures so failures can be attributed more cleanly to model behavior.

## Primary figure

Reliability/task success vs accumulated trajectory length, with separate lines/conditions for precision or evidence age.

---

# exp_005 — Repository-level validation

**Planned path:** `experiments/exp_005-repository_reasoning/`  
**Issue:** #11

## Motivation

Synthetic tasks give control but may not predict useful software-engineering behavior.

## Candidate task families

- identify where a feature is implemented;
- trace state mutation across modules;
- identify a seeded regression;
- diagnose a failing test;
- make a small patch and satisfy machine-checkable tests.

## Context strategy comparison

At minimum compare:

1. **curated** — only known relevant files;
2. **neighborhood** — relevant files plus nearby modules/tests;
3. **broad** — large repository slice/full repo where practical.

The same task and expected success condition must be used across strategies.

## Metrics

- correct file identification;
- diagnosis correctness;
- final test success;
- number of files/tools inspected;
- total input tokens;
- elapsed time;
- repeated/unproductive actions.

## Repository selection rules

Prefer tasks that are:

- legally redistributable or reproducible from a pinned public revision;
- small enough to run repeatedly;
- machine-checkable;
- not likely memorized verbatim from a famous benchmark issue;
- understandable enough for presentation failure examples.

---

# Cross-experiment synthesis

## Required comparisons

1. Is the effective context ordering consistent across task types?
2. Does quantization move the context-failure threshold?
3. Are agent failures predicted by synthetic position/context failures?
4. Does repository context breadth show the same “more context can hurt” pattern?
5. Which local configuration lies on the best capability/memory/latency frontier?

## Stop conditions

Do not keep adding experiments merely because an additional benchmark exists.

Stop expanding when the project can answer:

- where context begins to degrade;
- whether position matters;
- whether quantization changes that degradation;
- whether the effect transfers to an agent or repository task;
- what practical configuration we recommend.
