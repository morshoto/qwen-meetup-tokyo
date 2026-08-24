# Project documentation

This directory is the engineering and research handoff for the Qwen Meetup Tokyo project.

The project asks a practical research question:

> **When does a local LLM start to break?**

The case study is `Qwen/Qwen3.8-27B`, with emphasis on long context, quantization, and agentic workloads on local hardware.

## Read this first

If you are joining the project as an engineer or researcher, read these files in order:

1. [`research-plan.md`](research-plan.md) — why the project exists, research questions, hypotheses, scope.
2. [`experiment-program.md`](experiment-program.md) — the numbered experiment sequence and dependencies.
3. [`methodology.md`](methodology.md) — common controls, metrics, evaluation rules, and effective-context definition.
4. [`engineering-architecture.md`](engineering-architecture.md) — reusable `src/llm_lab` design and runtime boundaries.
5. [`data-and-result-contracts.md`](data-and-result-contracts.md) — shared prompt/task schemas and result-record requirements.
6. [`analysis-plan.md`](analysis-plan.md) — required figures, statistical treatment, and error taxonomy.
7. [`related-work.md`](related-work.md) — papers, benchmarks, talks, and why they matter to our design.
8. [`presentation-plan.md`](presentation-plan.md) — how measured results should become the meetup talk.
9. [`findings.md`](findings.md) — dated measured observations. Do **not** put hypotheses here as findings.
10. [`reproducibility-checklist.md`](reproducibility-checklist.md) — pre-run and pre-merge checklist.
11. [`glossary.md`](glossary.md) — project terminology.

## Source of truth by topic

| Topic | Source of truth |
|---|---|
| Research questions / hypotheses | `research-plan.md` |
| Experiment IDs and dependencies | `experiment-program.md` |
| Shared controls and metrics | `methodology.md` |
| Reusable implementation | `engineering-architecture.md` |
| Prompt/task/result schemas | `data-and-result-contracts.md` |
| Plots and cross-experiment analysis | `analysis-plan.md` |
| Literature / prior work | `related-work.md` |
| Measured conclusions | `findings.md` |
| Talk narrative | `presentation-plan.md` |

## GitHub issue mapping

The initial execution backlog is tracked in issues:

- [#1](../issues/1) — research questions, hypotheses, scope
- [#2](../issues/2) — related work
- [#3](../issues/3) — methodology and reproducibility
- [#4](../issues/4) — shared prompt/task/context data
- [#5](../issues/5) — model/runtime interface
- [#6](../issues/6) — runner, scoring, telemetry, result schema
- [#7](../issues/7) — `exp_001` context measurement
- [#8](../issues/8) — `exp_002` quantization baseline
- [#9](../issues/9) — `exp_003` quantization × context
- [#10](../issues/10) — `exp_004` agent context growth / Lost in the Agent
- [#11](../issues/11) — `exp_005` repository-level validation
- [#12](../issues/12) — cross-experiment error taxonomy
- [#13](../issues/13) — presentation synthesis

GitHub does not resolve `../issues/N` as a repository-relative file link in every renderer. If needed, use `https://github.com/morshoto/qwen-meetup-tokyo/issues/N`.

## Important status convention

Every statement in this documentation should fall into one of four categories:

- **Established background** — supported by cited prior work or official model/runtime documentation.
- **Project decision** — a methodology or engineering choice we have adopted.
- **Proposed default** — a starting value that still needs validation or may change after pilot runs.
- **Measured finding** — a result produced by this repository and traceable to experiment output.

Never turn a proposed default or illustrative example into a measured claim.

## Working model facts

The project currently treats the official Qwen model card as the primary source for model-level facts. At the time this project was planned, the model card described Qwen3.8-27B as a roughly 27B dense model with long-context, multimodal, coding, and agent-oriented capabilities. Any exact parameter counts, context limits, architecture details, or runtime features used in the talk must be rechecked against the current official model card before presentation freeze.

Primary model reference:

- https://huggingface.co/Qwen/Qwen3.8-27B

## Philosophy

The benchmark target is not “the model in isolation.” The target is the local system people actually want to use:

```text
quantized model
+
large local context
+
files / repository
+
tools
+
agent history
+
consumer hardware constraints
```

The project should therefore optimize for reproducible, interpretable measurements of **useful local behavior**, not benchmark quantity.
