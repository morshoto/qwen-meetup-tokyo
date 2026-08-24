# Related work

**Issue:** #2 — Build related-work map for long context, quantization, and local agents

This document is a research map, not a bibliography dump. For every reference, ask:

1. What question did the work ask?
2. How did it measure the phenomenon?
3. What did it find?
4. What limitation matters for us?
5. Which experiment in this repository does it influence?

Exact paper metadata and links should be rechecked before presentation freeze.

---

## 1. Long-context position bias

### Lost in the Middle: How Language Models Use Long Contexts

Reference:

- https://arxiv.org/abs/2307.03172

Core idea:

Long-context language models can show strong dependence on the position of relevant information. Performance may be stronger when relevant evidence is near the beginning or end and weaker when it is in the middle.

Why it matters here:

Our exp_001 uses controlled evidence positions, but should not merely reproduce the original setup. We extend the question along three dimensions:

- a newer 27B-class model;
- semantic and multi-hop task difficulty;
- later interaction with quantization and agent-history growth.

What to borrow:

- relative evidence-position manipulation;
- position-conditioned performance curves;
- careful comparison at fixed total context.

What not to assume:

- that Qwen3.8 must exhibit a U-shape;
- that literal retrieval predicts reasoning or agent reliability.

Affected experiments:

- exp_001;
- exp_003;
- exp_004.

### Recent positional-bias follow-up work

During planning we found 2026 work continuing to report positional bias and propose mitigations. These should be verified and summarized with exact citations in this section before the talk.

Relevant themes to extract:

- whether bias increases with context length;
- whether smaller models are more sensitive;
- whether self-consistency or repeated sampling mitigates or amplifies bias;
- RoPE/position-scaling interventions;
- task dependence of position effects.

Engineering implication:

The benchmark must record actual token offsets rather than assuming an inserted paragraph remains at exactly 50% after chat/system formatting.

---

## 2. Effective-context benchmarks

### RULER

Reference:

- https://arxiv.org/abs/2404.06654

Core idea:

Needle-in-a-haystack alone can overestimate long-context capability. RULER evaluates multiple synthetic task types and shows that performance can degrade well before nominal context limits depending on task complexity.

Why it matters here:

It motivates our distinction between:

```text
maximum accepted context
vs
usable/effective context
```

What to borrow:

- multiple task families;
- context-length sweeps;
- reporting degradation relative to shorter context.

Our difference:

We add quantization and local-systems telemetry, then transfer the phenomenon into agent/repository workloads.

Affected experiments:

- exp_001;
- exp_003.

### HELMET

Reference:

- https://arxiv.org/abs/2410.02694

Core idea:

Long-context evaluation should cover realistic downstream behavior rather than only retrieval needles. Strong simple-retrieval scores do not necessarily predict broader long-context performance.

Why it matters:

It supports our task ladder and our decision to validate synthetic findings with repository/agent tasks.

Affected experiments:

- exp_001;
- exp_005.

### NoLiMa

Reference:

- https://arxiv.org/abs/2502.05167

Core idea:

Long-context tests become substantially harder when lexical matching between query and evidence is removed.

Why it matters:

This directly motivates the semantic-retrieval task family.

Our test should distinguish:

```text
literal key lookup
from
semantic identification
```

Affected experiment:

- exp_001 and exp_003.

### LongBench / LongBench v2 and related suites

References to verify/include before freeze:

- original LongBench;
- LongBench v2 / newer long-context reasoning work;
- repository/code-oriented long-context evaluations.

Why it matters:

These benchmark families demonstrate that long-context ability is multi-dimensional and give examples of code/repository tasks.

Our goal is not to reproduce the entire suite. We use them to justify task diversity and design a smaller controlled study suitable for repeated local execution.

---

## 3. Multi-hop reasoning in long context

Recent work should be reviewed for “weakest-link” effects: a multi-hop question may fail if one required evidence span is poorly accessible even when other evidence is easy to retrieve.

Implication for exp_001:

For every multi-hop instance, store the token positions of all evidence spans individually. Do not reduce a multi-hop task to one average position value.

Useful analysis questions:

- is success governed by the least-accessible span?
- does placing one hop near the middle dominate outcome?
- does quantization change the weakest-link behavior?

---

## 4. Quantization

### General background

Local deployment typically uses reduced-precision model artifacts because full-precision weights consume substantially more memory. However, “Q4” or “Q8” is not a complete method description.

Relevant dimensions include:

- weight-only vs weight+activation schemes;
- group/block size;
- mixed precision;
- outlier handling;
- tensor-specific precision;
- quantization-aware runtime kernels;
- KV/cache precision;
- artifact conversion method.

Therefore this project compares **specified artifacts/methods**, not abstract bit labels.

### Adaptive / mixed-precision work

During project research we identified recent 2026 work on adaptive mixed-precision quantization, including RAMP and related approaches. Exact details and claims should be verified from primary papers before using them in slides.

Research lesson for us:

The modern question is increasingly not simply:

> Is 4-bit worse than 8-bit?

but:

> **Where is precision actually necessary for preserving useful behavior?**

Our behavioral analogue asks which capabilities fail first under low precision, especially as context grows.

Affected experiments:

- exp_002;
- exp_003;
- exp_004.

### Long-context quantization literature

Track papers that specifically evaluate low-bit models beyond 64K context. Important questions:

- does degradation accelerate with context length?
- which quantizers are robust?
- are retrieval and reasoning affected differently?
- does cache precision matter separately from weight precision?

Do not cite a dramatic worst-case percentage without preserving the method/model/task context.

---

## 5. KV cache and context-memory efficiency

Long contexts consume memory not only through model weights but also through runtime state/cache.

Relevant research themes:

- KV-cache quantization;
- token-adaptive cache precision;
- cache eviction/compression;
- paged KV cache;
- prefix caching;
- context reuse.

Why it matters:

Our initial core experiments should keep cache policy fixed so weight-quantization results are interpretable. Cache optimization becomes a stretch experiment after exp_003.

Systems lesson:

Peak memory as context grows should be recorded even if the project does not manipulate cache precision directly.

---

## 6. Speculative decoding and Multi-Token Prediction

Recent inference work in 2026 has continued exploring speculative decoding, including block/diffusion drafting and tree/parallel approaches. Qwen3.8's model/runtime ecosystem may expose Multi-Token Prediction or speculative acceleration paths.

Example research thread identified during planning:

- DFlash / block diffusion speculative decoding;
- follow-up diffusion/parallel drafting systems;
- benchmarking work questioning whether theoretical speedups survive production workloads.

Why it matters:

A reported “6× speedup” in a paper is not automatically the speedup on our local agent workload.

Potential stretch experiment:

```text
same model/artifact/task
normal decoding
vs
MTP/speculative path
```

Measure:

- TTFT;
- decode throughput;
- total task time;
- long-context dependence;
- result equivalence/capability.

Do not include this in the core matrix until exp_001–003 are stable.

---

## 7. Agentic edge/local evaluation

### MLPerf Edge Agentic Inference

Primary source to verify before freeze:

- https://mlcommons.org/

During planning, MLCommons' 2026 edge-agentic work was especially relevant because it treats tool calling and multi-turn coding-style trajectories as an edge inference workload rather than evaluating only single-turn QA.

Why it matters:

It validates the broader premise of exp_004:

> Local/edge inference increasingly includes long, growing histories created by tool-using agents.

What to borrow conceptually:

- function/tool-call validity;
- multi-turn trajectories;
- latency plus accuracy;
- quantized local model deployment.

Our difference:

We explicitly manipulate the age/position of a critical observation inside the growing trajectory — **Lost in the Agent**.

---

## 8. Repository-level code reasoning

Review current repository-level code understanding/agent benchmarks for:

- code retrieval;
- issue diagnosis;
- test-based task completion;
- context selection;
- large-repository reasoning.

Relevant lesson from recent work:

More retrieved code is not automatically better. Excessive irrelevant context can make identification and reasoning harder.

This motivates exp_005's controlled comparison:

```text
curated files
vs
local neighborhood
vs
broad repository context
```

Avoid using a benchmark task whose solution is likely memorized by the model if the purpose is to study context behavior.

---

## 9. Local inference on consumer hardware

The talk should include recent empirical work comparing local inference across hardware classes, especially memory-capacity vs throughput/energy trade-offs.

Useful themes:

- unified-memory systems can fit larger artifacts than discrete GPUs with similar price/power envelopes;
- memory bandwidth strongly influences decode throughput;
- long-context prefill may behave differently from token-by-token decode;
- energy-per-task may tell a different story than tokens/second.

Our project should avoid generalizing one machine's performance to “local LLMs” universally.

---

## 10. Recent local/Qwen talks and community signal

During planning, recent Qwen/local-model meetup and community discussions clustered around:

- agents rather than simple chat;
- benchmarking models on one's own hardware/workload;
- Apple Silicon/local runtime optimization;
- speculative decoding/MTP;
- memory/performance trade-offs;
- production/tool-use reliability.

This is useful as topic-selection evidence, not as scientific evidence.

Community reports can inspire experiments but should never substitute for controlled measurement or primary-paper citations.

---

## 11. Research gap this project targets

The gap is not that nobody has studied long context or quantization.

The gap we are trying to make useful for a meetup audience is the **intersection**:

```text
modern 27B local model
×
quantization
×
long-context position/task difficulty
×
growing agent history
×
real local systems measurements
```

Specifically, we want to answer whether a model that appears robust at short context remains reliable when:

1. compressed;
2. given much larger context;
3. asked to use semantically distributed evidence;
4. allowed to build its own long trajectory.

That intersection is the core novelty of the presentation, not any single benchmark technique.

## 12. Related-work maintenance

For each added paper/talk, use this template:

```markdown
### Title

- Link:
- Published:
- Venue/status:

Question:

Method:

Key finding:

Limitations / caveats:

Why it matters to us:

Affected experiment(s):
```

Before presentation freeze:

- [ ] verify all titles/authors/dates from primary sources;
- [ ] remove unverified community claims from scientific-background slides;
- [ ] cite official Qwen documentation for model facts;
- [ ] identify which results are prior work vs our own measurements.
