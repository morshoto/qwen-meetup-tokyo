# Related work

**Issue:** #2 — Build related-work map for long context, quantization, and
local agents

**Review date:** 2026-08-25

This document is a research map, not a flat bibliography. For every reference,
it records the question, method, key finding, limitation, and experiment idea
for this repository. Primary papers, official model documentation, and dated
community/event pages are separated by their evidentiary status.

The project is not trying to reproduce every benchmark below. Its proposed
contribution is a controlled local study of one current 27B-scale Qwen model
that measures the interaction among context length, precision, retained agent
history, and local systems cost.

## 1. Target model and local ecosystem

### Qwen Team — Qwen3.8-27B model card (2026-08-14)

[Model card](https://huggingface.co/Qwen/Qwen3.8-27B) · [official repository](https://github.com/QwenLM/Qwen3.8)

- **Research question:** What capabilities and deployment configurations does
  Qwen3.8-27B provide as an open local model?
- **Method:** The official card documents a 27B native vision-language model,
  262,144 native context tokens extendable to about one million, MTP training,
  thinking controls, and serving paths for Transformers, vLLM, SGLang, and
  other runtimes.
- **Key finding:** Qwen positions the model for coding, research, professional
  work, and long-horizon agentic tasks; the card exposes the exact model ID used
  by this repository.
- **Limitation:** Release/model-card results are not a controlled comparison of
  this repository's prompts, hardware, or quantization variants. Advertised
  context length is not effective context length.
- **Experiment idea:** Pin `Qwen/Qwen3.8-27B` to an immutable revision, test the
  256k operating point at multiple lengths and positions, and record whether
  MTP and thinking are on or off in every run.

### Unsloth — Qwen3.8-27B local inference guide (accessed 2026-08-25)

[Local model guide](https://unsloth.ai/models/qwen3.8-27b)

- **Research question:** What quantized formats and local memory budgets make
  Qwen3.8-27B practical on consumer hardware?
- **Method:** The guide lists GGUF/NVFP4 options, approximate memory bands from
  2-bit through BF16, hybrid thinking presets, and local agent integrations.
- **Key finding:** A 4-bit deployment is presented as a practical
  quality/memory compromise, with materially different resource requirements
  across bit-widths.
- **Limitation:** Deployment guidance is hardware- and runtime-dependent;
  headline memory figures do not replace task-level accuracy and latency
  measurements.
- **Experiment idea:** Use 4-bit as a candidate operating point, not a default
  winner. Compare q4/q5/q6/q8 and reference precision on the same tasks,
  including KV-cache settings.

## 2. Long-context position bias

### Liu et al. — Lost in the Middle (2023-07-06)

[Paper](https://arxiv.org/abs/2307.03172) · [code and data](https://github.com/nelson-liu/lost-in-the-middle)

- **Research question:** Do language models use all positions in a long input
  equally when relevant information is present?
- **Method:** Controlled multi-document QA and key-value retrieval with the
  relevant item moved through the context.
- **Key finding:** Performance is often strongest when evidence is near the
  beginning or end and degrades when it is in the middle.
- **Limitation:** The tasks emphasize retrieval and do not fully characterize
  semantic composition, multi-hop reasoning, runtime memory, or agent state.
- **Experiment idea:** `exp_001` uses matched evidence positions at 5%, 25%,
  50%, 75%, and 95%, then adds semantic and multi-hop tasks. Estimate the
  paired middle-vs-edge difference directly instead of relying on marginal CI
  overlap.

### Baker et al. — Lost in the Middle, and In-Between (2024-12-13)

[Paper](https://arxiv.org/abs/2412.10079)

- **Research question:** Does positional bias become more severe when a
  multi-hop answer depends on several pieces of evidence spread through input?
- **Method:** Multi-hop QA with relevant documents at different positions and
  separations, plus reduction and prompting interventions.
- **Key finding:** Multi-hop performance can degrade with both evidence position
  and distance between necessary pieces.
- **Limitation:** It is a benchmark intervention rather than a deployment study
  of quantized local inference or long agent trajectories.
- **Experiment idea:** Store every hop's token position and log which hop was
  first missed in each failure.

### Byerly and Khashabi — Self-Consistency Falls Short (TACL 2026)

[ACL Anthology](https://aclanthology.org/2026.tacl-1.15/) · [PDF](https://aclanthology.org/2026.tacl-1.15.pdf)

- **Research question:** Does self-consistency help long-context tasks in the
  same way it helps short-context reasoning?
- **Method:** Experiments across models, tasks, context lengths, and
  self-consistency formulations, analyzing position bias and repeated samples.
- **Key finding:** Self-consistency can fail to help and can actively degrade
  long-context performance because it amplifies positional errors; the effect
  worsens with longer context and smaller models.
- **Limitation:** The result concerns repeated sampling and does not isolate
  quantization, local hardware, or tool-state accumulation.
- **Experiment idea:** Keep primary `exp_001` greedy and deterministic, then
  make self-consistency an explicitly labeled stretch condition rather than
  assuming more samples repair position bias.

### Wang et al. — Layer-Specific Positional Embedding Scaling (ACL Findings 2026)

[ACL Findings paper](https://aclanthology.org/2026.findings-acl.1059/) · [arXiv](https://arxiv.org/abs/2606.27705)

- **Research question:** Can position bias be reduced by assigning different
  positional-embedding scaling factors to different layers?
- **Method:** LPES searches layer-specific scaling factors, using a genetic
  algorithm and Bézier curves, without fine-tuning model weights or adding
  multiple inference passes.
- **Key finding:** The paper reports more balanced attention and improvements on
  several long-context benchmarks, including a reported gain on key-value
  retrieval.
- **Limitation:** It is a mitigation method, not evidence that an unmodified
  Qwen3.8 deployment is unbiased; transfer to this architecture and runtime is
  untested here.
- **Experiment idea:** Treat positional mitigation as out of scope for v1, but
  use LPES as a reason to preserve raw position-conditioned measurements before
  any RoPE or runtime intervention.

## 3. Effective-context benchmarks

### Bai et al. — LongBench (2023-08-28)

[Paper](https://arxiv.org/abs/2308.14508) · [project](https://github.com/THUDM/LongBench)

- **Research question:** How can long-context understanding be evaluated across
  languages and task types rather than with one retrieval test?
- **Method:** A bilingual, multi-task suite covering 21 datasets and six task
  categories, including QA, summarization, few-shot learning, synthetic tasks,
  and code completion.
- **Key finding:** Performance varies across tasks; position extension and
  retrieval can help, but retrieval does not erase the gap to models with
  stronger long-context ability.
- **Limitation:** Its breadth and fixed datasets are heavier than a small
  talk-focused causal matrix, and format or contamination effects remain
  practical concerns.
- **Experiment idea:** Use its taxonomy to build a small versioned fixture set
  whose evidence and answer key can be moved across positions and reused across
  precision variants.

### Hsieh et al. — RULER (2024-04-09)

[Paper](https://arxiv.org/abs/2404.06654) · [project](https://github.com/NVIDIA/RULER)

- **Research question:** What is a model's real usable context size beyond a
  simple needle-in-a-haystack score?
- **Method:** Flexible synthetic tasks with multiple needles, multi-hop
  tracing, aggregation, and QA over configurable lengths and difficulty.
- **Key finding:** Models that pass vanilla retrieval can still degrade sharply
  as length and task complexity increase; nominal context size can exceed
  satisfactory useful context.
- **Limitation:** Synthetic generators can overrepresent template skills and do
  not by themselves measure local memory or latency.
- **Experiment idea:** Define an explicit effective-context breakpoint and add
  multi-hop/aggregation-like fixtures while keeping failures inspectable.

### Yen et al. — HELMET (2024-10-03)

[Paper](https://arxiv.org/abs/2410.02694) · [project](https://github.com/princeton-nlp/HELMET)

- **Research question:** Which evaluation choices produce reliable long-context
  rankings and useful downstream signals?
- **Method:** Seven application-centered categories, controllable lengths up to
  128k, few-shot prompting, and model-based evaluation for selected tasks.
- **Key finding:** NIAH-style scores are weak predictors of downstream
  performance; task categories have distinct trends and open models can lag on
  full-context reasoning and complex instruction following.
- **Limitation:** Broad coverage and model-based judging add cost and
  judge-dependence to a small local study.
- **Experiment idea:** Keep exact/structured grading primary and include a few
  application-shaped tasks alongside retrieval.

### Modarressi et al. — NoLiMa (2025-02-07)

[Paper](https://arxiv.org/abs/2502.05167) · [project](https://github.com/adobe-research/NoLiMa)

- **Research question:** How much long-context ability remains without literal
  lexical overlap between query and evidence?
- **Method:** Needle-in-a-haystack tasks designed with minimal lexical overlap,
  evaluated across long contexts.
- **Key finding:** Many models perform well at short lengths but fall below half
  of their short-context baseline at longer lengths when literal matching is
  unavailable.
- **Limitation:** It isolates retrieval difficulty and does not measure tool
  state, quantization, or end-to-end task completion.
- **Experiment idea:** Use controlled semantic retrieval fixtures so H2/H4
  distinguish literal lookup from evidence use.

### Bai et al. — LongBench v2 (2024-12-19)

[Paper](https://arxiv.org/abs/2412.15204) · [project](https://longbench2.github.io/)

- **Research question:** Can long-context evaluation test deeper reasoning over
  realistic documents and tasks rather than retrieval alone?
- **Method:** 503 challenging multiple-choice questions across six categories,
  with contexts from 8k to 2M words and human review.
- **Key finding:** Realistic long-context reasoning remains difficult even for
  strong systems; longer reasoning/test-time compute changes results.
- **Limitation:** Scale, licensing, and word-length ranges do not fit the first
  talk's controlled local matrix.
- **Experiment idea:** Select a few inspectable reasoning categories while
  measuring token lengths and intermediate failures directly.

## 4. Multi-hop and repository reasoning

Recent long-context work suggests a weakest-link effect: a multi-hop answer can
fail when one required span is poorly accessible even if other evidence is
easy to retrieve. For `exp_001`, record each required span's token position and
ask:

- Is success governed by the least-accessible span?
- Does placing one hop near the middle dominate the outcome?
- Does quantization change this weakest-link behavior?

For repository tasks, the same concern appears as file selection. `exp_005`
therefore pins the repository and revision, records every selected file, and
compares curated files with a local neighborhood and broad context. The goal is
not to reproduce a repository benchmark, but to test whether synthetic context
findings survive a machine-checkable coding workload.

## 5. Weight, activation, and mixed-precision quantization

“Q4” or “Q8” is not a complete method description. Relevant dimensions include
weight-only versus weight+activation schemes, group size, outlier handling,
tensor-specific precision, calibration data, runtime kernels, and KV-cache
precision. This project compares specified artifacts and methods, not abstract
bit labels.

### Frantar et al. — GPTQ (2022-10-31)

[Paper](https://arxiv.org/abs/2210.17323) · [code](https://github.com/IST-DASLab/gptq)

- **Research question:** Can large generative transformers be post-training
  quantized to very low weight precision without unacceptable accuracy loss?
- **Method:** One-shot, data-aware weight quantization using approximate
  second-order information, including 3- and 4-bit settings.
- **Key finding:** Weight-only low-bit compression can reduce storage and make
  large models runnable while preserving much of reported quality.
- **Limitation:** Weight-only results do not predict activation or KV-cache
  behavior; speed depends on kernels, hardware, batch size, and context.
- **Experiment idea:** Make quantizer, calibration data, packing format, and
  runtime part of the experimental identity; report measured throughput.

### Xiao et al. — SmoothQuant (2022-11-18)

[Paper](https://arxiv.org/abs/2211.10438) · [code](https://github.com/mit-han-lab/smoothquant)

- **Research question:** Can activation outliers be handled for efficient
  weight-and-activation INT8 inference?
- **Method:** An equivalent scaling transformation migrates quantization
  difficulty from activations to weights, enabling W8A8 PTQ.
- **Key finding:** W8A8 can reduce memory and improve speed with small reported
  accuracy loss across tested model families.
- **Limitation:** The result is not a direct prediction for Qwen3.8-27B,
  4-bit weight-only formats, or long-context KV pressure.
- **Experiment idea:** Treat W8A8 and W4A16 as distinct operating points and
  measure prefill and decode separately.

### Lin et al. — AWQ (2023-06-01)

[Paper](https://arxiv.org/abs/2306.00978) · [code](https://github.com/mit-han-lab/llm-awq)

- **Research question:** Can activation statistics identify weight channels
  whose precision matters most for low-bit local inference?
- **Method:** Activation-aware scaling protects salient channels while keeping
  a hardware-friendly low-bit weight-only format.
- **Key finding:** Protecting a small set of salient channels can retain quality
  while enabling practical 4-bit inference.
- **Limitation:** Calibration distribution and kernels matter; the method does
  not answer whether long-context evidence is unusually sensitive.
- **Experiment idea:** Preserve calibration provenance and test the same
  artifact at short and long context under H3.

### Singh Gautam and Jha — RAMP (2026-03-18)

[Paper](https://arxiv.org/abs/2603.17891)

- **Research question:** Can adaptive per-layer precision improve the
  size/quality trade-off for on-device LLM inference over uniform bit widths?
- **Method:** A reinforcement-learning policy assigns per-layer bit widths
  under a global budget using activation, weight, and structural features; the
  work also introduces Scale Folding and exports allocations to GGUF.
- **Key finding:** On reported Llama 2 and Mistral tests, adaptive assignments
  can improve the perplexity/memory trade-off over uniform AWQ/GPTQ baselines.
- **Limitation:** The paper does not establish transfer to Qwen3.8-27B or to
  repository/agent correctness; the reported runtime and quality trade-offs
  remain artifact- and hardware-dependent.
- **Experiment idea:** Use RAMP to sharpen H3's question—whether the first
  failures depend on where precision is removed—but keep it stretch work until
  a reproducible Qwen artifact and kernel path exist.

### Zhang et al. — QQQ (2024-06-17; v3 2024-07-31)

[Paper](https://arxiv.org/abs/2406.09904) · [code](https://github.com/HandH1998/QQQ)

- **Research question:** Can W4A8 improve both prefill and decode efficiency
  without the quality loss often associated with mixed precision?
- **Method:** Adaptive smoothing and Hessian-based compensation with W4A8
  kernels.
- **Key finding:** W4A8 is a distinct quality/speed point; reported results
  improve over selected W8A8 and W4A16 baselines in tested settings.
- **Limitation:** Hardware-specific kernels and model coverage limit transfer.
- **Experiment idea:** If a W4A8 path is available, include it as a labeled
  stretch variant and measure prefill and decode separately.

### Egiazarian et al. — AQLM (2024-01-09)

[Paper](https://arxiv.org/abs/2401.06118) · [code](https://github.com/Vahe1994/AQLM)

- **Research question:** Can additive/codebook quantization push useful weight
  compression below the usual 4-bit range?
- **Method:** Calibration-aware additive quantization with multiple codebooks
  and block-level optimization.
- **Key finding:** The paper reports a stronger 2–4-bit quality/compression
  frontier on selected Llama-family models.
- **Limitation:** It is not a guarantee for Qwen3.8-27B, and very low-bit
  formats may have limited runtime support or different latency.
- **Experiment idea:** Keep sub-4-bit methods out of the first factorial unless
  their runtime path is stable; otherwise report feasibility separately.

## 6. KV-cache compression and context-memory efficiency

Long contexts consume memory through both weights and runtime state. The core
experiments keep cache policy fixed so weight-quantization results remain
interpretable; cache optimization is a stretch factor after `exp_003`.

### Liu et al. — KIVI (2024-02-05)

[Paper](https://arxiv.org/abs/2402.02750) · [code](https://github.com/jy-yuan/KIVI)

- **Research question:** Can the KV cache be quantized asymmetrically at 2 bits
  without losing most model quality?
- **Method:** Per-channel key quantization, per-token value quantization, and a
  tuning-free asymmetric scheme.
- **Key finding:** KV cache becomes a memory bottleneck at long context; the
  method reports substantial memory reduction on tested model families.
- **Limitation:** KV quantization is separate from weight quantization, and its
  error interacts with attention, task type, position, and reasoning length.
- **Experiment idea:** Record KV precision explicitly in every quantization run
  and add it as a separate factor only after weight-only H3 is stable.

### Hooper et al. — KVQuant (2024-01-31)

[Paper](https://arxiv.org/abs/2401.18079)

- **Research question:** Can structured and non-uniform KV quantization support
  very long contexts at low precision?
- **Method:** Per-channel key, pre-RoPE, sensitivity-weighted non-uniform, and
  dense/sparse outlier handling.
- **Key finding:** The paper reports low perplexity degradation at 3-bit KV
  precision and larger feasible context on tested Llama/Mistral systems.
- **Limitation:** Perplexity and single-model throughput do not establish task
  reliability for Qwen3.8-27B.
- **Experiment idea:** Treat memory headroom and task accuracy as separate
  dependent variables; do not infer effective context from allocation alone.

### Li et al. — SnapKV (2024-04-22)

[Paper](https://arxiv.org/abs/2404.14469) · [code](https://github.com/FasterDecoding/SnapKV)

- **Research question:** Can important KV positions be selected before decoding
  to shrink the cache while preserving long-context quality?
- **Method:** An observation window and attention-head-specific clustering select
  important prompt positions.
- **Key finding:** The paper reports memory and decoding benefits on long
  sequence datasets and larger feasible contexts on a single GPU.
- **Limitation:** Selection can discard evidence needed by another task or a
  later agent turn.
- **Experiment idea:** Use `exp_005` to compare deployable curation with broad
  context, while keeping oracle selection as an upper-bound diagnostic only.

## 7. Speculative decoding and MTP

### Leviathan et al. — Speculative Decoding (2022-11-30)

[Paper](https://arxiv.org/abs/2211.17192) · [ICML version](https://proceedings.mlr.press/v202/leviathan23a.html)

- **Research question:** Can a small draft model propose multiple tokens while
  a large model verifies them without changing the target distribution?
- **Method:** Speculative execution and rejection sampling verify several draft
  tokens in a parallel pass.
- **Key finding:** Decode can accelerate without changing target output when the
  draft model is a good match.
- **Limitation:** Acceptance rate, draft overhead, batching, and KV memory
  determine practical speed; speculation does not improve context use.
- **Experiment idea:** Keep it out of the core capability comparison; later
  report acceptance rate and end-to-end latency separately.

### Cai et al. — Medusa (2024-01-19)

[Paper](https://arxiv.org/abs/2401.10774) · [code](https://github.com/FasterDecoding/Medusa)

- **Research question:** Can extra decoding heads replace a separate draft model
  for multi-token prediction?
- **Method:** Multiple heads propose future tokens and tree attention verifies
  candidates.
- **Key finding:** The reported method reduces serial decode steps, with speed
  depending on head training and workload.
- **Limitation:** Extra heads and runtime support are not free, and decode speed
  says nothing about long-context correctness.
- **Experiment idea:** Treat MTP/speculation as stretch systems work after the
  base decode path is measured.

## 8. Agentic and edge evaluation

### Liu et al. — AgentBench (2023-08-07)

[Paper](https://arxiv.org/abs/2308.03688) · [project](https://github.com/THUDM/AgentBench)

- **Research question:** How do LLMs perform as agents across interactive
  environments rather than isolated language tasks?
- **Method:** Eight environments covering multi-turn open-ended interaction,
  tested with API-based and open-source models.
- **Key finding:** Long-term reasoning, decision-making, and instruction
  following are recurrent obstacles, with a reported gap between leading
  proprietary and open models.
- **Limitation:** Its breadth is expensive and aggregate scores do not isolate
  retained context, precision, or harness effects.
- **Experiment idea:** Build a smaller deterministic sandbox and log the first
  unrecoverable step for H5.

### Yao et al. — tau-bench (2024-06-17; ICLR 2025)

[Paper](https://arxiv.org/abs/2406.12045) · [code](https://github.com/sierra-research/tau-bench)

- **Research question:** Can an agent consistently interact with a user and
  domain APIs while following written policies?
- **Method:** Simulated user conversations, domain tools/policies, final
  database-state grading, and `pass^k` reliability.
- **Key finding:** Single-run success hides inconsistency; repeated-trial
  reliability is a distinct property.
- **Limitation:** Retail/airline environments differ from a local coding or
  research agent, and user simulation adds another model.
- **Experiment idea:** Borrow end-state grading and `pass^k` in a deterministic
  local fixture set.

### Lu et al. — ToolSandbox (2024-08-08; NAACL 2025 Findings)

[Paper](https://arxiv.org/abs/2408.04682) · [project](https://github.com/apple/ToolSandbox)

- **Research question:** How should tool use be evaluated when tools have state,
  implicit dependencies, and multi-turn conversation?
- **Method:** Stateful execution, user simulation, on-policy interaction, and
  intermediate/final milestone grading.
- **Key finding:** State dependency, canonicalization, and insufficient
  information are difficult even for strong models.
- **Limitation:** The full framework is larger than the first talk and does not
  isolate quantization × history length.
- **Experiment idea:** Use state dependencies and intermediate milestones in
  `exp_004`, with a small inspectable tool set.

### MLCommons — MLPerf Client / Edge Agentic Inference (2026)

[Client benchmark](https://mlcommons.org/benchmarks/client/) · [Edge Agentic data](https://inference.mlcommons-storage.org/index.html)

- **Research question:** How should agentic AI performance be measured on
  personal computers and edge systems?
- **Method:** Standardized workloads and performance measurement, including
  software-engineering/data-analyst agents and multi-turn/function-calling
  datasets.
- **Key finding:** Agentic inference needs system-level metrics and realistic
  workloads, not only tokens/second on one prompt.
- **Limitation:** Official rules, datasets, and hardware targets are broader
  than this talk; full reproduction is unnecessary for the five hypotheses.
- **Experiment idea:** Adopt the separation between accuracy and performance:
  report end-state success with TTFT, total latency, throughput, and memory.

## 9. Recent Qwen talks, events, and community experiments

### Qwen Team — Qwen3: Think Deeper, Act Faster (2025-04-28)

[Official announcement](https://qwenlm.github.io/blog/qwen3/) · [Qwen3 repository](https://github.com/QwenLM/Qwen3)

- **Question:** How can one open model family combine thinking/non-thinking
  modes with reasoning, coding, and agent behavior?
- **Method:** Official release, technical report, and benchmark suite across
  dense and MoE Qwen3 models.
- **Key finding:** Controllable reasoning and tool/agent use are presented as
  deployment modes, not just chat quality.
- **Limitation:** Release benchmarks are not a controlled study of Qwen3.8-27B
  under a fixed local hardware and precision budget.
- **Experiment idea:** Make thinking mode an explicit switch and include it in
  the run manifest so reasoning-token growth is not confused with context or
  quantization effects.

### Qwen Meetup Singapore (2026; accessed 2026-08-25)

[Qwen Events / Alibaba Cloud event listing](https://luma.com/yneoke5d)

- **Question:** What are local developers building with Qwen models in a
  community setting?
- **Method:** A Qwen Events meetup program with a Qwen3 application talk, an
  Alibaba Cloud application talk, open-mic demos, and community networking.
- **Key finding:** Recent community programming emphasizes concrete apps,
  local building, open-source/hosted comparisons, and practical workflows
  rather than only leaderboard scores.
- **Limitation:** An event listing is topic-selection evidence, not a controlled
  evaluation or primary research result; claims from talks require separate
  source checking.
- **Experiment idea:** Use it to justify a practical meetup narrative, while
  keeping the repository's evidence in reproducible task fixtures and logs.

### Community local-run reports (accessed 2026-08-25)

[Reproducible Ryzen/llama.cpp report](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/41) · [local inference benchmark](https://github.com/pxzleo/qwen3.8-27b-local-inference-benchmark) · [Atomic Chat quant comparison](https://atomic.chat/blog/guides/how-to-run-qwen-3-8-locally)

- **Question:** What happens when Qwen3.8-27B is run with consumer hardware,
  GGUF/NVFP4 variants, MTP, and different runtimes?
- **Method:** Community reports measure concrete model/quant/runtime/hardware
  combinations, sometimes publishing commands and throughput tables.
- **Key finding:** Local speed and memory are highly configuration-dependent;
  reports already explore q4–q6 formats, cache choices, and MTP.
- **Limitation:** Different prompts, hardware, versions, and quality tests make
  the reports engineering evidence, not controlled scientific comparisons.
- **Experiment idea:** Use them to choose feasible runtime candidates and list
  required controls. Re-run quality and latency on this repository's fixtures.

## 10. What is different about this repository?

Prior work gives us strong reasons not to collapse the problem into one
needle-in-a-haystack score or one quantized perplexity number. The planned
evaluation differs in five ways:

1. **One model, one local contract:** Qwen3.8-27B is pinned to a revision,
   runtime, and hardware setup.
2. **Matched causes:** the same evidence, tasks, and answer keys are reused
   across context lengths, positions, and precision variants.
3. **Capability × systems × trajectory:** correctness, memory/latency, and
   end-state agent reliability are measured together with first-failure traces.
4. **Synthetic-to-real transfer:** `exp_005-repository_reasoning` tests whether
   controlled findings survive pinned, machine-checkable coding tasks.
5. **Curated versus broad context:** the project tests whether context selection
   is more valuable than filling the window, while treating input length as an
   explicit systems variable.

This is a focused evaluation charter, not a new model or benchmark family.
Novelty, if supported by the data, will be the causal map of breakpoints for a
current local model and practical evidence for when curating context is better
than filling the window.

## 11. Experiment-to-literature map

| Planned experiment | Prior work that motivates it | What this repository adds |
| --- | --- | --- |
| `exp_001-context_measurement` | Lost in the Middle; RULER; HELMET; NoLiMa; LongBench; Byerly & Khashabi; LPES | A compact position × length × task matrix with paired bootstrap effects and local resource logging. |
| `exp_002-quantization_llama_cpp_gguf` | GPTQ; AWQ; SmoothQuant; QQQ; RAMP | A Qwen3.8-27B GGUF comparison naming the artifact/runtime and reporting task accuracy beside prefill/decode and memory. |
| `exp_003-context_x_quantization` | AWQ; QQQ; RAMP; KIVI; KVQuant | Directly tests whether precision changes the context breakpoint rather than assuming a constant penalty. |
| `exp_004-agent_context_growth` | AgentBench; tau-bench; ToolSandbox; MLPerf Edge Agentic | A deterministic local tool sandbox with intermediate validation, first-failure attribution, and repeated-trial reliability. |
| `exp_005-repository_reasoning` | HELMET; LongBench v2; NoLiMa; SnapKV; repository-level agent work | A pinned, machine-checkable software-engineering validation of curated, neighborhood, and broad context. |

## 12. Related-work maintenance

For each added paper, talk, or community report, preserve this template:

```markdown
### Title

- Link:
- Published/accessed:
- Venue/status:

Question:

Method:

Key finding:

Limitations / caveats:

Why it matters to us:

Affected experiment(s):
```

Before presentation freeze:

- verify titles, authors, dates, and claims from primary sources;
- remove unverified community claims from scientific-background slides;
- cite official Qwen documentation for model facts;
- identify which results are prior work versus repository measurements; and
- record the model/runtime/artifact revision for every local result.
