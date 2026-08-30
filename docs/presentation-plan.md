# Presentation plan

**Issue:** #13 — Synthesize findings into the meetup presentation

This is a **presentation plan**, not the final slide deck. The narrative should change if the experiments produce a different strongest finding.

As of 2026-08-31 this should be presented as an intermediate measurement
report. exp_001 is fixture/harness-only, exp_002 has one calibrated Q8_0 pilot
condition, and exp_003 has only a legacy smoke audit; no full capability or
repository-transfer claim is frozen.

## 1. Audience promise

The audience should leave knowing:

1. why nominal context length does not establish useful capability;
2. whether local quantization changes useful long-context behavior;
3. what happens when a local model's own agent history grows;
4. which local configuration appears practically attractive under the measured hardware/runtime;
5. where the experiment is limited.

## 2. Core narrative

A research-style talk should follow:

```text
question
→ prior work / why unsure
→ experimental design
→ results
→ failure analysis
→ systems trade-off
→ implication
```

Avoid:

```text
model specs
→ benchmark table
→ feature list
→ demo
```

## 3. Opening

Suggested opening claim/question:

> Qwen3.8-27B can accept a very large context and can be run locally in quantized form. But “it fits” and “it can accept the prompt” do not tell us how much useful capability survives.

Then:

> **When does a local LLM start to break?**

Introduce the three stressors:

```text
more context
lower precision
longer agent history
```

## 3a. Current 12-slide version

Use this shorter sequence until the main matrices are measured:

1. title and question: when does a local LLM start to break?
2. break definition: capability, systems cost, and trajectory reliability;
3. experiment map: what is measured, pilot, harness-only, or unrun;
4. scorer calibration: exact, answer-bearing, and format-valid outcomes;
5. exp_002 artifact footprint (Q8_0 29.05 GB → Q4_K_M 16.81 GB);
6. long-input cost: stream TTFT is a proxy, not a native kernel counter;
7. exp_003 legacy smoke: `insufficient_data`, not an interaction result;
8. agent transition: static context versus self-generated history;
9. exp_004 observed end-state/tool-validity split;
10. what is known and unknown, including backend and synthetic-task limits;
11. measurement quality gate and claim-evidence provenance;
12. next measurements and conclusion.

Do not use a Q4/Q8 quality ranking, an effective-context number, a position-bias
claim, or a 262K capability claim until the corresponding calibrated matrix is
complete.

## 4. Act I — Experimental subject

Spend only enough time on Qwen3.8-27B to explain why it is a useful subject:

- 27B-class dense local model;
- long-context support;
- coding/agent orientation;
- practical quantized deployment;
- selected runtime/hardware.

All exact model facts must be cited to current official documentation.

Do not make the talk a Qwen product overview.

## 5. Act II — “Context length” is not one capability

Explain:

```text
accepted context
≠ retrieval context
≠ reasoning context
≠ agent context
```

Introduce prior Lost-in-the-Middle/effective-context research briefly.

Then show exp_001 methodology:

```text
context length × evidence position × task difficulty
```

Ideal first result slide:

- position curves or heatmap;
- one sentence measured takeaway.

## 6. Act III — Compress the model

Explain why quantization is unavoidable/practical locally.

Do not frame:

> Q4 is bad, Q8 is good.

Frame:

> What ability do we trade for the memory we save?

Show exp_002 Pareto plot, then exp_003 interaction.

Potential headline formats, filled only after results exist:

- “Short-context accuracy survived Q4; long-context semantic retrieval did not.”
- “Quantization barely moved the context threshold.”
- “Position bias dominated precision.”

## 7. Act IV — Lost in the Agent

This should be the conceptual centerpiece if exp_004 yields meaningful evidence.

Visualize a growing trajectory:

```text
objective
↓
read file
↓
discover fact A
↓
search
↓
test output
↓
more files
↓
...
↓
need fact A again
```

Ask:

> Can the model use something important it learned thirty actions ago?

Show task-success/state-reuse vs history length/critical-observation position.

## 8. Act V — Real repository validation

Use exp_005 to answer the audience's natural objection:

> Does any of this matter outside synthetic needles?

Compare:

```text
curated context
vs
more repository context
```

One strong case study can accompany aggregate data, but never replace it.

## 9. Act VI — Systems reality

Report:

- peak memory;
- prefill speed/time;
- decode speed;
- total task time;
- optional energy.

Key message:

Long-context agents may be bottlenecked by repeatedly ingesting context, not just token generation.

Use a Pareto-style view if possible.

## 10. Failure section

Reserve meaningful time for failures.

Good examples:

- evidence was present but ignored;
- correct fact discovered then forgotten;
- agent repeatedly opened wrong files;
- broad context harmed task success;
- quantized variant produced a structurally valid but semantically wrong tool call.

Pair anecdotes with the cross-experiment error taxonomy.

## 11. Tentative full slide sequence

For ~20–25 minutes:

1. Title — When Does a Local LLM Start to Break?
2. Why local models are now an interesting systems problem
3. Why Qwen3.8-27B is the subject
4. Research questions / hypotheses
5. “262K context” vs effective context
6. Prior Lost-in-the-Middle result (brief)
7. exp_001 design
8. exp_001 result — position
9. exp_001 result — task difficulty/effective context
10. Why quantize?
11. exp_002 memory/performance frontier
12. exp_003 design
13. exp_003 interaction heatmap
14. Main quantization-context takeaway
15. Static prompts are not agents
16. Lost in the Agent design
17. exp_004 result
18. exp_005 repository validation
19. Failure taxonomy
20. Prefill/memory/local systems results
21. Local intelligence Pareto frontier
22. What surprised us / null results
23. Limitations
24. Conclusions

The actual deck may be shorter once findings are known.

## 12. Lightning-talk path

If time is very short, keep only:

1. Question: “When does a local LLM start to break?”
2. Experiment matrix.
3. Best context-position figure.
4. Best quantization × context figure.
5. Lost in the Agent result.
6. One practical local configuration takeaway.
7. Limitations/conclusion.

## 13. Candidate final messages

Choose only one or two, based on measured evidence:

### Context message

> A context window is a capacity specification; effective context is a capability measurement.

### Agent message

> For agents, the hardest information to remember may be information they discovered themselves.

### Systems message

> The local limit may be context processing and memory behavior before it is raw model intelligence.

### Quantization message

> The right question is not how many bits we remove, but which useful capabilities disappear first.

## 14. Claim-evidence rule

For every final slide claim, maintain a mapping:

```text
claim
→ experiment ID / prior-work citation
→ figure/table
→ notebook/result manifest
```

No claim should depend only on speaker intuition.

## 15. Limitations slide

At minimum mention:

- one primary model family;
- one/few local runtimes;
- one hardware environment;
- limited task-instance count at the longest contexts if applicable;
- quantization method specificity;
- synthetic tasks cannot fully model real workflows;
- repository/agent tasks are a small validation set;
- context thresholds are tested checkpoints, not exact universal constants.

## 16. Presentation freeze checklist

- [ ] all model facts rechecked against official documentation;
- [ ] all prior-work citations verified from primary sources;
- [ ] every result figure is measured, reproducible, and labeled;
- [ ] hypothetical numbers removed;
- [ ] quantization formats described precisely;
- [ ] sample counts and uncertainty available;
- [ ] null/negative results included where material;
- [ ] talk thesis matches the strongest evidence, not the original hypothesis;
- [ ] limitations explicit;
- [ ] repository reproduction instructions tested.
