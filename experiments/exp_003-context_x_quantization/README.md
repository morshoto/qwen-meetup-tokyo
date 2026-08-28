# exp_003-context_x_quantization

## Goal

Measure whether quantization degradation is approximately constant across
context length or grows at long context, and whether lower precision amplifies
evidence-position effects.

This experiment reuses the exact `data/tasks/core.v001.jsonl`,
`prompt.qa.v001`, `ExpectedAnswerScorer`, and synthetic context generator used
by exp_001. It selects the `q8_0`, `q6_k`, `q5_k_m`, and `q4_k_m` artifacts from
the resolved exp_002 manifest. A context instance is generated once for each
task, context length, evidence position, and seed, then reused for every
quantization variant so matched-cell comparisons are valid.

## Controlled matrix

| Variable | Main values |
| --- | --- |
| Context length | 8K, 32K, 64K, 128K, 262K or highest practical |
| Evidence position | 5%, 25%, 50%, 75%, 95% |
| Quantization | Q8_0, Q6_K, Q5_K_M, Q4_K_M from exp_002 |
| Task family | literal, semantic, multi-hop |
| Repeats | 20 per cell where resources permit |
| Sampling | greedy, `temperature: 0.0`, max 64 generated tokens |

The committed config is the protocol. Artifact identity, model/tokenizer
revisions, runtime settings, and conversion provenance come from
`../exp_002-quantization_llama_cpp_gguf/results/manifest.json`; the output run
manifest records its SHA-256 so later source-manifest changes cannot silently
alter the interpretation of a run.

The output run manifest keeps inherited source runtime options separately from
the effective per-variant options used for execution, including derived
context capacity and model path settings.

The runner loads phase lengths, evidence positions, repeats, backend, and the
default quantization variants from `config.yaml`. Matching CLI flags are
explicit overrides for a selected run.

## Run phases

From the repository root, the fixture smoke run needs no model weights:

```bash
PYTHONPATH=src python3 experiments/exp_003-context_x_quantization/runner.py \
  --source-manifest experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json \
  --phase smoke --backend fixture
```

The pilot/main phases require the locally provisioned GGUF artifacts referenced
by the resolved exp_002 manifest and suitable hardware:

```bash
PYTHONPATH=src python3 experiments/exp_003-context_x_quantization/runner.py \
  --source-manifest experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json \
  --phase pilot --backend llama.cpp
```

The fixture backend validates matching, context construction, scoring, storage,
and coverage. It is not a Qwen measurement and must not be copied into
`docs/findings.md` as a model finding. Runtime/OOM/timeout cells remain in raw
results and are listed as exclusions with reasons rather than being silently
dropped.

## Analysis

`analysis.ipynb` loads the generated raw results, processed summary, and output
run manifest. It fails loudly when measured inputs are missing or incomplete and
produces:

1. context × quantization accuracy/degradation heatmaps by task type;
2. position × context accuracy heatmaps for each quantization;
3. matched quantization gap versus context with sample counts;
4. effective context separately for every quantization/task type.

The interaction label is descriptive: `approximately_constant`,
`context_dependent`, or `insufficient_data`. It is not a statistical
significance claim. Relative degradation is measured from each variant's
shortest-context baseline at the same task and evidence position.

## Findings handoff

Only a real llama.cpp run with a resolved source manifest, raw trial evidence,
processed summary, coverage audit, and generated figures can produce a dated
entry in `docs/findings.md`. Until then, `exp_003` remains explicitly
`not yet measured`; no fixture value or hypothesis is presented as a finding.
