# exp_003-context_x_quantization

## Goal

Measure whether quantization degradation is approximately constant across
context length or grows at long context, and whether lower precision amplifies
evidence-position effects.

This experiment reuses the versioned `data/tasks/core.v002.jsonl` catalog with
10 independent tasks in each QA family, `prompt.qa.v001`,
`CalibratedAnswerScorer` (`calibrated.v1`), and the synthetic context generator
used by exp_001. It selects the `q8_0`, `q6_k`, `q5_k_m`, and `q4_k_m`
artifacts from the resolved exp_002 manifest. A context instance is generated
once for each task, context length, evidence position, and seed, then reused
for every quantization variant so matched-cell comparisons are valid.

The checked-in exp_003 outputs are historical three-task smoke measurements from
the previous catalog version. They remain unchanged until a real llama.cpp run
produces the issue #21 measurement for the expanded catalog.

For new runs, the resolved exp_002 manifest is the execution source of truth
for the catalog path and hash, selected task IDs, model artifacts, scorer
policy, and runtime provenance. Every selected task ID must exist in the
verified `core.v002.jsonl` catalog.

The checked-in historical smoke records retain their original `expected.v1`
scores. New runs require the manifest-declared `calibrated.v1` policy and do
not relabel or reuse those historical scores as calibrated measurements.

## Controlled matrix

| Variable | Main values |
| --- | --- |
| Context length | 8K, 32K, 64K, 128K |
| Evidence position | 5%, 25%, 50%, 75%, 95% |
| Quantization | Q8_0, Q6_K, Q5_K_M, Q4_K_M from exp_002 |
| Task family | literal, semantic, multi-hop |
| Capability repeats | 1 per independent task; timing repeats are a separate probe |
| Sampling | greedy, `temperature: 0.0`, max 64 generated tokens |

The committed config is the protocol. Artifact identity, model/tokenizer
revisions, runtime settings, and conversion provenance come from
`../exp_002-quantization_llama_cpp_gguf/results/manifest.full.json`; the output run
manifest records its SHA-256 so later source-manifest changes cannot silently
alter the interpretation of a run.

The output run manifest keeps inherited source runtime options separately from
the effective per-variant options used for execution, including derived
context capacity and model path settings. The resolved exp_002 manifest is the
source of truth for the v002 catalog, catalog hash, artifact identities, and
`calibrated.v1` scorer policy.

The runner loads phase lengths, evidence positions, the capability repeat
count, backend, and the default quantization variants from `config.yaml`.
When `--source-manifest` is omitted, it uses the pinned
`experiment.source_manifest` value from that same config.
The larger `repeats` value is retained as a timing/legacy envelope; matching
CLI flags are explicit overrides for a selected run.

## Run phases

From the repository root, the fixture smoke run needs no model weights:

```bash
PYTHONPATH=src python3 experiments/exp_003-context_x_quantization/runner.py \
  --source-manifest experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.full.json \
  --phase smoke --backend fixture
```

The pilot/main phases require the locally provisioned GGUF artifacts referenced
by the resolved exp_002 manifest and suitable hardware:

```bash
PYTHONPATH=src python3 experiments/exp_003-context_x_quantization/runner.py \
  --source-manifest experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.full.json \
  --phase pilot --backend llama.cpp
```

The fixture backend validates matching, context construction, scoring, storage,
and coverage. It is not a Qwen measurement and must not be copied into
`docs/findings.md` as a model finding. Runtime/OOM/timeout trials remain in raw
results and in the end-to-end denominator; a cell is excluded only when
planned attempts are missing.

The issue #21 main capability matrix is four quantization variants × four
context lengths × five evidence positions × 30 independent tasks × one run =
2,400 attempted trials before runtime exclusions. Q8_0 and Q4_K_M are the required
matched comparison; Q6_K and Q5_K_M remain in the declared matrix for the same
artifact-controlled interaction view. Repeating a deterministic greedy prompt
does not add an independent capability observation; collect any timing repeats
as a separate, explicitly labelled probe.

After a model run, regenerate the verified task-level tables before opening the
notebook:

```bash
PYTHONPATH=src python3 experiments/exp_003-context_x_quantization/analyze.py \
  --manifest experiments/exp_003-context_x_quantization/results/manifests/main.json
```

This writes `summary.csv`, `relative-degradation.csv`, `interaction.json`, and
`effective-context.json` only after source-manifest, catalog, raw-result,
scorer, coverage, and matched-context validation succeeds.

The primary degradation and interaction rates use end-to-end success over all
attempted trials. Runtime and invalid-output failures remain in that
denominator; `scored_n` and the exact/answer-bearing/format-valid metrics are
reported separately for diagnosis.

## Analysis

`analysis.ipynb` loads the generated raw results, processed summary, and output
run manifest. It fails loudly when measured inputs are missing or incomplete and
produces:

1. context × quantization end-to-end-success/degradation heatmaps by task type;
2. position × context end-to-end-success heatmaps for each quantization;
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
