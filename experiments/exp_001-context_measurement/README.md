# exp_001-context_measurement

## Goal

Measure the effective context behavior of `Qwen/Qwen3.8-27B` across increasing
context lengths and evidence positions before introducing quantization as an
additional variable.

## Questions and hypotheses

- RQ1 / H2: does useful context differ between literal retrieval, semantic
  retrieval, and multi-hop reasoning?
- RQ2 / H1: does evidence position affect accuracy, and does the position gap
  grow with context length?
- H4: can literal retrieval remain reliable while distributed multi-hop evidence
  fails at the same nominal context length?

## Controlled matrix

| Variable | Main values |
| --- | --- |
| Context length | 8K, 32K, 64K, 128K, 262K or highest practical |
| Evidence position | 5%, 25%, 50%, 75%, 95% |
| Task family | literal, semantic, multi-hop |
| Repeats | 20 per cell where resources permit |
| Sampling | temperature 0.0, max 32 generated tokens |

The 8K reference condition is used for the 80% baseline-validity gate. Effective
context uses the first sustained accuracy drop below `alpha × baseline`; no
crossing is reported as right-censored, not unlimited. Position comparisons use
the requested position for the curve and retain actual token offsets in every
trial.

## Run phases

The executable defaults are in `runner.py`; `config.yaml` is the reviewable
experiment contract.

```bash
# From the repository root; no model weights required.
PYTHONPATH=src python3 experiments/exp_001-context_measurement/runner.py \
  --phase smoke --backend fixture

# Pilot/main runs use the optional local Transformers backend.
python3 -m pip install -e '.[transformers,analysis]'
PYTHONPATH=src python3 experiments/exp_001-context_measurement/runner.py \
  --phase pilot --backend transformers
PYTHONPATH=src python3 experiments/exp_001-context_measurement/runner.py \
  --phase main --backend transformers
```

The fixture backend returns catalog answers by design. It validates task
construction, evidence offsets, scoring, append-only storage, and coverage; it
is not Qwen evidence and must not be copied into `docs/findings.md` as a model
finding. The committed smoke manifest records all 18 harness cells as valid.
The main matrix remains resource-dependent; runtime/OOM/timeout cells are kept
in raw JSONL and listed as exclusions with reasons in the phase manifest.

## Provenance and outputs

Each raw trial records the task/catalog provenance, fixture/task seeds, target and
actual context counts, requested and actual evidence positions, evidence
start/end offsets, model/runtime identity, score, status, timing, memory, and
environment. The manifest records planned/actual counts, excluded cells, raw
SHA-256, and the repository SHA.

The committed smoke artifacts are:

- `results/raw/smoke-trials.jsonl` — 18 deterministic fixture trials;
- `results/manifests/smoke.json` — coverage and provenance manifest;
- `results/processed/summary.csv` — notebook-ready aggregation.

### Preliminary status — harness only

The recorded fixture smoke phase completed and scored 18 of 18 planned cells
across both smoke context lengths, three positions, and three task families;
the manifest lists zero exclusions. This establishes that the matrix,
provenance, scoring, and processing path are reproducible. It does not measure
Qwen behavior, so no accuracy or effective-context conclusion is drawn from it.

## Analysis

Run `analysis.ipynb` from this directory after a phase has produced raw results.
It regenerates `results/processed/summary.csv`, audits coverage, and produces:

1. evidence-position curves by task and context length;
2. context degradation curves by task family;
3. baseline-gated effective-context output;
4. systems-cost tables for prompt time, decode throughput, and peak memory when
   the runtime supplies those measurements.

No Qwen model finding is claimed until a pilot/main manifest contains measured
Qwen trials and the resulting processed tables are reviewed.
