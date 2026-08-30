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
| Context length | 8K, 32K, 64K, 128K |
| Evidence position | 5%, 25%, 50%, 75%, 95% |
| Task family | literal, semantic, multi-hop; 10 independent tasks per family |
| Repeats | 20 per cell where resources permit |
| Sampling | temperature 0.0, max 32 generated tokens |

The 8K reference condition is used for the 80% baseline-validity gate. Effective
context uses the first sustained accuracy drop below `alpha × baseline`; no
crossing is reported as right-censored, not unlimited. Position comparisons use
the requested position for the curve and retain actual token offsets in every
trial. A task family below the 80% gate is marked `baseline-limited` and is not
assigned a relative effective-context breakpoint.

## Run phases

The executable defaults are in `runner.py`; `config.yaml` is the reviewable
experiment contract.

```bash
# From the repository root; no model weights required.
PYTHONPATH=src python3 experiments/exp_001-context_measurement/runner.py \
  --phase smoke --backend fixture --overwrite-smoke

# Pilot/main runs use the optional local Transformers backend and local Qwen weights.
python3 -m pip install -e '.[transformers,analysis]'
PYTHONPATH=src python3 experiments/exp_001-context_measurement/runner.py \
  --phase pilot --backend transformers
PYTHONPATH=src python3 experiments/exp_001-context_measurement/runner.py \
  --phase main --backend transformers

# If a model run is interrupted, continue from its append-only JSONL and the
# sampling checkpoint in the existing manifest.
PYTHONPATH=src python3 experiments/exp_001-context_measurement/runner.py \
  --phase main --backend transformers --resume

# Regenerate tables from a completed, verified model manifest.
PYTHONPATH=src python3 experiments/exp_001-context_measurement/analyze.py \
  --manifest experiments/exp_001-context_measurement/results/manifests/main.json
```

The real-model command is intentionally not a download command. The required
Transformers/Torch stack, Qwen weights, and hardware must already be available;
otherwise the run is blocked and fixture output must not be substituted.

The fixture backend returns catalog answers by design. It validates task
construction, evidence offsets, scoring, append-only storage, and coverage; it
is not Qwen evidence and must not be copied into `docs/findings.md` as a model
finding. The committed smoke manifest records all 18 harness cells and 180
independent-task trials as valid.
The explicit `--overwrite-smoke` flag makes this deterministic fixture command
safe to rerun against the committed artifact paths; other existing output paths
fail closed so append-only trial data cannot be accidentally duplicated.
The main matrix remains resource-dependent; runtime/OOM/timeout cells are kept
in raw JSONL and listed as exclusions with reasons in the phase manifest. The
runner's `--resume` option uses deterministic trial IDs to fill only missing
attempts. A model run writes its resolved sampling checkpoint before the first
trial; resume requires that checkpoint and rejects changes to its effective
settings or to sampling provenance already stored in raw trials. It also
requires an in-progress checkpoint matching the experiment phase, backend,
raw-results path, and resolved model/tokenizer identity; completed or cross-run
manifests are rejected. Context-generation provenance (source revision, fixture
seed, config and catalog hashes, task IDs, and the exact phase matrix) is also
checkpointed and
must match before existing trials are reused.

## Provenance and outputs

Each raw trial records the task/catalog provenance, fixture/task seeds, target and
actual context counts, requested and actual evidence positions, evidence
start/end offsets, model/runtime identity, score, status, timing, memory, and
environment. Model-backed target context counts use the loaded inference
tokenizer; final prompt token counts after chat formatting are recorded by the
runtime. Fixture smoke is explicitly `whitespace-v1` harness data. The manifest
records planned/actual counts, declared dimensions, independent task counts,
excluded cells and reasons, raw SHA-256, effective-context controls, sampling
settings, and the repository SHA.

The committed smoke artifacts are:

- `results/raw/smoke-trials.jsonl` — 180 deterministic fixture trials;
- `results/manifests/smoke.json` — coverage and provenance manifest;
- `results/processed/summary.csv` — notebook-ready aggregation;
- `results/processed/position-gap.csv` — `A_edge - A_middle` by task/context;
- `results/processed/effective-context.json` and
  `results/processed/effective-context-by-position.json` — baseline-gated
  breakpoint tables;
- `results/figures/position-gap-vs-context.png` and
  `results/figures/effective-context-vs-context.png` — regenerated figures.

### Preliminary status — harness only

The recorded fixture smoke phase completed and scored 180 of 180 planned trials
across both smoke context lengths, three positions, ten independent tasks per
family, and three task families; the manifest lists zero exclusions. This
establishes that the matrix, provenance, scoring, and processing path are
reproducible. It does not measure Qwen behavior, so no accuracy or
effective-context conclusion is drawn from it.

## Analysis

Run `analysis.ipynb` from this directory after a measured phase has produced
raw results. It calls `analyze.py`, verifies the manifest hash and scorer,
audits coverage, and produces:

1. evidence-position curves by task and context length;
2. context degradation curves by task family;
3. `A_edge - A_middle` position-gap curves;
4. aggregate and evidence-position-conditioned effective-context output;
5. systems-cost tables for prompt time, decode throughput, and peak memory when
   the runtime supplies those measurements.

Set `EXP001_PHASE=smoke EXP001_ALLOW_FIXTURE=1` only to validate the harness
path. The default is `EXP001_PHASE=main` and fixture-only input is rejected.
No Qwen model finding is claimed until a pilot/main manifest contains measured
Qwen trials and the resulting processed tables and figures are reviewed.
