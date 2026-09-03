# exp_001 results

## Layout

```text
results/
├── raw/        # append-only TrialResult JSONL; large runs remain ignored
├── processed/  # notebook-ready CSV/tables derived from raw JSONL
├── figures/    # regenerated measured-data figures
└── manifests/  # planned coverage, exclusions, hashes, and environment
```

The regeneration entry point writes `processed/summary.csv`,
`processed/position-gap.csv`, `processed/effective-context.json`, and
`processed/effective-context-by-position.json`; the notebook saves the
corresponding position-gap and effective-context figures under `figures/`.
For the bounded Q8 feasibility phase it also writes
`processed/feasibility-summary.csv` with one classification per tested length.
Run `analyze.py --manifest results/manifests/main.json` from this experiment
directory after a real-model phase has completed. The command verifies the raw
JSONL SHA-256 before writing any processed output.

Every planned task × context-length × evidence-position cell must be either
represented by scored trials or listed in its phase manifest as an exclusion
with the observed status and a resource/runtime reason. Runtime failures stay
in the raw denominator and are never silently dropped.

Feasibility classifications are `accepted_and_useful`,
`accepted_but_not_useful`, or `operational_failure`. They are bounded to the
pinned Q8 artifact, explicit task IDs, prompt construction, hardware, and
fixed timeout recorded in `manifests/feasibility.json`; they are not a general
effective-context claim.

The committed smoke run is a deterministic harness check, not a model result
and not a Qwen measurement:

- raw: `raw/smoke-trials.jsonl`;
- manifest: `manifests/smoke.json`;
- processed summary: `processed/summary.csv`;
- backend: `fixture`, 18 cells, 10 independent tasks per family, 180 trials,
  zero exclusions.

The raw directory is ignored for ordinary large runs. The small smoke JSONL is
force-tracked so the processed summary and manifest can be reproduced directly
in review. The manifest's SHA-256 must match its raw JSONL before analysis.
