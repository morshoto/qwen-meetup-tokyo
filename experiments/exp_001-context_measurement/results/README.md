# exp_001 results

## Layout

```text
results/
├── raw/        # append-only TrialResult JSONL; large runs remain ignored
├── processed/  # notebook-ready CSV/tables derived from raw JSONL
└── manifests/  # planned coverage, exclusions, hashes, and environment
```

Every planned task × context-length × evidence-position cell must be either
represented by scored trials or listed in its phase manifest as an exclusion
with the observed status and a resource/runtime reason. Runtime failures stay
in the raw denominator and are never silently dropped.

The committed smoke run is a deterministic harness check, not a model result:

- raw: `raw/smoke-trials.jsonl`;
- manifest: `manifests/smoke.json`;
- processed summary: `processed/summary.csv`;
- backend: `fixture`, 18 cells, 10 independent tasks per family, 180 trials,
  zero exclusions.

The raw directory is ignored for ordinary large runs. The small smoke JSONL is
force-tracked so the processed summary and manifest can be reproduced directly
in review. The manifest's SHA-256 must match its raw JSONL before analysis.
