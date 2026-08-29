# exp_002 GGUF quantization results

Expected outputs include:

```text
results/
├── manifest.json              # resolved, hashed artifact/control manifest
├── raw/
│   └── trials.jsonl            # ignored by Git; append-only trial evidence
├── processed/
│   └── summary.csv             # generated notebook input
└── figures/
    ├── accuracy-vs-memory.png
    └── speed-vs-memory.png
```

`manifest.json` must be derived from the experiment's template and contain
actual model/runtime revisions, artifact SHA-256 digests, and artifact sizes.
The template or a result with placeholder values is not a completed run.

The current v002 pilot command selects `q8_0`, context length `8192`, and
`--repeats 1`, for 30 trials. Running a new v002 output again without selectors
resumes the same JSONL file and completes the full 1,200-trial matrix;
duplicate trial IDs are rejected. The committed result files in this directory
are historical v001 measurements over three tasks and 120 trials; their
provenance is intentionally preserved.
