# exp_002 GGUF quantization results

Expected outputs include:

```text
results/
├── manifest.json              # resolved, hashed artifact/control manifest
├── raw/
│   └── trials.jsonl            # ignored by Git; append-only trial evidence
├── processed/
│   ├── summary.csv             # generated notebook input
│   ├── rescored-summary.csv    # issue #28 diagnostic comparison
│   └── rescoring-report.md     # issue #28 provenance and caveat
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

## Diagnostic re-scoring (issue #28)

The historical raw file is intentionally Git-ignored. When the local raw
artifact is available, re-score it without rerunning inference:

```bash
PYTHONPATH=src python3 experiments/exp_002-quantization_llama_cpp_gguf/rescore.py \
  --raw experiments/exp_002-quantization_llama_cpp_gguf/results/raw/trials.jsonl \
  --catalog data/tasks/core.v001.jsonl
```

This command requires the matching raw input and fails closed if it is missing
or empty. It writes:

- `processed/rescored-summary.csv` — old `expected.v1` versus new
  `calibrated.v1` results by task, quantization variant, and context length;
- `processed/rescoring-report.md` — the raw trial count, SHA-256, failure
  categories, and diagnostic-only interpretation boundary.

The current 120-trial input is recorded by SHA-256 as
`24ec3aca8687d369a58bc33843a8d2beaf37643c781b32fee2d198c035cf0188`.
Diagnostic re-scoring does not replace the historical `summary.csv` and must
not be used for a final quantization claim without a formal re-measurement
under the calibrated policy.
