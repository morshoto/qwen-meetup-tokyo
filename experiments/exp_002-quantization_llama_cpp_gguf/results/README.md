# exp_002 GGUF quantization results

Expected outputs include:

```text
results/
├── manifest.json              # resolved, hashed v002 artifact/control manifest
├── manifest.v001.json         # preserved historical v001 manifest
├── raw/
│   ├── trials-v002.jsonl      # ignored by Git; append-only v002 evidence
│   └── trials.jsonl           # ignored historical v001 evidence
├── processed/
│   ├── pilot-v002-summary.csv  # measured 30-trial v002 pilot summary
│   ├── pilot-v002-report.md    # pilot hash, observations, and completion boundary
│   ├── summary.v001.csv        # preserved historical v001 summary
│   ├── pilot-v001-summary.csv  # preserved historical v001 pilot summary
│   ├── rescored-summary.csv    # issue #28 diagnostic comparison
│   └── rescoring-report.md     # issue #28 provenance and caveat
└── figures/                   # generated only after the complete v002 run
```

`manifest.json` must be derived from the experiment's template and contain
actual model/runtime revisions, artifact SHA-256 digests, and artifact sizes.
The template or a result with placeholder values is not a completed run.

The v002 pilot command selects `q8_0`, context length `8192`, and
`--repeats 1`, for 30 trials:

```bash
PYTHONPATH=src python3 experiments/exp_002-quantization_llama_cpp_gguf/runner.py \
  --manifest experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json \
  --output experiments/exp_002-quantization_llama_cpp_gguf/results/raw/trials-v002.jsonl \
  --processed experiments/exp_002-quantization_llama_cpp_gguf/results/processed/pilot-v002-summary.csv \
  --condition-id q8_0 --context-length 8192 --repeats 1
```

Running the same command again resumes by deterministic trial ID. Use the
same `--output` and `--processed` paths plus `--repeats 1` to extend the
capability pilot across the remaining variants and context length. A newly
resolved manifest with `capability_repeats: 1` completes the full 240-trial
capability matrix; the checked-in historical manifest still declares a
five-repeat envelope and therefore reports 1,200 legacy trial IDs. Duplicate or
mismatched trial records are rejected. The v001 manifest,
summary, and historical raw path remain preserved separately. The committed
`pilot-v002-summary.csv` is pilot evidence only; a full v002 `summary.csv` is
not valid until all variants, contexts, and tasks have been measured. The
capability matrix uses one greedy run per independent task. Additional timing
probes must use a separate output and denominator.
The committed notebook has no cached outputs, and stale figures are removed;
the two figures are regenerated from the selected calibrated phase. Pilot
figures are explicitly pilot-only; a full comparison requires the complete
v002 summary.
The committed pilot was regenerated under the source-revision resume guard;
extend its raw JSONL only when the manifest and recorded source revisions still
match.

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
Diagnostic re-scoring does not replace the historical `summary.v001.csv` and must
not be used for a final quantization claim without a formal re-measurement
under the calibrated policy.
