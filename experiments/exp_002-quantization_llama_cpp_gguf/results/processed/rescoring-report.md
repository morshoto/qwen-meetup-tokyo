# exp_002 diagnostic re-scoring report

**Status: Diagnostic re-scoring only.** This report reinterprets existing
generated outputs; it is not a new model run or formal quantization
measurement.

## Provenance

- Raw input: `results/raw/trials.jsonl`
- Raw trial count: 120
- Raw SHA-256: `24ec3aca8687d369a58bc33843a8d2beaf37643c781b32fee2d198c035cf0188`
- Task catalog: `data/tasks/core.v001.jsonl`
- Legacy scorer: `expected.v1` (preserved in the raw records)
- Calibrated scorer: `calibrated.v1`
- Processing entry point: `rescore.py`

## Old/new comparison by quantization variant

The detailed comparison table is `rescored-summary.csv`, with one row
per task, quantization variant, and context length.

| Variant | Old correct/scored | New exact correct/scored | New answer-bearing correct/scored | New format-valid/scored | Mismatch | Format failure | Runtime failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 10/30 | 10/30 | 30/30 | 20/30 | 10 | 10 | 0 |
| Q5_K_M | 10/30 | 10/30 | 30/30 | 20/30 | 10 | 10 | 0 |
| Q6_K | 10/30 | 10/30 | 30/30 | 20/30 | 10 | 10 | 0 |
| Q8_0 | 10/30 | 10/30 | 30/30 | 20/30 | 10 | 10 | 0 |

## Failure classification

- `mismatch`: completed output is not an exact calibrated answer but
  remains format-valid.
- `format_failure`: completed output is empty or violates the expected
  answer shape; this category takes precedence over mismatch.
- `runtime_failure`: the original trial did not complete; no output is
  rescored and it remains in the attempted denominator.

## Interpretation boundary

The historical raw JSONL and its `expected.v1` scores are preserved.
This diagnostic table must not be used for a final quantization claim
without the required caveat and a formal re-measurement under the
calibrated policy. The existing formal `summary.csv` is unchanged.
