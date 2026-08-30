# TDD Plan: Re-score existing exp_002 raw outputs and classify evaluation failures (#28)

**Type**: Task
**Issue**: https://github.com/morshoto/qwen-meetup-tokyo/issues/28
**Complexity**: Medium
**TDD Entry Point**: `RescoringTests.test_completed_trials_are_classified_by_calibrated_outcome`

---

## Issue Summary

Issue #28 asks for a diagnostic re-score of the existing 120 `exp_002` raw
trials. The old `expected.v1` scores must remain intact while the calibrated
policy adds exactness, answer-bearing, format, and failure classification
dimensions. The output must be traceable to the raw JSONL hash and must not be
presented as a new formal quantization measurement.

## Issue Excerpt

> Re-score the existing `exp_002` raw outputs with the calibrated scoring policy and classify evaluation failures.

## Scope

**In scope**

- Re-score completed raw trial outputs with `calibrated.v1` using the matching
  versioned task catalog.
- Preserve the legacy score and report calibrated exact, answer-bearing, and
  format dimensions for every task/variant/context cell.
- Classify each trial as `exact_match`, `mismatch`, `format_failure`, or
  `runtime_failure`, with runtime status taking precedence over output scoring.
- Record the raw JSONL SHA-256 and emit a diagnostic comparison table/report.

**Out of scope**

- Re-running model inference or changing the ignored raw JSONL.
- Rewriting historical `expected.v1` trial records or the existing formal
  quantization summary.
- Using diagnostic re-scoring as a new quantization recommendation or final
  quantization claim.

## Behaviour Inventory

| ID | Behaviour | Test Level | First Test File | Notes |
| --- | --- | --- | --- | --- |
| B1 | A completed trial is re-scored with `calibrated.v1` while its legacy score remains available. | Unit | `tests/analysis/test_rescoring.py` | Exact, answer-bearing, and format fields are retained. |
| B2 | A completed output is classified as an exact match, mismatch, or format failure. | Unit | `tests/analysis/test_rescoring.py` | Format failure is checked before mismatch. |
| B3 | A non-completed trial is classified as a runtime failure without scoring an absent output. | Unit | `tests/analysis/test_rescoring.py` | OOM, timeout, and runtime-error statuses remain failures. |
| B4 | Comparison rows aggregate old/new scores and failure categories by task, quantization variant, and context. | Unit | `tests/analysis/test_rescoring.py` | Rows preserve attempted and scored denominators. |
| B5 | The report records the raw JSONL SHA-256 and labels the result diagnostic-only. | Integration | `tests/analysis/test_rescoring.py` | Missing raw input fails closed; no formal claim is emitted. |

## Acceptance Criteria as Tests

| Acceptance Criterion | Test ID | Test Name |
| --- | --- | --- |
| The raw JSONL SHA-256 is recorded. | B5 | `test_report_records_raw_sha256_and_diagnostic_caveat` |
| A comparison table exists for old and new scoring. | B1, B4 | `test_comparison_rows_include_legacy_and_calibrated_metrics` |
| Results are reported by task and quantization variant. | B4 | `test_comparison_rows_group_by_task_variant_and_context` |
| Re-scored data is not used for a final quantization claim without the required caveat. | B5 | `test_report_records_raw_sha256_and_diagnostic_caveat` |

## Test-First Implementation Cycles

### Cycle 1: Re-score one trial and classify its outcome

**Red**

- Add `tests/analysis/test_rescoring.py`.
- Test name: `test_completed_trials_are_classified_by_calibrated_outcome`.
- Use a semantic output that contains the accepted answer but is not exact, an
  identifier output with trailing generated text, and an exact multi-hop output.
- Expected failure before implementation: `ModuleNotFoundError` for the new
  rescoring analysis module.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_rescoring.RescoringTests.test_completed_trials_are_classified_by_calibrated_outcome -v`

**Green**

- Add `src/llm_lab/analysis/rescoring.py` with a small trial-level result type,
  calibrated re-scoring, and deterministic category precedence.
- Reuse `CalibratedAnswerScorer`; do not mutate or replace `TrialResult.score`.
- Run the focused test command above.

**Refactor**

- Extract category constants and output-field helpers only while the focused
  test remains green.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_rescoring -v`

### Cycle 2: Aggregate the old/new comparison table

**Red**

- Extend `tests/analysis/test_rescoring.py` with
  `test_comparison_rows_group_by_task_variant_and_context` and
  `test_runtime_failure_is_not_counted_as_a_calibrated_output`.
- Expected failure before implementation: comparison aggregation API is absent
  or does not expose task/variant/context rows and denominators.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_rescoring -v`

**Green**

- Add aggregation that reports legacy correctness, calibrated exactness,
  answer-bearing correctness, format validity, and each failure category.
- Derive the variant from `variant_condition_id` and retain the execution
  context/condition metadata; reject unknown task IDs rather than guessing.
- Export the public analysis functions from `llm_lab.analysis`.
- Run the focused rescoring test file.

**Refactor**

- Keep CSV field ordering deterministic through the existing summary writer or
  an equivalent sorted-field writer.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_rescoring tests.analysis.test_aggregation -v`

### Cycle 3: Produce a provenance-bearing diagnostic report

**Red**

- Add report-generation assertions to
  `test_report_records_raw_sha256_and_diagnostic_caveat`.
- Expected failure before implementation: no report writer/CLI exists and the
  input hash/caveat are absent.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_rescoring.RescoringTests.test_report_records_raw_sha256_and_diagnostic_caveat -v`

**Green**

- Add `experiments/exp_002-quantization_llama_cpp_gguf/rescore.py` as a CLI that
  accepts raw JSONL, task catalog, and output paths, hashes the input, writes a
  comparison CSV, and writes a Markdown report.
- Generate the report under `results/processed/` without copying the ignored
  raw model outputs into Git.
- State that the result is diagnostic re-scoring only and cannot support a new
  final quantization claim.
- Run the focused report test and the CLI against the available 120-trial raw
  file.

**Refactor**

- Add only the minimum README guidance needed to reproduce the diagnostic
  command and distinguish it from the formal measurement runner.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_rescoring tests.experiments.test_exp_002_runner -v`

## Affected Files

| File | Action | TDD Role | Description |
| --- | --- | --- | --- |
| `tests/analysis/test_rescoring.py` | Create | Red | Trial classification, aggregation, provenance, and caveat contracts. |
| `src/llm_lab/analysis/rescoring.py` | Create | Green | Pure calibrated re-scoring and comparison aggregation. |
| `src/llm_lab/analysis/__init__.py` | Modify | Green | Public exports for the rescoring seam. |
| `experiments/exp_002-quantization_llama_cpp_gguf/rescore.py` | Create | Green | Reproducible diagnostic report CLI. |
| `experiments/exp_002-quantization_llama_cpp_gguf/results/processed/rescored-summary.csv` | Create | Evidence | Per-task, per-variant, per-context comparison table. |
| `experiments/exp_002-quantization_llama_cpp_gguf/results/processed/rescoring-report.md` | Create | Evidence | Raw hash, scorer provenance, classifications, and caveat. |
| `experiments/exp_002-quantization_llama_cpp_gguf/results/README.md` | Modify | Documentation | Diagnostic re-scoring command and artifact roles. |

## Test Commands

**Focused cycle commands**

```bash
PYTHONPATH=src python3 -m unittest tests.analysis.test_rescoring -v
PYTHONPATH=src python3 -m unittest tests.analysis.test_rescoring tests.analysis.test_aggregation tests.experiments.test_exp_002_runner -v
```

**Final verification**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
git diff --check
```

## Refactor Checkpoints

- After Cycle 1: ensure trial-level classification does not mutate raw records.
- After Cycle 2: ensure all denominators distinguish attempted, scored, and
  runtime-failure trials.
- After Cycle 3: ensure the committed report is derived from the recorded hash
  and carries the diagnostic-only caveat.

## Risks and Mitigations

- **Risk**: A missing or wrong task catalog could silently change expected
  answers.
  **Mitigation**: require every raw task ID to resolve in the explicitly passed
  versioned catalog and fail closed on unknown IDs.
- **Risk**: Runtime failures could be misreported as model mismatches.
  **Mitigation**: classify non-completed statuses before invoking the scorer and
  retain attempted denominators.
- **Risk**: Diagnostic re-scoring could be mistaken for a new measurement.
  **Mitigation**: preserve the old raw score, record the raw hash, and repeat
  the diagnostic-only caveat in the report and results README.
- **Risk**: The ignored raw file may be unavailable to another reviewer.
  **Mitigation**: make the required input path and SHA explicit; do not fabricate
  raw outputs or claim reproducibility without the matching input artifact.

## Rollout / Review Notes

- This change is analysis-only and does not invoke the model runtime.
- The historical `summary.v001.csv` and `docs/findings.md` status remain unchanged.
- Reviewers should check the raw hash, task catalog version, scorer version,
  category precedence, and diagnostic caveat before treating the table as
  evidence.

## Definition of Done

- [ ] Every behavior in the inventory has a passing test.
- [ ] The raw JSONL SHA-256 is recorded in the committed report.
- [ ] Old and calibrated scores are compared by task, variant, and context.
- [ ] Mismatch, format failure, and runtime failure categories are explicit.
- [ ] The report clearly excludes the output from final quantization claims.
- [ ] Focused and full verification commands pass.
- [ ] The branch is pushed and the PR is created with the validation evidence.
