# TDD Plan: Run the real-model exp_001 context baseline (#23)

**Type**: Feature / Experiment
**Issue**: https://github.com/morshoto/qwen-meetup-tokyo/issues/23
**Complexity**: High
**TDD Entry Point**: `ContextAnalysisTests.test_position_gap_uses_edge_minus_middle_accuracy`

## Issue Summary

Issue #23 turns the existing `exp_001` fixture harness into a reproducible
real-model baseline. The run must measure the reference Qwen model at 8K, 32K,
64K, and 128K tokens at all five evidence positions, retain raw failures and
provenance, and regenerate the processed tables and figures without inventing
missing cells or treating fixture output as model evidence.

## Issue Excerpt

> Run the real-model `exp_001` context baseline.

## Scope

**In scope**

- A frozen real-model matrix with five evidence positions and multiple
  independent tasks per task-family cell.
- Raw JSONL, phase manifest, processed summary, position-gap table/figure, and
  task/evidence-position effective-context table/figure regeneration.
- Explicit 80% baseline-validity classification and fail-visible resource
  exclusions.
- A measured-data-only notebook and reviewable run instructions.

**Out of scope**

- Quantization comparisons or changes to `exp_002`/`exp_003`.
- Fabricating model results when Qwen weights or suitable hardware are absent.
- Adding measured findings to `docs/findings.md` before the real manifest and
  processed artifacts have been reviewed.

## Behaviour Inventory

| ID | Behaviour | Test Level | First Test File | Notes |
| --- | --- | --- | --- | --- |
| B1 | Position-gap analysis computes `Δ_pos(C) = A_edge(C) - A_middle(C)` while preserving context, task, and scored counts. | Unit | `tests/analysis/test_effective_context.py` | Edge means the 5% and 95% cells; middle is 50%. |
| B2 | Analysis regeneration verifies raw/manifest provenance, aggregates the declared scorer, and writes summary, position-gap, and effective-context outputs. | Integration | `tests/experiments/test_exp_001_analysis.py` | Runtime failures remain in attempted denominators. |
| B3 | The runner follows the committed phase matrix and records every planned cell, including resource/runtime exclusions with reasons. | Integration | `tests/experiments/test_exp_001_runner.py` | Main requires 8K/32K/64K/128K and all five positions. |
| B4 | Real-model analysis rejects fixture-only input unless explicitly requested for harness validation. | Contract / notebook | `tests/experiments/test_exp_001_contract.py` | Fixture output never becomes a Qwen finding. |
| B5 | A valid 8K baseline below 80% is classified `baseline-limited`; valid baselines retain the existing effective-context/right-censoring rule. | Unit | `tests/analysis/test_effective_context.py` | No relative breakpoint is claimed for baseline-limited families. |

## Acceptance Criteria as Tests

| Acceptance Criterion | Test ID | Test Name |
| --- | --- | --- |
| Real-model raw results, manifest, and processed summary are recorded. | B2, B3 | `test_regeneration_writes_provenance_checked_outputs` |
| 8K, 32K, 64K, and 128K conditions are measured. | B3 | `test_main_matrix_contains_required_context_lengths_and_positions` |
| All five evidence positions and multiple independent tasks are represented. | B3 | `test_manifest_counts_independent_tasks_per_cell` |
| Families below the 80% baseline gate are marked baseline-limited. | B5 | `test_effective_context_marks_sub_gate_baselines` |
| Position-gap and effective-context figures can be regenerated. | B1, B2, B4 | `test_notebook_exposes_regeneration_outputs` |

## Test-First Implementation Cycles

### Cycle 1: Freeze the analysis contract

**Red**

- Add position-gap tests and a contract test for the required main matrix and
  regeneration output names.
- Expected failure: the position-gap helper and real-run contract are absent.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_effective_context tests.experiments.test_exp_001_contract -v`

**Green**

- Add `position_gap_rows` to the reusable analysis package.
- Add the explicit four-length main phase and output paths to `config.yaml`.
- Keep the existing smoke artifact and its fixture-only interpretation intact.
- Run the focused tests.

**Refactor**

- Reuse the existing context-cell validation and weighted scored counts; do not
  duplicate effective-context rules in the experiment script.
- Run the focused tests and `git diff --check`.

### Cycle 2: Add measured-data regeneration

**Red**

- Add `tests/experiments/test_exp_001_analysis.py` with temporary model-like
  raw records and a manifest whose SHA-256 is checked.
- Assert summary CSV, position-gap CSV, effective-context JSON/CSV, and
  exclusion/denominator metadata are regenerated from the raw input.
- Run: `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_001_analysis -v`

**Green**

- Add `experiments/exp_001-context_measurement/analyze.py` as a dependency-light
  regeneration entry point.
- Require a non-fixture backend by default, verify the raw hash and scorer, and
  write only data derived from the raw JSONL and manifest.
- Run the focused analysis integration tests.

**Refactor**

- Keep CSV/JSON I/O in the experiment entry point and calculations in
  `src/llm_lab/analysis`; make errors identify the missing or mismatched input.
- Run analysis and aggregation tests.

### Cycle 3: Harden the model runner and provenance

**Red**

- Extend runner tests for config-driven dimensions, runtime failure status,
  exclusion reasons, actual model-mode provenance, and resumable raw output.
- Run: `PYTHONPATH=src python3 -m unittest tests.evaluation.test_runner tests.experiments.test_exp_001_runner -v`

**Green**

- Load the committed phase controls, use the required main grid, classify
  timeout/OOM failures, and include planned dimensions, scorer/effective-context
  controls, and exclusion reasons in the manifest.
- Preserve append-only trial IDs and the explicit fixture smoke overwrite gate.
- Run the focused runner/evaluation tests.

**Refactor**

- Keep runtime setup injectable and avoid making optional Transformers imports
  part of the base package import path.
- Run the affected experiment/evaluation suites.

### Cycle 4: Complete the notebook and delivery documentation

**Red**

- Add contract assertions for measured-only validation, saved position-gap and
  effective-context figures, and the required four-length/five-position matrix.
- Run: `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_001_contract -v`

**Green**

- Update `analysis.ipynb`, the experiment README, and results README with the
  real-run command, artifact locations, exclusions policy, and no-fabrication
  handoff.
- Attempt the real-model run only when the local Transformers stack and Qwen
  weights are available; preserve the exact blocker otherwise.
- Run the full test suite, notebook JSON validation, compilation, and diff
  checks.

**Refactor**

- Ensure figure/table paths, manifest hashes, and README commands agree, then
  rerun the complete verification set before opening the PR.

## Affected Files

| File | Action | TDD Role | Description |
| --- | --- | --- | --- |
| `src/llm_lab/analysis/effective_context.py` | Modify | Green | Position-gap rows and existing effective-context calculations. |
| `src/llm_lab/analysis/__init__.py` | Modify | Green | Public analysis export. |
| `experiments/exp_001-context_measurement/config.yaml` | Modify | Contract | Required model matrix, sampling, and output controls. |
| `experiments/exp_001-context_measurement/analyze.py` | Create | Green | Provenance-checked processed output regeneration. |
| `experiments/exp_001-context_measurement/runner.py` | Modify | Green | Config-driven run controls and exclusion provenance. |
| `experiments/exp_001-context_measurement/analysis.ipynb` | Modify | Contract / Green | Measured-data-only tables and saved figures. |
| `experiments/exp_001-context_measurement/README.md` | Modify | Review | Real-model procedure and interpretation boundary. |
| `experiments/exp_001-context_measurement/results/README.md` | Modify | Contract | Artifact roles and regeneration commands. |
| `tests/analysis/test_effective_context.py` | Modify | Red | Position-gap and baseline-gate tests. |
| `tests/experiments/test_exp_001_analysis.py` | Create | Red | Regeneration integration tests. |
| `tests/experiments/test_exp_001_contract.py` | Create | Red | Matrix/notebook/results contract tests. |
| `tests/experiments/test_exp_001_runner.py` | Modify | Red | Runner and manifest provenance tests. |
| `tests/evaluation/test_runner.py` | Modify | Red | OOM/timeout status preservation tests. |

## Test Commands

**Focused cycle commands**

```bash
PYTHONPATH=src python3 -m unittest tests.analysis.test_effective_context tests.experiments.test_exp_001_contract -v
PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_001_analysis -v
PYTHONPATH=src python3 -m unittest tests.evaluation.test_runner tests.experiments.test_exp_001_runner -v
```

**Final verification**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests experiments/exp_001-context_measurement
python3 -m json.tool experiments/exp_001-context_measurement/analysis.ipynb >/dev/null
git diff --check
```

The real main run additionally requires the optional Transformers/Torch
dependencies, locally available Qwen weights, and hardware that can execute
the declared context lengths. Missing those prerequisites is a blocker, not a
reason to substitute fixture output.

## Risks and Mitigations

- **Risk**: resource failures disappear from the baseline.
  **Mitigation**: retain one raw trial record per attempted task, classify
  timeout/OOM statuses, and include excluded cells plus reasons in the manifest.
- **Risk**: a weak 8K baseline creates a misleading relative breakpoint.
  **Mitigation**: retain absolute accuracy and mark the family
  `baseline-limited` below the 80% gate.
- **Risk**: fixture data is mistaken for model evidence.
  **Mitigation**: fail measured analysis closed on `backend: fixture` and keep
  smoke artifacts explicitly labeled as harness-only.
- **Risk**: notebook output drifts from raw data.
  **Mitigation**: regenerate summary/tables from raw JSONL after verifying the
  manifest hash and scorer version.

## Definition of Done

- [ ] Required matrix and five positions are declared in config and manifest.
- [ ] Every planned cell is valid or has a reasoned exclusion.
- [ ] Raw results, manifest, processed summary, position-gap, and
  effective-context outputs are reproducible from the run inputs.
- [ ] Baseline-limited families are explicitly classified.
- [ ] No fixture output is presented as a model finding.
- [ ] Focused and final verification commands pass.
- [ ] The branch is pushed and the PR points to issue #23.
