# TDD Plan: 🧪 exp_003 — Measure the full context × quantization interaction (#21)

**Type**: Feature / Experiment
**Issue**: https://github.com/morshoto/qwen-meetup-tokyo/issues/21
**Complexity**: High
**TDD Entry Point**: `Exp003ExperimentContractTests.test_main_declares_issue_21_matrix`

---

### Issue Summary

Issue #21 turns the existing exp_003 harness into the full matched real-model
measurement. Q8 and Q4 must use the same versioned tasks and generated context
instances at 8K, 32K, 64K, and 128K across all evidence positions, with
independent tasks and repeated trials. The processed result must calculate the
context × quantization interaction directly and classify it as
`approximately_constant`, `context_dependent`, or `insufficient_data`.

The current checkout already contains the earlier issue #9 harness and fixture
smoke data. The checked-in exp_002 manifest now identifies the v002 task catalog,
calibrated scorer, model artifacts, and llama.cpp runtime, but no exp_003 real
run exists. Fixture output remains harness evidence only.

### Issue Excerpt

> Run the full `exp_003` context × quantization interaction measurement.

### Scope

**In scope**

- A committed exp_003 main protocol with 8K, 32K, 64K, and 128K input-token
  contexts, all five evidence positions, Q8/Q4 coverage, independent catalog
  tasks, and multiple repeats.
- Loading the resolved exp_002 catalog and calibrated scorer policy as the
  execution source of truth, with catalog hash and scorer provenance on every
  trial and run manifest.
- Matched context construction once per task/context/position and reuse across
  every selected quantization variant and repeat.
- Task-level aggregation that preserves independent task identity, validates
  complete matched dimensions, computes short-context-relative degradation,
  and reports a direct weighted interaction gap by context.
- A measured-data-only regeneration script/notebook and durable processed
  interaction outputs when a real llama.cpp run completes.
- A real pilot or main run only when the local artifacts and hardware can
  complete it; raw failures remain in the denominator and no fixture result is
  promoted to `docs/findings.md`.

**Out of scope**

- Committing GGUF weights, local caches, or raw JSONL trial data.
- Reusing the historical exp_003 v001 smoke output as v002 model evidence.
- Claiming a statistical significance result from the descriptive classifier.
- Adding a findings entry before the resolved run manifest, raw hash, processed
  summary, coverage audit, and figures are reviewed.

### Behaviour Inventory

| ID | Behaviour | Test Level | First Test File | Notes |
| --- | --- | --- | --- | --- |
| B1 | The main protocol declares the four issue-required contexts, all evidence positions, Q8/Q4, independent tasks, and more than one repeat. | Contract | `tests/experiments/test_exp_003_contract.py` | 128K is required; 262K is not part of this issue's required main matrix. |
| B2 | The runner loads the resolved source catalog and calibrated scorer, records their identity, and plans only declared task IDs. | Integration | `tests/experiments/test_exp_003_runner.py` | A catalog hash mismatch fails closed before inference. |
| B3 | Every quantization receives the same generated task/context instance, including stable context ID, text hash, seed, and evidence offsets. | Integration | `tests/experiments/test_exp_003_runner.py` | Independent tasks must not collide during aggregation. |
| B4 | Repeated raw trials aggregate at task × variant × context × position level while retaining attempted, scored, and failure denominators. | Unit / integration | `tests/analysis/test_interaction.py` | Runtime and invalid-output trials remain attempted. |
| B5 | Matched analysis rejects missing, duplicate, unexpected, or mismatched context identities and calculates each variant's short-context-relative degradation. | Unit | `tests/analysis/test_interaction.py` | Matching is fail-closed. |
| B6 | The interaction report compares Q8/Q4 at each context and labels the result with the declared three-way classification. | Unit | `tests/analysis/test_interaction.py` | Weight by matched scored trials and call the result descriptive. |
| B7 | Analysis regenerates only from a verified real run manifest and writes interaction/effective-context outputs. | Integration / contract | `tests/experiments/test_exp_003_analysis.py` | Fixture input requires an explicit harness-only opt-in. |

### Acceptance Criteria as Tests

| Acceptance Criterion | Test ID | Test Name |
| --- | --- | --- |
| Q8 and Q4 are compared under matched conditions. | B2, B3, B5 | `test_runner_reuses_context_instance_across_variants` |
| 8K, 32K, 64K, and 128K plus all evidence positions are included. | B1 | `test_main_declares_issue_21_matrix` |
| Multiple repeats and independent tasks are recorded. | B1, B3, B4 | `test_task_level_aggregation_preserves_independent_tasks` |
| The context × quantization interaction is calculated directly. | B5, B6 | `test_interaction_report_identifies_context_dependent_gap` |
| The result is classified as approximately constant, context-dependent, or insufficient data. | B6 | `test_interaction_report_classifies_insufficient_data` |
| Measured outputs are provenance-checked and fixture-only runs are not findings. | B7 | `test_regeneration_rejects_fixture_without_explicit_opt_in` |

### Test-First Implementation Cycles

#### Cycle 1: Freeze the issue #21 protocol

**Red**

- Extend `tests/experiments/test_exp_003_contract.py` with
  `test_main_declares_issue_21_matrix`.
- Assert the exact main contexts and positions, at least Q8/Q4, a repeat count
  greater than one, and the v002 catalog/scorer boundary.
- Expected failure before implementation: the current main protocol still
  declares the legacy 262K condition and does not identify the calibrated
  v002 execution boundary.
- Run:
  `uv run --offline python -m unittest tests.experiments.test_exp_003_contract -v`

**Green**

- Update `experiments/exp_003-context_x_quantization/config.yaml` to the issue
  #21 main matrix and explicit calibrated/v002 controls.
- Update the experiment and results README files with the real-run matrix,
  expected trial-count formula, and the fixture-only evidence boundary.
- Run the focused contract test and `git diff --check`.

**Refactor**

- Keep protocol values in config and make the README refer to those values
  without duplicating contradictory legacy controls.
- Run the focused contract test again.

#### Cycle 2: Bind execution to v002 provenance and task identity

**Red**

- Add runner tests for manifest-declared catalog loading, catalog hash
  verification, calibrated scoring, task-level trial provenance, and source
  revision fingerprinting.
- Update the existing runner fixture assertions to expect selected manifest
  task IDs rather than silently using the hard-coded catalog population.
- Expected failure before implementation: the runner imports
  `ExpectedAnswerScorer`, always loads `core.v002` from a constant, and writes
  family-level summaries without catalog/scorer provenance.
- Run:
  `uv run --offline python -m unittest tests.experiments.test_exp_003_runner -v`

**Green**

- Make `runner.py` load the local catalog and hash from the resolved exp_002
  manifest, require `calibrated.v1`, and select the manifest task IDs.
- Use `CalibratedAnswerScorer`, aggregate with the expected scorer and
  `group_by_task=True`, and attach catalog/scorer/source revision metadata to
  each task request and run manifest.
- Preserve deterministic IDs, append-only resume, artifact verification, and
  effective per-variant runtime options.
- Run the focused runner tests.

**Refactor**

- Extract catalog/path/provenance helpers and keep fixture runtime injection
  independent from the model-backed path.
- Run the runner tests plus the existing evaluation, quantization, and
  experiment contract tests.

#### Cycle 3: Make independent-task matching and interaction analysis strict

**Red**

- Add task-aware interaction tests covering two independent tasks per family,
  duplicate/missing matched cells, shared context identity, baseline-relative
  degradation, context-dependent gaps, approximately constant gaps, and
  insufficient data.
- Expected failure before implementation: the current four-field key treats
  two independent tasks of the same family as duplicate cells and cannot safely
  use the task-level summary produced by Cycle 2.
- Run:
  `uv run --offline python -m unittest tests.analysis.test_interaction -v`

**Green**

- Extend `src/llm_lab/analysis/interaction.py` with an optional task-aware key
  while retaining compatibility with the existing family-level fixtures.
- Include task identity in baseline matching, validate all selected task IDs,
  and aggregate task-level rows into weighted per-variant/context interaction
  points with explicit scored counts.
- Keep the existing effective-context threshold and right-censoring semantics.
- Run the focused analysis tests.

**Refactor**

- Separate dimension validation, row enrichment, and descriptive reporting;
  keep plotting out of reusable analysis code.
- Run all analysis tests and `git diff --check`.

#### Cycle 4: Add measured-data regeneration and evidence handoff

**Red**

- Add `tests/experiments/test_exp_003_analysis.py` for raw hash validation,
  complete coverage, calibrated scorer validation, task-level summary output,
  interaction report output, and explicit fixture rejection.
- Extend the notebook contract for regeneration outputs and the three-way
  classification handoff.
- Expected failure before implementation: exp_003 has no analysis regeneration
  module, and its notebook aggregates legacy family-level smoke data.
- Run:
  `uv run --offline python -m unittest tests.experiments.test_exp_003_analysis tests.experiments.test_exp_003_contract -v`

**Green**

- Add `experiments/exp_003-context_x_quantization/analyze.py` to validate the
  run manifest, source manifest hash, raw-result hash, scorer, complete
  variant/context/position/task coverage, and output provenance before writing.
- Regenerate task-level `summary.csv`, relative-degradation CSV, interaction
  JSON, and per-variant/task effective-context JSON from verified measured data.
- Update `analysis.ipynb` to call the regeneration path and save only measured
  interaction heatmaps/gap plots. Require explicit fixture opt-in for harness
  checks and keep `docs/findings.md` unmeasured until real evidence exists.
- Run focused analysis/contract tests and notebook JSON validation.

**Refactor**

- Make missing-cell and fixture-only errors actionable, keep descriptive labels
  condition-limited, and avoid writing partial outputs after failed validation.
- Run the full test suite, compile checks, notebook validation, and a clean
  fixture smoke regeneration in a temporary output directory.

#### Cycle 5: Execute and verify the real measurement when provisioned

**Red**

- Add an artifact-level contract check that requires a real llama.cpp manifest,
  four required contexts, Q8/Q4 coverage, all five positions, independent task
  counts, and repeated attempted trials before a finding is eligible.
- Expected failure before implementation/evidence: the checkout has only the
  historical fixture smoke artifact and no exp_003 real-run manifest.
- Run the focused artifact contract before launching inference.

**Green**

- Run a Q8/Q4 pilot first using the resolved exp_002 artifacts and the same
  append-only raw path, then resume to the declared main matrix when runtime
  and hardware permit.
- Keep runtime errors, invalid output, OOM, and timeout records in raw JSONL;
  list incomplete cells and reasons in the run manifest.
- Commit only the resolved manifest, processed summaries/figures, and evidence
  documentation; keep raw JSONL and model artifacts ignored.
- Add a dated `docs/findings.md` entry only after the interaction report and
  coverage audit satisfy the issue acceptance criteria.

**Refactor**

- Recheck artifact/catalog/source hashes, trial-count arithmetic, matched
  context IDs, denominators, classification labels, and the final git diff.
- Run the full suite, `compileall`, notebook validation, and a deterministic
  resume check against the produced raw evidence.

### Affected Files

| File | Action | TDD Role | Description |
| --- | --- | --- | --- |
| `docs/issue-21-exp003-full-measurement-plan.md` | Create | Plan | Issue context and executable TDD cycles. |
| `experiments/exp_003-context_x_quantization/config.yaml` | Modify | Green | Issue #21 main matrix and provenance controls. |
| `experiments/exp_003-context_x_quantization/README.md` | Modify | Green | Real-run matrix and evidence boundary. |
| `experiments/exp_003-context_x_quantization/results/README.md` | Modify | Green | Output and raw-data provenance contract. |
| `experiments/exp_003-context_x_quantization/runner.py` | Modify | Green | Manifest-driven calibrated matched execution. |
| `experiments/exp_003-context_x_quantization/analyze.py` | Create | Green | Verified processed-output regeneration. |
| `experiments/exp_003-context_x_quantization/analysis.ipynb` | Modify | Green | Measured interaction views. |
| `src/llm_lab/analysis/interaction.py` | Modify | Green | Task-aware matched analysis. |
| `src/llm_lab/analysis/__init__.py` | Modify | Green | Public analysis exports if needed. |
| `tests/experiments/test_exp_003_contract.py` | Modify | Red | Protocol and notebook contract. |
| `tests/experiments/test_exp_003_runner.py` | Modify | Red | Provenance, scoring, matching, and resume. |
| `tests/analysis/test_interaction.py` | Modify | Red | Independent-task analysis behavior. |
| `tests/experiments/test_exp_003_analysis.py` | Create | Red | Verified regeneration and evidence boundary. |
| `docs/findings.md` | Preserve / modify only with real evidence | Review boundary | No fixture finding. |

### Test Commands

**Focused cycle commands**

```bash
uv run --offline python -m unittest tests.experiments.test_exp_003_contract -v
uv run --offline python -m unittest tests.experiments.test_exp_003_runner -v
uv run --offline python -m unittest tests.analysis.test_interaction -v
uv run --offline python -m unittest tests.experiments.test_exp_003_analysis -v
```

**Final verification**

```bash
uv run --offline python -m unittest discover -s tests -v
uv run --offline python -m compileall -q src tests experiments/exp_003-context_x_quantization
uv run --offline python -m json.tool experiments/exp_003-context_x_quantization/analysis.ipynb >/dev/null
git diff --check
```

The real pilot/main run requires the resolved local GGUF artifacts and suitable
hardware. Fixture smoke is harness validation only.

### Risks and Mitigations

- **Risk**: Different variants receive different generated contexts.
  **Mitigation**: build one task/context set per task condition, persist a
  stable context ID and hash, and fail analysis on identity disagreement.
- **Risk**: Independent tasks are collapsed into one family row or counted as
  duplicate matched cells.
  **Mitigation**: aggregate by task ID, validate task-level dimensions, and
  weight interaction points by matched scored trial counts.
- **Risk**: A stale source catalog or scorer changes the result silently.
  **Mitigation**: verify the resolved manifest catalog SHA-256 and scorer before
  inference and record both in every trial and processed manifest.
- **Risk**: Long-context runtime failures hide an incomplete interaction.
  **Mitigation**: retain all attempted failures, mark excluded cells, and
  classify incomplete reports as `insufficient_data`.
- **Risk**: Fixture output is mistaken for Qwen behavior.
  **Mitigation**: reject fixture analysis by default and keep `docs/findings.md`
  explicitly unmeasured until real llama.cpp evidence is reviewed.

### Rollout / Review Notes

- Do not commit weights or raw JSONL; reviewers need the resolved source
  manifest and raw SHA-256 to reproduce the run locally.
- Reviewers should check Q8/Q4 matched context IDs, all four required lengths,
  all five positions, task-family counts, repeat counts, and failure
  denominators before interpreting any classification.
- The classifier is descriptive. It must not be presented as a hypothesis test
  or generalized beyond the tested Qwen revision, llama.cpp build, hardware, and
  task catalog.

### Definition of Done

- [ ] The issue #21 main matrix is declared and tested.
- [ ] Q8 and Q4 use identical task/context instances and recorded provenance.
- [ ] Multiple repeats and independent tasks remain visible in raw and processed data.
- [ ] Direct context × quantization gaps are calculated and classified.
- [ ] Incomplete or fixture-only data cannot produce a model finding.
- [ ] Real measured outputs, if provisioned, have verified source/raw/artifact provenance.
- [ ] Every production change is justified by a prior failing test.
- [ ] Focused and final verification commands pass.

