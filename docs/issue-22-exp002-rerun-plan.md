## TDD Plan: 🧪 exp_002 — Re-run quantization trade-offs with calibrated tasks and scoring (#22)

**Type**: Feature / Experiment
**Issue**: https://github.com/morshoto/qwen-meetup-tokyo/issues/22
**Complexity**: High
**TDD Entry Point**: `Exp002RunnerTests.test_runner_uses_manifest_task_catalog_and_records_provenance`

---

### Issue Summary

Issue #22 asks for a new `exp_002` measurement using the calibrated scorer and
the expanded 30-task `core.v002` catalog. The current checkout already has the
runtime and scoring pieces, but the checked-in manifest and processed result
files are the historical 3-task, 120-trial `expected.v1` run. The new run must
be independently traceable and must not silently reuse that historical output.

### Issue Excerpt

> Re-run `exp_002` with the calibrated scorer and expanded task catalog.

### Current Delivery Status

The implementation is complete and a real-model v002 pilot is committed. The
pilot covers 30 Q8_0 trials at 8,192 input tokens; the remaining 1,170 trials
of the declared 1,200-trial matrix are intentionally unrun, so the full
quantization comparison and recommendation remain pending.

### Scope

**In scope**

- Make the resolved run manifest the source of truth for the task catalog and
  calibrated scorer, including a catalog SHA-256.
- Preserve per-task trial records and produce task-level processed summaries
  for all Q8_0, Q6_K, Q5_K_M, and Q4_K_M conditions at both context lengths.
- Keep artifact revision, conversion command, SHA-256, and byte size attached to
  every declared variant and trial metadata.
- Keep stream TTFT and throughput fields explicitly separate from unavailable
  native prefill/decode counters.
- Report end-to-end success and failure rate using all attempted trials, while
  retaining runtime and invalid-output failures as distinct outcomes.
- Run the v002 pilot/full matrix only from a resolved manifest and record the
  resulting summary and analysis provenance.

**Out of scope**

- Committing GGUF weights, model caches, or raw JSONL trial output.
- Rewriting the historical v001 manifest/summary as if it were v002 evidence.
- Calling stream-derived timing proxies native llama.cpp kernel metrics.
- Choosing Q4_K_M solely because it is the smallest artifact.

### Behaviour Inventory

| ID | Behaviour | Test Level | First Test File | Notes |
| --- | --- | --- | --- | --- |
| B1 | A resolved manifest declares the versioned task catalog, its SHA-256, and `calibrated.v1`; the runner loads that catalog rather than a hard-coded source. | Contract / integration | `tests/experiments/test_exp_002_runner.py` | A mismatched catalog digest fails closed before inference. |
| B2 | Every generated v002 trial records task, variant/artifact, catalog, scorer, context, and sampling provenance. | Integration | `tests/experiments/test_exp_002_runner.py` | Trial IDs remain deterministic and resumable. |
| B3 | Processed summaries retain one row per task, variant, context, and repeat aggregate. | Unit / integration | `tests/analysis/test_aggregation.py` | Legacy family-level aggregation remains the default for other experiments. |
| B4 | Calibrated scores distinguish exact correctness, answer-bearing correctness, format validity, and runtime failures. | Unit | `tests/evaluation/test_scoring_calibration.py` / `tests/evaluation/test_runner.py` | Existing scorer contracts already cover the dimensions; v002 must use them. |
| B5 | Analysis separates capability outcomes from systems-cost measurements and labels stream-derived timing as proxy data. | Contract / notebook | `tests/analysis/test_quantization.py` | Required metrics remain end-to-end success, failure rate, artifact size, memory, TTFT, and stream proxies. |
| B6 | A completed v002 result set is reproducible from the resolved manifest and contains every declared variant/task/context cell. | Integration / artifact | `tests/experiments/test_exp_002_runner.py` | Missing artifacts, cells, or scorer/catalog identity fail loudly. |

### Acceptance Criteria as Tests

| Acceptance Criterion | Test ID | Test Name |
| --- | --- | --- |
| Every artifact has revision, conversion command, SHA-256, and size provenance. | B1, B2 | `test_resolver_records_real_artifact_identity_and_revisions` and `test_trial_records_artifact_and_catalog_provenance` |
| Task-level results are available for every quantization variant. | B2, B3, B6 | `test_task_level_summary_keeps_task_ids` and `test_full_selection_has_1200_expected_trials` |
| Stream-derived proxies are clearly separated from native metrics. | B4, B5 | `test_streamed_generation_records_measurement_fields` and `test_notebook_separates_capability_and_systems_metrics` |
| Runtime failures remain in the denominator. | B4, B5 | `test_aggregation_reports_calibrated_metrics_and_failure_kinds` and `test_tradeoff_rows_reports_failures_and_recommendation_uses_end_to_end_success` |
| Capability and systems-cost figures are separate. | B5 | `test_notebook_separates_capability_and_systems_metrics` |

### Test-First Implementation Cycles

#### Cycle 1: Bind the run to the catalog and scorer policy

**Red**

- Extend `tests/experiments/test_exp_002_runner.py` with
  `test_runner_uses_manifest_task_catalog_and_records_provenance` and a
  manifest/catalog fixture whose source is not the runner's default constant.
- Extend `tests/experiments/test_exp_002_resolver.py` with
  `test_resolver_records_task_catalog_hash_and_scorer_policy`.
- Expected failure before implementation: the manifest model has no catalog
  provenance, and the runner either cannot load the declared source or omits
  the catalog/scorer metadata from trial input.
- Run:
  `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_002_runner tests.experiments.test_exp_002_resolver -v`

**Green**

- Add optional task-catalog path/hash and scorer-policy fields to
  `QuantizationManifest` with backward-compatible parsing for historical
  manifests.
- Make `resolve_manifest.py` hash the explicitly selected catalog and write the
  resolved identity atomically.
- Make `runner.py` load and verify the manifest-declared catalog, require
  `calibrated.v1`, and attach its identity plus variant artifact provenance to
  every request.
- Do not overwrite the historical v001 result files in this cycle.
- Run the focused tests above.

**Refactor**

- Centralize path/hash validation and keep the old default only as a migration
  path for historical fixtures.
- Run:
  `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_002_runner tests.experiments.test_exp_002_resolver tests.quantization.test_specs -v`

#### Cycle 2: Preserve task-level results and denominator semantics

**Red**

- Add `test_task_level_summary_keeps_task_ids` to
  `tests/analysis/test_aggregation.py` and extend the exp002 runner fixture to
  assert that the v002 output has one summary row per task/cell.
- Expected failure before implementation: aggregation collapses distinct task
  IDs into only a task-family row.
- Run:
  `PYTHONPATH=src python3 -m unittest tests.analysis.test_aggregation tests.experiments.test_exp_002_runner -v`

**Green**

- Add an opt-in task-level grouping mode to `aggregate_trials`/
  `aggregate_jsonl` and use it for the exp002 processed summary.
- Include variant label/type, artifact size, and catalog identity in the
  summary where those values are common across the group.
- Keep attempted, completed, scored, invalid-output, runtime-failure, and
  end-to-end-success denominators explicit.
- Run the focused aggregation and exp002 tests.

**Refactor**

- Preserve the existing family-level default used by other experiments and
  keep CSV field ordering deterministic.
- Run:
  `PYTHONPATH=src python3 -m unittest tests.analysis.test_aggregation tests.analysis.test_quantization tests.experiments.test_exp_002_runner -v`

#### Cycle 3: Make the analysis boundary explicit

**Red**

- Extend `tests/analysis/test_quantization.py` with
  `test_notebook_separates_capability_and_systems_metrics`.
- Expected failure before implementation: the notebook does not expose named
  capability/systems tables or a clear stream-proxy boundary.
- Run:
  `PYTHONPATH=src python3 -m unittest tests.analysis.test_quantization -v`

**Green**

- Update the notebook and experiment README to expose capability fields
  (`scored_accuracy`, `end_to_end_success`, `failure_rate`) separately from
  systems fields (artifact size, peak memory, stream TTFT, and throughput
  proxies), with native prefill/decode counters explicitly unavailable for
  this binding.
- Keep the recommendation based on measured end-to-end success within a
  declared tolerance, then artifact size/memory/speed tie-breakers.
- Run the focused notebook-contract tests.

**Refactor**

- Keep all data validation in reusable Python helpers and plotting/reporting in
  the notebook.
- Run:
  `PYTHONPATH=src python3 -m unittest tests.analysis.test_quantization tests.experiments.test_exp_002_runner -v`

#### Cycle 4: Resolve, measure, and verify v002 evidence

**Red**

- Add contract assertions for the resolved v002 manifest, pilot count
  (30 trials), full matrix count (1,200 trials), and complete task/variant/
  context coverage.
- Expected failure before implementation/evidence: the checked-in manifest and
  processed summary still identify v001/120 historical trials.
- Run the focused contract tests before any model run.

**Green**

- Resolve all four local GGUF artifacts against the v002 template with the
  exact model/tokenizer/runtime/converter revisions and catalog hash.
- Run the v002 pilot first, then resume the same append-only JSONL to complete
  the full 4 × 2 × 30 × 5 matrix when runtime/hardware permits. The committed
  evidence currently contains the pilot only.
- Generate task-level `summary.csv` after the full matrix, run the measured-only
  notebook, and retain runtime/invalid-output rows rather than filtering them.
- Commit only the resolved manifest, processed summaries/figures, README, and
  provenance documentation; keep raw trials and weights ignored.

**Refactor**

- Verify manifest/artifact/catalog hashes, matrix coverage, scorer version,
  denominator arithmetic, and the proxy/native metric boundary from the actual
  output files.
- Run the full test suite, `compileall`, `git diff --check`, and a clean
  re-run/resume check against the produced v002 raw output.

### Affected Files

| File | Action | TDD Role | Description |
| --- | --- | --- | --- |
| `docs/issue-22-exp002-rerun-plan.md` | Create | Plan | Issue context, behavior inventory, and executable TDD cycles. |
| `src/llm_lab/quantization/specs.py` | Modify | Green | Resolved task-catalog/scorer provenance in manifest controls. |
| `experiments/exp_002-quantization_llama_cpp_gguf/manifest.template.json` | Modify | Green | v002 catalog hash and scorer placeholders. |
| `experiments/exp_002-quantization_llama_cpp_gguf/resolve_manifest.py` | Modify | Green | Catalog hashing and resolved identity. |
| `experiments/exp_002-quantization_llama_cpp_gguf/runner.py` | Modify | Green | Manifest-driven catalog/scorer and trial provenance. |
| `src/llm_lab/analysis/aggregation.py` | Modify | Green | Opt-in task-level summaries and provenance columns. |
| `experiments/exp_002-quantization_llama_cpp_gguf/analysis.ipynb` | Modify | Green | Separate capability and systems-cost analysis. |
| `experiments/exp_002-quantization_llama_cpp_gguf/README.md` | Modify | Refactor | v002 run, metric, and evidence instructions. |
| `tests/experiments/test_exp_002_runner.py` | Modify | Red | Manifest-driven v002 matrix and trial provenance. |
| `tests/experiments/test_exp_002_resolver.py` | Modify | Red | Catalog hash resolution. |
| `tests/analysis/test_aggregation.py` | Modify | Red | Task-level grouping and denominator preservation. |
| `tests/analysis/test_quantization.py` | Modify | Red | Analysis boundary contract. |
| `experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json` | Replace after measurement | Evidence | Resolved v002 artifact/control manifest. |
| `experiments/exp_002-quantization_llama_cpp_gguf/results/processed/pilot-v002-summary.csv` | Create | Evidence | Measured 30-trial v002 pilot summaries. |
| `experiments/exp_002-quantization_llama_cpp_gguf/results/processed/summary.csv` | Pending full matrix | Evidence | Task-level measured v002 summaries after all 1,200 trials. |

### Test Commands

**Focused cycle commands**

```bash
PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_002_runner tests.experiments.test_exp_002_resolver -v
PYTHONPATH=src python3 -m unittest tests.analysis.test_aggregation tests.analysis.test_quantization -v
```

**Final verification**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests experiments/exp_002-quantization_llama_cpp_gguf
git diff --check
```

### Risks and Mitigations

- **Risk**: A stale or wrong task catalog changes expected answers silently.
  **Mitigation**: resolve and verify the catalog SHA-256 before inference and
  record it in every v002 trial.
- **Risk**: The 1,200-trial 27B matrix is long-running and may encounter OOM or
  timeout failures.
  **Mitigation**: run the 30-trial pilot first, resume by deterministic trial
  ID, and keep every failure in the attempted denominator.
- **Risk**: Historical v001 output is accidentally mixed with v002 output.
  **Mitigation**: use a new manifest fingerprint/scorer/catalog identity and
  fail closed on existing records outside the selected run.
- **Risk**: Stream timing proxies are mistaken for native backend metrics.
  **Mitigation**: preserve `timing_source`/`timing_semantics`, leave native
  prefill/decode unavailable, and separate systems tables in the notebook.
- **Risk**: Q4 is selected from size alone.
  **Mitigation**: recommendation uses end-to-end success tolerance first and
  retains capability and systems-cost figures side by side.

### Rollout / Review Notes

- No weights or raw JSONL are committed; reviewers need the local artifact
  manifest and raw SHA-256 to reproduce the measured output.
- The old v001 result is historical evidence and must remain identifiable as
  such after v002 files are published.
- Reviewers should check all 30 task IDs across four variants, two context
  lengths, and the historical five-repeat envelope, then inspect
  runtime/invalid-output counts before
  interpreting a quantization recommendation.

### Definition of Done

- [ ] Every behavior in the inventory has a passing test.
- [ ] Each production change is justified by a prior failing test.
- [ ] The resolved manifest records artifact and task-catalog provenance plus
  `calibrated.v1`.
- [ ] v002 task-level results cover every declared variant/context/task cell.
- [ ] Runtime failures remain in attempted denominators and are classified.
- [ ] Stream-derived proxy metrics are separated from native metrics.
- [ ] Capability and systems-cost figures are separate in analysis.
- [ ] Focused and final verification commands pass.
- [ ] Measured evidence is committed without weights/raw trials or fabricated
  results.
