# TDD Plan: Measure quantization × context-length interaction (#9)

**Type**: Feature / Experiment
**Issue**: https://github.com/morshoto/qwen-meetup-tokyo/issues/9
**Complexity**: High
**TDD Entry Point**: `Exp003ExperimentContractTests.test_config_declares_matched_context_quantization_matrix`

## Issue Summary

Issue #9 adds the central interaction experiment: run the same versioned tasks,
prompt, generated context instances, and scoring rules across selected GGUF
quantizations while varying context length and evidence position. The analysis
must distinguish a roughly constant quantization gap from a context-dependent
gap and report effective context independently for every quantization/task pair.

## Issue Excerpt

> Sweep context length × evidence position × quantization.

## Scope

**In scope**

- A committed `exp_003-context_x_quantization` protocol referencing the resolved
  `exp_002` artifact manifest and selecting its reproducible quantization IDs.
- A resumable runner that constructs each task/context/position instance once
  and executes every selected quantization against that same instance.
- Raw trial provenance for the matched context instance, quantization artifact,
  target/actual token counts, evidence offsets, and run fingerprint.
- Reusable matched-cell, short-context-relative degradation, interaction, and
  per-quantization/task effective-context analysis helpers.
- A notebook with context × quantization and position × context heatmaps,
  quantization-gap views, and measured-data-only validation.

**Out of scope**

- Committing 27B model artifacts or large raw result files.
- Treating fixture smoke output as Qwen evidence.
- Adding a finding with invented values before a real `exp_003` manifest and
  processed results exist.

## Behaviour Inventory

| ID | Behaviour | Test Level | First Test File | Notes |
| --- | --- | --- | --- | --- |
| B1 | The experiment config names the shared prompt/task catalog, selected `exp_002` variants, full context/position controls, and resource phases. | Contract | `tests/experiments/test_exp_003_contract.py` | The config is the reviewable protocol; no hidden matrix values. |
| B2 | The runner plans the declared context × position grid and records a stable matched context-instance identity. | Unit / integration | `tests/experiments/test_exp_003_runner.py` | The same task/seed/context/position is reused for every variant. |
| B3 | A fixture run executes the selected variant matrix, writes scored summaries, preserves artifact/variant provenance, and resumes without duplicate trial IDs. | Integration | `tests/experiments/test_exp_003_runner.py` | Fixture proves harness behavior only; real artifacts remain required for llama.cpp. |
| B4 | Matched-cell analysis rejects incomplete or duplicate dimensions and emits direct per-variant comparisons. | Unit | `tests/analysis/test_interaction.py` | Task type, context length, position, and quantization are required keys. |
| B5 | Relative degradation is computed against each variant's shortest-context same-position baseline. | Unit | `tests/analysis/test_interaction.py` | Baseline and measured cells retain sample counts. |
| B6 | The analysis labels the observed quantization gap as approximately constant, context-dependent, or insufficient data using a declared descriptive tolerance. | Unit | `tests/analysis/test_interaction.py` | This is an interpretation aid, not a significance test. |
| B7 | Effective context is computed independently for each quantization/task type. | Unit | `tests/analysis/test_interaction.py` | Existing baseline gates and right-censoring rules are reused. |
| B8 | The notebook validates measured inputs and contains the required interaction visualizations and findings handoff. | Contract / notebook | `tests/experiments/test_exp_003_contract.py` | Missing raw/processed/manifest data fails loudly. |

## Acceptance Criteria as Tests

| Acceptance Criterion | Test ID | Test Name |
| --- | --- | --- |
| Experiment is reproducible from a committed config. | B1 | `test_config_declares_matched_context_quantization_matrix` |
| Results support direct matched-cell comparisons across quantizations. | B2, B3, B4 | `test_runner_reuses_context_instance_across_variants` |
| Analysis reports whether quantization loss is constant or context-dependent. | B5, B6 | `test_interaction_report_identifies_context_dependent_gap` |
| Analysis includes context × quantization and position × quantization visualizations. | B8 | `test_notebook_contains_required_interaction_sections` |
| Effective context length is computed separately for each quantization/task type. | B7 | `test_effective_context_is_grouped_by_variant_and_task` |
| Findings are added without overstating statistical certainty. | B8 | `test_notebook_and_readme_separate_measured_findings_from_hypotheses` |

## Test-First Implementation Cycles

### Cycle 1: Freeze the experiment contract

**Red**

- Add `tests/experiments/test_exp_003_contract.py` for the experiment path,
  config controls, selected variants, and notebook/results contract.
- Expected failure: the `exp_003` directory and notebook do not exist.
- Run: `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_003_contract -v`

**Green**

- Add the committed `config.yaml`, `README.md`, `results/README.md`, and a
  measured-data-only notebook skeleton.
- Document that the source manifest is `exp_002`'s resolved manifest and that
  the full run still depends on locally provisioned artifacts and hardware.
- Run the focused contract test and `git diff --check`.

**Refactor**

- Keep all controls in the config and README aligned with the test assertions;
  remove any language that implies fixture measurements are model findings.
- Run the focused contract test again.

### Cycle 2: Implement matched context construction and execution

**Red**

- Add `tests/experiments/test_exp_003_runner.py` with an injected fake runtime
  and temporary resolved source manifest.
- Test the planned smoke grid, stable `context_instance_id`, identical context
  text/seed across variants, artifact provenance, summary output, and resume.
- Expected failure: the `exp_003` runner module and execution contract are
  missing.
- Run: `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_003_runner -v`

**Green**

- Add `experiments/exp_003-context_x_quantization/runner.py` using the shared
  `TaskCatalog`, `EvaluationTask`, `ExpectedAnswerScorer`, `SyntheticContextGenerator`,
  and `EvaluationRunner`.
- Load and validate `exp_002`'s `QuantizationManifest`, select variants by stable
  ID, construct contexts once per task/length/position, and attach the same
  instance ID to every quantization trial.
- Preserve append-only JSONL behavior, deterministic trial IDs, source-manifest
  SHA-256, coverage, exclusions, and raw-result SHA-256 in the run manifest.
- Run the focused runner tests.

**Refactor**

- Isolate selection, artifact verification, task construction, manifest building,
  and fixture/runtime setup without changing the matched-cell behavior.
- Run the runner tests plus the existing experiment tests.

### Cycle 3: Add matched interaction analysis

**Red**

- Add `tests/analysis/test_interaction.py` for complete matched-cell rows,
  per-variant short-context degradation, context-dependent gap reporting, and
  separate effective-context groups.
- Expected failure: interaction helpers are missing.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_interaction -v`

**Green**

- Add reusable helpers under `src/llm_lab/analysis/interaction.py` for strict
  dimension validation, matched-cell rows, relative degradation, descriptive
  interaction reporting, and per-variant/task effective-context calculation.
- Export the helpers from `llm_lab.analysis` and reuse the existing effective
  context gate, sustained crossing, provisional, and right-censored semantics.
- Run the focused analysis tests.

**Refactor**

- Keep aggregation/statistics in reusable Python and leave plotting concerns in
  the notebook; use explicit names for scored accuracy versus end-to-end success.
- Run all analysis tests.

### Cycle 4: Complete notebook analysis and findings handoff

**Red**

- Extend the notebook contract tests for required input checks, context ×
  quantization heatmaps, position × context heatmaps per variant, gap reporting,
  and per-variant/task effective-context output.
- Expected failure: the notebook skeleton lacks the required executable cells.
- Run: `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_003_contract -v`

**Green**

- Populate `analysis.ipynb` with cells that locate the repository, load the
  generated raw/processed results and run manifest, validate all expected
  dimensions, regenerate `summary.csv`, call the interaction helpers, and save
  only measured-data-derived figures/tables.
- Add a README findings workflow that links a real result manifest, processed
  table, and figures before any dated entry is added to `docs/findings.md`.
- Keep `docs/findings.md` at `exp_003: not yet measured` until a real run exists.
- Run notebook JSON/code-cell compilation and the focused contract tests.

**Refactor**

- Remove stale placeholder language, make missing-data errors actionable, and
  ensure every interpretation is labeled descriptive and condition-limited.
- Run the full suite, notebook validation, and `git diff --check`.

## Affected Files

| File | Action | TDD Role | Description |
| --- | --- | --- | --- |
| `experiments/exp_003-context_x_quantization/config.yaml` | Create | Contract | Reproducible controls and phase matrix. |
| `experiments/exp_003-context_x_quantization/README.md` | Create | Contract / review | Method, provenance, resource rules, and findings handoff. |
| `experiments/exp_003-context_x_quantization/results/README.md` | Create | Contract | Raw/processed/manifest/figure roles. |
| `experiments/exp_003-context_x_quantization/runner.py` | Create | Green | Matched context construction and resumable quantization execution. |
| `experiments/exp_003-context_x_quantization/analysis.ipynb` | Create / modify | Green | Measured interaction analysis and plots. |
| `src/llm_lab/analysis/interaction.py` | Create | Green | Reusable matched-cell and interaction calculations. |
| `src/llm_lab/analysis/__init__.py` | Modify | Green | Public analysis exports. |
| `tests/experiments/test_exp_003_contract.py` | Create | Red | Protocol and notebook contract. |
| `tests/experiments/test_exp_003_runner.py` | Create | Red | Runner and matched provenance regression tests. |
| `tests/analysis/test_interaction.py` | Create | Red | Interaction/effective-context analysis tests. |
| `docs/findings.md` | Preserve | Review boundary | No measured claim until real exp003 artifacts exist. |

## Test Commands

**Focused cycle commands**

```bash
PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_003_contract -v
PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_003_runner -v
PYTHONPATH=src python3 -m unittest tests.analysis.test_interaction -v
```

**Final verification**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests experiments/exp_003-context_x_quantization
python3 -m json.tool experiments/exp_003-context_x_quantization/analysis.ipynb >/dev/null
git diff --check
```

The real pilot/main run requires the resolved `exp_002` GGUF artifacts and
appropriate hardware. Fixture smoke is harness evidence only.

## Risks and Mitigations

- **Risk**: Quantization rows could use different generated contexts and make
  the comparison confounded.
  **Mitigation**: construct contexts before variant execution, record a stable
  `context_instance_id`, and test equality across variants.
- **Risk**: Missing or failed cells could disappear from interaction plots.
  **Mitigation**: validate the complete declared dimension product and preserve
  runtime failures in raw results and coverage manifests.
- **Risk**: Short-context baseline differences could be mistaken for context
  interactions.
  **Mitigation**: report same-position short-context-relative degradation and
  the matched high/low precision gap separately.
- **Risk**: Fixture output could be reported as a model finding.
  **Mitigation**: fail the notebook without measured inputs and keep
  `docs/findings.md` explicitly unmeasured until a real manifest is available.

## Definition of Done

- [ ] Every behavior in the inventory has a passing test.
- [ ] Every execution change is justified by a prior failing test.
- [ ] Variant comparisons reuse identical generated task/context instances.
- [ ] Analysis computes interaction gaps and per-variant/task effective context.
- [ ] Notebook contains both required heatmap families and fails on incomplete data.
- [ ] No unmeasured or statistically overstated finding is added to `docs/findings.md`.
- [ ] Focused and final verification commands pass.
