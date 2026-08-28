# TDD Plan: Measure agent reliability as trajectory context grows (#10)

**Type**: Feature / Experiment  
**Issue**: https://github.com/morshoto/qwen-meetup-tokyo/issues/10  
**Complexity**: High  
**TDD Entry Point**: `AgentTrajectoryTests.test_trajectory_round_trips_ordered_tool_events`

## Issue Summary

Issue #10 adds `exp_004`, a deterministic local agent harness that varies the
length and relative position of a critical observation in accumulated tool-use
history. It must preserve matched task/tool observations across selected
quantizations, log complete machine-readable trajectories and provenance, and
provide measured-data-only analysis that separates retrieval/state-tracking
failures from tool/planning failures where the record supports that distinction.

## Issue Excerpt

> Evaluate whether a local Qwen agent becomes less reliable as its own multi-turn tool-use history accumulates.

## Scope

**In scope**

- A small reusable trajectory/message/tool contract and deterministic tool environment.
- A committed `exp_004-agent_context_growth` protocol with trajectory-length,
  critical-position, selected-variant, repeat, and runtime controls.
- A resumable runner that executes the same task and deterministic tool
  observations for every selected quantization and records effective settings,
  trajectory events, metrics, and artifact provenance.
- Analysis rows and plots for success/reliability by trajectory length and
  critical-information position, with descriptive failure taxonomy.
- A notebook that refuses to interpret missing or fixture-only inputs as Qwen findings.

**Out of scope**

- Committing local model artifacts or large raw result files.
- Rerunning the full `exp_002` quantization matrix.
- Treating fixture/smoke output as a measured Qwen result.
- Claiming causal or statistically significant effects without sufficient repeated measurements.

## Behaviour Inventory

| ID | Behaviour | Test Level | First Test File | Notes |
| --- | --- | --- | --- | --- |
| B1 | Trajectory events preserve ordered user, assistant, and tool messages and round-trip as JSON. | Unit | `tests/agents/test_trajectory.py` | Tool outputs and source metadata are evidence, not hidden state. |
| B2 | The deterministic environment returns stable tool outputs and rejects unknown or malformed calls. | Unit | `tests/agents/test_trajectory.py` | Fixture behavior is deterministic and explicitly labeled. |
| B3 | The runner plans the declared trajectory-length × critical-position matrix and reuses the same task/control identity across variants. | Integration | `tests/experiments/test_exp_004_runner.py` | Position controls map to pre/post distractor events. |
| B4 | Fixture runs log complete trajectories, outcome metrics, variant provenance, manifests, and append-only resume state. | Integration | `tests/experiments/test_exp_004_runner.py` | Fixture evidence cannot be mistaken for model measurements. |
| B5 | Analysis aggregates reliability by trajectory length and critical position and classifies observable failures. | Unit | `tests/analysis/test_agent_reliability.py` | Runtime failures remain in attempted denominators. |
| B6 | Analysis emits measured-data-only plots and rejects incomplete or fixture-only input. | Contract / notebook | `tests/experiments/test_exp_004_contract.py` | The notebook is an analysis surface, not a runner. |

## Acceptance Criteria as Tests

| Acceptance Criterion | Test ID | Test Name |
| --- | --- | --- |
| Agent trajectories and tool outputs are reproducibly machine-readable. | B1, B2, B4 | `test_fixture_run_records_ordered_trajectory_events` |
| The committed config is the source of truth for trajectory, position, and quantization controls. | B3, B6 | `test_config_declares_agent_growth_matrix_and_provenance_controls` |
| Matched comparisons reuse deterministic task/tool observations across variants. | B3, B4 | `test_runner_reuses_controlled_observations_across_variants` |
| Run manifests record effective runtime and model/artifact provenance. | B4 | `test_manifest_records_source_and_effective_runtime_settings` |
| Reliability is plotted against trajectory length and critical-information position. | B5, B6 | `test_analysis_rows_preserve_length_and_position_dimensions` |
| Retrieval/state-tracking and tool/planning failures are separated where observable. | B5 | `test_failure_taxonomy_distinguishes_retrieval_and_planning_failures` |
| Fixture/smoke results are not reported as Qwen findings. | B4, B6 | `test_notebook_requires_measured_non_fixture_inputs` |

## Test-First Implementation Cycles

### Cycle 1: Add the trajectory and deterministic-tool contract

**Red**

- Add `tests/agents/test_trajectory.py` for ordered event serialization,
  deterministic tool outputs, invalid calls, and stable control identities.
- Expected failure: the reusable agent contract does not exist.
- Run: `PYTHONPATH=src python3 -m unittest tests.agents.test_trajectory -v`

**Green**

- Add `src/llm_lab/agents/trajectory.py` with typed events, tool definitions,
  a deterministic environment, and JSON-safe records.
- Keep the environment small: one critical-fact tool and deterministic
  distractor observations are enough for the controlled task.
- Run the focused agent tests.

**Refactor**

- Keep serialization independent from any runtime backend and make validation
  errors identify the invalid action or event.
- Run the focused agent tests plus `git diff --check`.

### Cycle 2: Implement the matched exp_004 runner

**Red**

- Add `tests/experiments/test_exp_004_runner.py` with an injected deterministic
  fake runtime and temporary source manifest.
- Test the declared matrix, shared control/observation hashes, trajectory logs,
  runtime/artifact provenance, summary output, and resume without duplicate IDs.
- Expected failure: the experiment directory and runner do not exist.
- Run: `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_004_runner -v`

**Green**

- Add the committed config, task fixture, runner, and results contract.
- Construct a controlled task trajectory once per task/condition, execute the
  selected quantizations against the same deterministic environment inputs, and
  retain failures and effective settings in a run manifest.
- Require a resolved source manifest for real artifact runs; fixture smoke is
  explicit and labeled as harness validation.
- Run the focused runner tests.

**Refactor**

- Isolate config loading, matrix selection, source-manifest validation,
  runtime setup, trajectory execution, and manifest writing without changing
  matched behavior.
- Run the runner tests and the existing experiment/evaluation suites.

### Cycle 3: Add reliability and failure-taxonomy analysis

**Red**

- Add `tests/analysis/test_agent_reliability.py` for weighted reliability rows,
  position/length dimensions, failure categories, and missing-data rejection.
- Expected failure: agent analysis helpers are missing.
- Run: `PYTHONPATH=src python3 -m unittest tests.analysis.test_agent_reliability -v`

**Green**

- Add reusable analysis helpers under `src/llm_lab/analysis/agent_reliability.py`.
- Preserve attempted/scored denominators, expose critical-fact reuse,
  tool-call validity, repeated actions, recoveries, total input tokens, and
  descriptive categories (`retrieval`, `state_tracking`, `tool_planning`,
  `runtime`, `success`) when evidence supports them.
- Add measured-row plotting helpers for length and position views.
- Run focused analysis tests.

**Refactor**

- Keep aggregation and taxonomy in reusable Python; leave notebook I/O and
  figure styling in the experiment analysis surface.
- Run all analysis tests.

### Cycle 4: Complete the notebook and review handoff

**Red**

- Add `tests/experiments/test_exp_004_contract.py` for config controls,
  results layout, notebook sections, and fixture/measured separation.
- Expected failure: the experiment contract and measured-data-only notebook do
  not exist.
- Run: `PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_004_contract -v`

**Green**

- Add `analysis.ipynb` that loads a resolved run manifest and processed summary,
  validates the declared dimensions and non-fixture backend, calls the reusable
  analysis helpers, and writes only measured-data-derived figures.
- Document the real-run prerequisite of selected `exp_003` artifacts and keep
  `docs/findings.md` explicitly unmeasured until those artifacts exist.
- Run the focused contract tests, notebook JSON validation, and compilation.

**Refactor**

- Make missing-input errors actionable and ensure all conclusions are labeled
  descriptive and condition-limited.
- Run the full test suite, notebook validation, and `git diff --check`.

## Affected Files

| File | Action | TDD Role | Description |
| --- | --- | --- | --- |
| `src/llm_lab/agents/trajectory.py` | Create | Green | Serializable trajectory and deterministic tool primitives. |
| `src/llm_lab/agents/__init__.py` | Modify | Green | Public agent exports. |
| `src/llm_lab/analysis/agent_reliability.py` | Create | Green | Reliability aggregation, taxonomy, and plotting. |
| `src/llm_lab/analysis/__init__.py` | Modify | Green | Public analysis exports. |
| `experiments/exp_004-agent_context_growth/config.yaml` | Create | Contract | Declarative protocol and phase controls. |
| `experiments/exp_004-agent_context_growth/runner.py` | Create | Green | Matched, resumable agent execution. |
| `experiments/exp_004-agent_context_growth/README.md` | Create | Contract / review | Provenance, runtime, smoke, and findings rules. |
| `experiments/exp_004-agent_context_growth/analysis.ipynb` | Create | Contract / Green | Measured-data-only analysis. |
| `experiments/exp_004-agent_context_growth/results/README.md` | Create | Contract | Artifact roles and exclusions. |
| `data/tasks/agent.v001.jsonl` | Create | Fixture | Versioned deterministic agent task family. |
| `tests/agents/test_trajectory.py` | Create | Red | Agent contract tests. |
| `tests/experiments/test_exp_004_runner.py` | Create | Red | Runner and matched provenance tests. |
| `tests/analysis/test_agent_reliability.py` | Create | Red | Analysis and failure taxonomy tests. |
| `tests/experiments/test_exp_004_contract.py` | Create | Red | Config, notebook, and results contract tests. |

## Test Commands

**Focused cycle commands**

```bash
PYTHONPATH=src python3 -m unittest tests.agents.test_trajectory -v
PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_004_runner -v
PYTHONPATH=src python3 -m unittest tests.analysis.test_agent_reliability -v
PYTHONPATH=src python3 -m unittest tests.experiments.test_exp_004_contract -v
```

**Final verification**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests experiments/exp_004-agent_context_growth
python3 -m json.tool experiments/exp_004-agent_context_growth/analysis.ipynb >/dev/null
git diff --check
```

The real pilot/main run requires a resolved selected-variant manifest from
`exp_003`, local model artifacts, and suitable hardware. Fixture smoke is
harness evidence only.

## Risks and Mitigations

- **Risk**: Different variants could receive different observations.
  **Mitigation**: hash the task/control/environment inputs, reuse them across
  variants, and assert equality in the runner tests.
- **Risk**: Tool/planning errors could be mislabeled as retrieval failures.
  **Mitigation**: retain every generated action and tool result, then classify
  only observable failure paths with an explicit `insufficient_evidence` fallback.
- **Risk**: Runtime failures could disappear from reliability curves.
  **Mitigation**: retain failed trial records and include attempted denominators.
- **Risk**: Fixture output could be reported as a model finding.
  **Mitigation**: record backend purpose, reject fixture-only notebook inputs, and
  keep `docs/findings.md` unmeasured until a real run manifest exists.

## Definition of Done

- [ ] Every behavior in the inventory has a passing test.
- [ ] Each execution change is justified by a prior failing test.
- [ ] Matched variants reuse deterministic task/tool observations.
- [ ] Raw trajectories, metrics, effective runtime settings, and provenance are logged.
- [ ] Analysis covers trajectory length, critical position, and observable failure taxonomy.
- [ ] The notebook rejects missing or fixture-only measurement inputs.
- [ ] No unmeasured finding is added to `docs/findings.md`.
- [ ] Focused and final verification commands pass.
