## TDD Plan: Select quantization approach and measure precision trade-offs (#8)

**Type**: Feature / Experiment
**Issue**: https://github.com/morshoto/qwen-meetup-tokyo/issues/8
**Complexity**: High
**TDD Entry Point**: `QuantizationManifestTests.test_manifest_preserves_variant_provenance_and_controls`

### Issue Summary

Issue #8 turns the `exp_002` placeholder into a reproducible GGUF quantization
comparison. The selected implementation is `llama.cpp` through the optional
`llama-cpp-python` binding, with Q8_0, Q6_K, Q5_K_M, and Q4_K_M artifacts plus an
F16 reference when available.

### Issue Excerpt

> Select the exact quantization implementation/runtime and rename the experiment
> directory away from `approach_a`.

### Scope

**In scope**

- A typed quantization manifest that records artifact source, revisions,
  conversion procedure, hashes, sizes, format, and runtime kernel.
- An injectable `llama.cpp` runtime adapter that preserves the shared generation
  contract and exposes explicitly named stream-derived TTFT and throughput
  proxy measurements.
- A renamed `exp_002-quantization_llama_cpp_gguf` protocol with fixed prompts,
  tasks, short/medium context lengths, greedy sampling, and Q8/Q6/Q5/Q4
  conditions.
- Notebook-ready end-to-end-success-vs-memory and explicit stream-proxy
  separate artifact-size, sampled-RSS, TTFT, and throughput-proxy figures plus
  a data-driven recommendation rule.

**Out of scope**

- Committing 27B model weights or generated raw trial data.
- Comparing unrelated quantization families, alternate runtimes, or tuning
  kernels in the first sweep.
- Treating placeholder hashes or missing artifacts as measured evidence.

### Behaviour Inventory

| ID | Behaviour | Test Level | First Test File | Notes |
| --- | --- | --- | --- | --- |
| B1 | A quantization manifest retains stable condition IDs and complete artifact provenance. | Unit | `tests/quantization/test_specs.py` | Duplicate conditions and incomplete hashes are rejected. |
| B2 | The GGUF runtime loads a declared model path and forwards fixed sampling/runtime options. | Unit | `tests/runtimes/test_llama_cpp.py` | Uses an injected fake client; no model download. |
| B3 | A streamed response records output, token usage, explicit stream timing proxies, and runtime metadata. | Unit | `tests/runtimes/test_llama_cpp.py` | Native prefill/decode fields remain unavailable for this binding. |
| B4 | The renamed experiment declares the same model, prompt, tasks, sampling, context lengths, and quantization variants. | Contract / file | `tests/quantization/test_experiment_contract.py` | Q8_0, Q6_K, Q5_K_M, and Q4_K_M are required. |
| B5 | Notebook-facing aggregation joins measured trial summaries to artifact size and emits required trade-off fields. | Unit | `tests/analysis/test_quantization.py` | Missing required conditions or metrics fail loudly. |
| B6 | The recommendation selects the smallest measured artifact within a declared tolerance of the best end-to-end success. | Unit | `tests/analysis/test_quantization.py` | No recommendation is returned when measurements are incomplete. |

### Acceptance Criteria as Tests

| Acceptance Criterion | Test ID | Test Name |
| --- | --- | --- |
| Quantization approach and runtime are named and documented. | B2, B4 | `test_experiment_config_names_llama_cpp_gguf_and_fixed_conditions` |
| All compared artifacts are traceable to reproducible sources/procedures. | B1, B4 | `test_manifest_preserves_variant_provenance_and_controls` |
| Short/medium runs use fixed prompts, tasks, runtime, and sampling. | B4 | `test_experiment_config_names_llama_cpp_gguf_and_fixed_conditions` |
| End-to-end success, failure rate, memory, and explicit stream timing proxies are available to analysis. | B3, B5 | `test_streamed_generation_records_measurement_fields` and `test_tradeoff_rows_join_required_metrics` |
| Notebook analysis contains separate capability-vs-artifact-size, capability-vs-sampled-RSS, TTFT, and stream-throughput-proxy figures. | B5 | `test_notebook_contains_required_analysis_sections` |
| A recommended baseline is justified by measured data. | B6 | `test_recommendation_prefers_smallest_artifact_within_accuracy_tolerance` |

### Test-First Implementation Cycles

#### Cycle 1: Quantization provenance contracts

**Red**

- Add `tests/quantization/test_specs.py` for variant validation, duplicate
  condition rejection, and manifest serialization.
- Expected failure: `ModuleNotFoundError` because the quantization contracts do
  not exist.
- Run: `PYTHONPATH=src python -m unittest tests.quantization.test_specs -v`

**Green**

- Add `src/llm_lab/quantization/specs.py` and export its public types.
- Keep artifact hashing and runtime loading out of this first change.
- Run the focused test file, then the existing suite.

**Refactor**

- Centralize validation messages and keep serialized field names stable.
- Run `PYTHONPATH=src python -m unittest tests.quantization.test_specs -v`.

#### Cycle 2: `llama.cpp` runtime boundary

**Red**

- Add `tests/runtimes/test_llama_cpp.py` with an injected fake streaming client.
- Expected failure: the runtime adapter and stream normalization are missing.
- Run: `PYTHONPATH=src python -m unittest tests.runtimes.test_llama_cpp -v`

**Green**

- Add `src/llm_lab/runtimes/llama_cpp.py` with lazy optional dependency loading,
  fixed option forwarding, stream assembly, token accounting, and timing.
- Export `LlamaCppRuntime` without importing `llama_cpp` at package import time.
- Run the focused runtime tests.

**Refactor**

- Isolate backend chunk parsing and document `first_stream_chunk` timing
  semantics; keep the common runtime protocol unchanged.
- Run the runtime tests and existing runtime tests.

#### Cycle 3: Rename and freeze the experiment protocol

**Red**

- Add `tests/quantization/test_experiment_contract.py` for the experiment path,
  config values, required variants, and fixed controls.
- Expected failure: the old `approach_a` path/config does not satisfy the
  contract.
- Run: `PYTHONPATH=src python -m unittest tests.quantization.test_experiment_contract -v`

**Green**

- Rename the directory to `experiments/exp_002-quantization_llama_cpp_gguf/`.
- Update `README.md`, `config.yaml`, `results/README.md`, shared experiment
  references, and a manifest template with conversion and provenance fields.
- Keep generated weights/raw results out of Git and require exact hashes in a
  resolved run manifest.
- Run the contract tests and `git diff --check`.

**Refactor**

- Remove all stale `approach_a` references and make the README explain the
  short/medium matrix and kernel/runtime caveats.
- Run the contract tests again.

#### Cycle 4: Notebook-ready trade-off analysis

**Red**

- Add `tests/analysis/test_quantization.py` for aggregation, required metric
  validation, and the recommendation rule; add a notebook contract test for
  the required figure sections.
- Expected failure: no quantization analysis helpers or required notebook cells
  exist.
- Run: `PYTHONPATH=src python -m unittest tests.analysis.test_quantization -v`

**Green**

- Add `src/llm_lab/analysis/quantization.py` with joins, required-field checks,
  Pareto-ready rows, and recommendation selection.
- Replace the placeholder notebook with cells that load recorded summaries and
  the resolved manifest, plot end-to-end-success-vs-memory and explicit
  stream-proxy-vs-memory,
  and print the recommendation and limitations.
- Run focused analysis tests and validate notebook JSON/syntax without claiming
  a result when `results/` is absent.

**Refactor**

- Keep plotting in the notebook and all data rules in reusable Python helpers;
  make missing cells/metrics fail loudly.
- Run the complete suite and notebook structure checks.

### Final Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```

The 27B sweep itself requires locally provisioned GGUF artifacts, a resolved
model/runtime revision, and suitable hardware. Until those inputs exist, the
notebook and recommendation code must remain unexecuted rather than turning
fixture values into research claims.

### Definition of Done

- [ ] Every behaviour in the inventory has a passing test.
- [ ] Each production change is justified by a prior failing test.
- [ ] The experiment path no longer contains `approach_a`.
- [ ] Runtime, quantization format, provenance, controls, and caveats are
  documented.
- [ ] Analysis produces the required figures from measured result files.
- [ ] Missing artifacts/results fail loudly and no fabricated recommendation is
  committed.
- [ ] Focused and final verification commands pass.
