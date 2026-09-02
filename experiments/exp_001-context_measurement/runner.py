"""Run the exp_001 context-length and evidence-position matrix.

The fixture backend validates the experiment plumbing without model weights.
Use ``--backend llama.cpp`` with the resolved Q8_0 GGUF for the operational
Qwen reference, or ``--backend transformers`` when a matching full-precision
environment is intentionally being measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.context import (  # noqa: E402
    ContextTokenizer,
    Evidence,
    InferenceTokenizer,
    SyntheticContextGenerator,
)
from llm_lab.datasets import TaskCatalog  # noqa: E402
from llm_lab.evaluation import (  # noqa: E402
    EvaluationRunner,
    EvaluationTask,
    CalibratedAnswerScorer,
    TrialResult,
    TrialStatus,
    load_trial_results,
)
from llm_lab.generation import (  # noqa: E402
    GenerationRequest,
    GenerationResponse,
    GenerationTiming,
    RuntimeMetadata,
    SamplingConfig,
    TokenUsage,
)
from llm_lab.models import ModelSpec, qwen38_model_spec  # noqa: E402
from llm_lab.runtimes import RuntimeConfig  # noqa: E402
from llm_lab.runtimes.llama_cpp import LlamaCppRuntime  # noqa: E402
from llm_lab.runtimes.transformers import QwenTransformersRuntime  # noqa: E402
from llm_lab.telemetry import capture_environment  # noqa: E402


CONFIG_PATH = ROOT / "experiments/exp_001-context_measurement/config.yaml"
TASK_CATALOG = ROOT / "data/tasks/core.v002.jsonl"
EXPERIMENT_ID = "exp_001"
SCORER_VERSION = CalibratedAnswerScorer.name
TASK_TYPES = ("literal_retrieval", "semantic_retrieval", "multi_hop")
SMOKE_CONTEXT_LENGTHS = (8192, 32768)
PILOT_CONTEXT_LENGTHS = (8192, 32768, 65536)
MAIN_CONTEXT_LENGTHS = (8192, 32768, 65536, 131072)
SMOKE_POSITIONS = (0.05, 0.50, 0.95)
FULL_POSITIONS = (0.05, 0.25, 0.50, 0.75, 0.95)
REPEATS = {"smoke": 1, "pilot": 5, "main": 20}


class FixtureTokenizer:
    """Deterministic tokenizer used by tokenizer-contract tests only."""

    name = "tokenizer-v1"

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


@dataclass(frozen=True)
class Condition:
    target_context_tokens: int
    evidence_position: float

    @property
    def condition_id(self) -> str:
        return (
            f"baseline:ctx{self.target_context_tokens:06d}:"
            f"p{int(self.evidence_position * 100):03d}"
        )


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the committed YAML experiment contract."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("exp_001 requires the PyYAML dependency") from error
    try:
        value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except yaml.YAMLError as error:
        raise ValueError(f"invalid experiment config: {config_path}") from error
    if not isinstance(value, dict) or not isinstance(value.get("phases"), dict):
        raise ValueError("experiment config must declare phases")
    return value


def planned_conditions(
    phase: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> list[Condition]:
    """Return the deterministic condition grid from the phase contract."""

    phase_config = (config or load_config()).get("phases", {}).get(phase)
    if not isinstance(phase_config, Mapping):
        raise ValueError(f"unsupported phase: {phase!r}")
    try:
        lengths = tuple(int(value) for value in phase_config["lengths"])
        positions = tuple(float(value) for value in phase_config["evidence_positions"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"phase {phase!r} must declare lengths and evidence_positions") from error
    if not lengths or not positions:
        raise ValueError(f"phase {phase!r} must declare non-empty dimensions")
    return [
        Condition(length, position)
        for length in lengths
        for position in positions
    ]


def build_tasks(
    catalog: TaskCatalog,
    condition: Condition,
    *,
    fixture_seed: int,
    tokenizer: ContextTokenizer | None = None,
) -> list[EvaluationTask]:
    """Build reproducible evaluation tasks for one context/position condition."""

    generator = SyntheticContextGenerator(tokenizer=tokenizer)
    tasks: list[EvaluationTask] = []
    for definition in catalog.tasks:
        task_seed = fixture_seed + int(definition.metadata["seed"])
        generated = generator.generate(
            [
                Evidence(
                    id=str(item["id"]),
                    text=str(item["text"]),
                )
                for item in definition.evidence
            ],
            target_tokens=condition.target_context_tokens,
            evidence_position=condition.evidence_position,
            seed=task_seed,
        )
        spans = [asdict(span) for span in generated.evidence]
        metadata = {
            "corpus_id": "corpus.synthetic.000001",
            "fixture_seed": fixture_seed,
            "task_seed": task_seed,
            "target_context_tokens": condition.target_context_tokens,
            "actual_context_tokens": generated.token_count,
            "requested_evidence_position": condition.evidence_position,
            "actual_evidence_position": sum(
                span["actual_position"] for span in spans
            )
            / len(spans),
            "evidence_spans": spans,
            "context_generator": generated.metadata["generator"],
            "context_tokenization": generated.metadata["tokenization"],
            "context_tokenization_mode": generated.metadata.get(
                "tokenization_mode", "whitespace-fixture"
            ),
            "target_unit": (
                "inference-tokenizer-tokens"
                if tokenizer is not None
                else "whitespace-fixture-tokens"
            ),
            "context_instance_id": (
                f"{definition.task_id}:{condition.condition_id}:seed{task_seed}"
            ),
            "context_sha256": hashlib.sha256(generated.text.encode("utf-8")).hexdigest(),
        }
        tasks.append(
            EvaluationTask.from_definition(
                definition,
                context=generated.text,
                metadata=metadata,
            )
        )
    return tasks


class FixtureRuntime:
    """Deterministic backend for smoke validation; it is not model evidence."""

    name = "fixture"

    def __init__(self, answers: Mapping[str, str] | None = None) -> None:
        self._answers = dict(answers or {})

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        del model, config

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        task_id = str(request.metadata["task_id"])
        output = self._answers[task_id]
        prompt_tokens = len(request.prompt.split())
        return GenerationResponse(
            output_text=output,
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=1),
            timing=GenerationTiming(
                ttft_seconds=0.001,
                prefill_seconds=0.002,
                decode_seconds=0.001,
                total_seconds=0.004,
            ),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version="1.0",
                model_id=request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config={"purpose": "harness-smoke-only"},
            ),
        )

    def close(self) -> None:
        return None


def _fixture_answers(catalog: TaskCatalog) -> dict[str, str]:
    """Return canonical answers for the deterministic smoke backend."""

    answers: dict[str, str] = {}
    for task in catalog.tasks:
        value = task.expected.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"fixture smoke requires a string expected value: {task.task_id}"
            )
        answers[task.task_id] = value
    return answers


def run_experiment(
    *,
    phase: str,
    backend: str,
    output_path: Path,
    manifest_path: Path,
    fixture_seed: int | None = None,
    overwrite_smoke: bool = False,
    resume: bool = False,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Execute one phase and write raw JSONL plus a coverage manifest."""

    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    experiment_config = config.get("experiment")
    if not isinstance(experiment_config, Mapping):
        raise ValueError("experiment config must declare experiment settings")
    configured_catalog = experiment_config.get("task_catalog")
    if not isinstance(configured_catalog, str) or not configured_catalog.strip():
        raise ValueError("experiment config must declare task_catalog")
    catalog_path = _rooted(Path(configured_catalog))
    model_config = config.get("model")
    configured_model = (
        model_config.get("model") if isinstance(model_config, Mapping) else None
    )
    model = qwen38_model_spec(
        revision=_declared_revision(
            model_config.get("revision") if isinstance(model_config, Mapping) else None
        ),
        tokenizer_revision=_declared_revision(
            model_config.get("tokenizer_revision")
            if isinstance(model_config, Mapping)
            else None
        ),
    )
    if configured_model != model.model_id:
        raise ValueError(
            f"config model {configured_model!r} does not match {model.model_id!r}"
        )
    if fixture_seed is None:
        fixture_seed = int(experiment_config.get("fixture_seed", 42))
    output_path = _rooted(output_path)
    manifest_path = _rooted(manifest_path)
    resume_manifest = _load_resume_manifest(manifest_path) if resume else None
    if overwrite_smoke and (phase != "smoke" or backend != "fixture"):
        raise ValueError("overwrite_smoke is only supported for the fixture smoke phase")
    if backend == "fixture" and phase != "smoke":
        raise ValueError(
            "fixture backend is harness-only and is permitted only for the smoke phase"
        )
    if overwrite_smoke and resume:
        raise ValueError("overwrite_smoke and resume cannot be combined")
    if output_path == manifest_path:
        raise ValueError("output and manifest paths must be different")
    if not overwrite_smoke and not resume and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            "output or manifest already exists; choose new paths or pass "
            "--resume for an interrupted model run or --overwrite-smoke for an "
            "explicit fixture regeneration"
        )
    if resume and backend == "fixture":
        raise ValueError("resume is reserved for model-backed runs; use overwrite_smoke for fixture data")

    temporary_output = _temporary_sibling(output_path) if overwrite_smoke else None
    temporary_manifest = _temporary_sibling(manifest_path) if overwrite_smoke else None
    working_output_path = temporary_output or output_path
    catalog = TaskCatalog.from_jsonl(catalog_path)
    conditions = planned_conditions(phase, config=config)
    max_context_tokens = max(
        condition.target_context_tokens for condition in conditions
    )
    context_tokenizer: ContextTokenizer | None = None
    runtime_options: dict[str, Any] = {}
    runtime_record: dict[str, Any] = {}
    if backend == "fixture":
        runtime: Any = FixtureRuntime(_fixture_answers(catalog))
        runtime_record = {"name": "fixture", "version": None, "options": {}}
    elif backend == "transformers":
        runtime = QwenTransformersRuntime()
        runtime.load(
            model,
            RuntimeConfig(
                name="transformers",
                options={
                    "device_map": "auto",
                    "trust_remote_code": True,
                },
            ),
        )
        runtime_record = {
            "name": "transformers",
            "version": None,
            "options": {"device_map": "auto", "trust_remote_code": True},
        }
        model = runtime.resolved_model_spec()
        if not model.revision or not model.tokenizer_revision:
            runtime.close()
            raise RuntimeError(
                "real-model runs require resolved model and tokenizer commit hashes"
            )
        tokenizer_id = model.tokenizer_id or model.model_id
        revision = model.tokenizer_revision or "unresolved"
        context_tokenizer = InferenceTokenizer(
            backend=runtime.get_tokenizer(),
            name=f"transformers:{tokenizer_id}@{revision}",
        )
    elif backend == "llama.cpp":
        runtime = LlamaCppRuntime()
        resolved_runtime = _llama_cpp_options(
            config,
            max_context_tokens=max_context_tokens,
        )
        runtime_version = resolved_runtime.pop("version", None)
        artifact_record = {
            "uri": resolved_runtime.pop("_artifact_uri"),
            "sha256": resolved_runtime.pop("_artifact_sha256"),
            "size_bytes": resolved_runtime.pop("_artifact_size_bytes"),
        }
        runtime_options = dict(resolved_runtime)
        runtime.load(
            model,
            RuntimeConfig(
                name="llama.cpp",
                version=runtime_version,
                options=runtime_options,
            ),
        )
        runtime_record = {
            "name": "llama.cpp",
            "version": runtime_version,
            "options": dict(runtime_options),
            "artifact": artifact_record,
        }
        tokenizer_id = model.tokenizer_id or model.model_id
        revision = model.tokenizer_revision or "embedded"
        # ``runtime.get_tokenizer()`` already returns the llama.cpp-specific
        # adapter.  Do not wrap it in the Transformers adapter, whose
        # ``add_special_tokens`` call is not part of the llama.cpp surface.
        context_tokenizer = runtime.get_tokenizer()
    else:
        raise ValueError(f"unsupported backend: {backend!r}")

    working_output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[TrialResult] = []
    phase_config = config["phases"][phase]
    # Greedy repeats are not independent capability observations.  The phase
    # may retain a larger ``repeats`` envelope for legacy/timing probes, while
    # new capability runs explicitly use one run per independent task.
    repeats = int(
        phase_config.get("capability_repeats", phase_config["repeats"])
    )
    context_provenance = _context_provenance(
        config_path=config_file,
        catalog_path=catalog_path,
        catalog=catalog,
        conditions=conditions,
        repeats=repeats,
        fixture_seed=fixture_seed,
    )
    if resume:
        _validate_resume_checkpoint(
            resume_manifest,
            phase=phase,
            backend=backend,
            output_path=output_path,
            model=model,
            context_provenance=context_provenance,
        )
    sampling_config = config.get("sampling", {})
    if not isinstance(sampling_config, Mapping):
        raise ValueError("sampling config must be an object")
    sampling, generation_seed_policy = _sampling_from_config(
        sampling_config,
        resume_manifest=resume_manifest,
    )
    existing_results = load_trial_results(working_output_path) if resume else []
    if resume:
        _validate_resume_sampling(existing_results, sampling)
    existing_ids = {result.trial_id for result in existing_results}
    if backend in ("transformers", "llama.cpp") and not resume:
        _write_resume_checkpoint(
            manifest_path=manifest_path,
            phase=phase,
            backend=backend,
            model=model,
            output_path=output_path,
            sampling=sampling,
            generation_seed_policy=generation_seed_policy,
            context_provenance=context_provenance,
        )
    try:
        try:
            runner = EvaluationRunner(
                runtime=runtime,
                model=model,
                scorer=CalibratedAnswerScorer(),
                experiment_id=EXPERIMENT_ID,
                output_path=working_output_path,
            )
            for condition in conditions:
                for task in build_tasks(
                    catalog,
                    condition,
                    fixture_seed=fixture_seed,
                    tokenizer=context_tokenizer,
                ):
                    repeat_indices = tuple(
                        index
                        for index in range(1, repeats + 1)
                        if not resume
                        or runner_trial_id(
                            EXPERIMENT_ID,
                            task.task_id,
                            condition.condition_id,
                            index,
                        ) not in existing_ids
                    )
                    if not repeat_indices:
                        continue
                    new_results = runner.run(
                        [task],
                        repeats=repeats,
                        repeat_indices=repeat_indices,
                        condition_id=condition.condition_id,
                        sampling=sampling,
                    )
                    results.extend(new_results)
                    existing_ids.update(result.trial_id for result in new_results)
        finally:
            runtime.close()
    except BaseException:
        _remove_temporary(temporary_output)
        _remove_temporary(temporary_manifest)
        raise

    if temporary_output is not None:
        os.replace(temporary_output, output_path)
    persisted_results = load_trial_results(output_path)
    manifest = _manifest(
        phase=phase,
        backend=backend,
        output_path=output_path,
        manifest_path=manifest_path,
        conditions=conditions,
        catalog=catalog,
        repeats=repeats,
        results=persisted_results,
        fixture_seed=fixture_seed,
        catalog_path=catalog_path,
        config=config,
        sampling=sampling,
        generation_seed_policy=generation_seed_policy,
        model=model,
        context_provenance=context_provenance,
        runtime_record=runtime_record,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        (temporary_manifest or manifest_path).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if temporary_manifest is not None:
            os.replace(temporary_manifest, manifest_path)
    except BaseException:
        _remove_temporary(temporary_manifest)
        raise
    return manifest


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    return temporary_path


def _remove_temporary(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def _manifest(
    *,
    phase: str,
    backend: str,
    output_path: Path,
    manifest_path: Path,
    conditions: Iterable[Condition],
    catalog: TaskCatalog,
    repeats: int,
    results: Iterable[TrialResult],
    fixture_seed: int,
    catalog_path: Path | None = None,
    config: Mapping[str, Any] | None = None,
    sampling: SamplingConfig | None = None,
    generation_seed_policy: str | None = None,
    model: ModelSpec | None = None,
    context_provenance: Mapping[str, Any] | None = None,
    runtime_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result_list = list(results)
    condition_list = list(conditions)
    by_cell: dict[tuple[str, str], list[TrialResult]] = {}
    for result in result_list:
        by_cell.setdefault(
            (result.input["task_type"], result.input["condition_id"]), []
        ).append(result)

    coverage: list[dict[str, Any]] = []
    for condition in condition_list:
        for task_type in TASK_TYPES:
            cell_results = by_cell.get((task_type, condition.condition_id), [])
            statuses = Counter(result.status.value for result in cell_results)
            scored_n = sum(
                result.score.get("correct") is not None for result in cell_results
            )
            independent_task_n = sum(
                task.task_type == task_type for task in catalog.tasks
            )
            expected_trial_n = independent_task_n * repeats
            coverage.append(
                {
                    "task_type": task_type,
                    "condition_id": condition.condition_id,
                    "target_context_tokens": condition.target_context_tokens,
                    "requested_evidence_position": condition.evidence_position,
                    "independent_task_n": independent_task_n,
                    "expected_trial_n": expected_trial_n,
                    "trial_n": len(cell_results),
                    "scored_n": scored_n,
                    "statuses": dict(sorted(statuses.items())),
                    "status": (
                        "valid"
                        if len(cell_results) == expected_trial_n
                        else "excluded"
                    ),
                    "exclusion_reason": _exclusion_reason(
                        expected_trial_n=expected_trial_n,
                        trial_n=len(cell_results),
                        scored_n=scored_n,
                        statuses=statuses,
                    ),
                }
            )

    raw_sha256 = _sha256(output_path)
    context_lengths = sorted({condition.target_context_tokens for condition in condition_list})
    evidence_positions = sorted({condition.evidence_position for condition in condition_list})
    effective_context = (
        dict(config.get("effective_context", {}))
        if config is not None and isinstance(config.get("effective_context"), Mapping)
        else {}
    )
    sampling_record = (
        sampling.to_record()
        if sampling is not None
        else SamplingConfig().to_record()
    )
    if generation_seed_policy is not None:
        sampling_record["generation_seed_policy"] = generation_seed_policy
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scorer_version": SCORER_VERSION,
        "phase": phase,
        "backend": backend,
        "runtime": dict(runtime_record or {"name": backend, "version": None, "options": {}}),
        "fixture_seed": fixture_seed,
        "task_catalog": _relative_or_absolute(catalog_path or TASK_CATALOG),
        "task_ids": list(catalog.ids),
        "context_lengths": context_lengths,
        "evidence_positions": evidence_positions,
        "task_types": list(TASK_TYPES),
        "independent_task_n_by_type": {
            task_type: sum(task.task_type == task_type for task in catalog.tasks)
            for task_type in TASK_TYPES
        },
        "effective_context": effective_context,
        "sampling": sampling_record,
        "model": _model_record(model) if model is not None else None,
        "repeats": repeats,
        "capability_repeats": repeats,
        "timing_repeats": (
            int(
                config.get("phases", {})
                .get(phase, {})
                .get("timing_repeats", repeats)
            )
            if isinstance(config, Mapping)
            else repeats
        ),
        "planned_condition_n": len(condition_list),
        "planned_cell_n": len(coverage),
        "planned_trial_n": len(condition_list) * len(catalog.tasks) * repeats,
        "actual_trial_n": len(result_list),
        "raw_results": _relative_or_absolute(output_path),
        "raw_results_sha256": raw_sha256,
        "manifest_path": _relative_or_absolute(manifest_path),
        "environment": capture_environment(ROOT),
        "coverage": coverage,
        "excluded_cells": [
            row for row in coverage if row["status"] == "excluded"
        ],
        "interpretation": (
            "Fixture backend validates task construction, scoring, storage, and "
            "coverage only; it is not a Qwen measurement."
            if backend == "fixture"
            else "Results are model/runtime observations under the recorded environment."
        ),
    }
    if context_provenance is not None:
        manifest["context_provenance"] = dict(context_provenance)
    return manifest


def _context_provenance(
    *,
    config_path: Path,
    catalog_path: Path,
    catalog: TaskCatalog,
    conditions: Iterable[Condition],
    repeats: int,
    fixture_seed: int,
) -> dict[str, Any]:
    environment = capture_environment(ROOT)
    source_revision = environment.get("git_sha")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise RuntimeError(
            "exp_001 requires a resolvable repository revision for resume provenance"
        )
    return {
        "source_revision": source_revision,
        "fixture_seed": fixture_seed,
        "config_path": _relative_or_absolute(config_path),
        "config_sha256": _sha256(config_path),
        "task_catalog": _relative_or_absolute(catalog_path),
        "task_catalog_sha256": _sha256(catalog_path),
        "task_ids": list(catalog.ids),
        "task_types": list(TASK_TYPES),
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "target_context_tokens": condition.target_context_tokens,
                "evidence_position": condition.evidence_position,
            }
            for condition in conditions
        ],
        "repeats": repeats,
    }


def _declared_revision(value: Any) -> str | None:
    if value is None or value == "record-at-run-time":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model revisions must be strings or record-at-run-time")
    return value


def _model_record(model: ModelSpec | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {
        "id": model.model_id,
        "revision": model.revision,
        "tokenizer_id": model.tokenizer_id,
        "tokenizer_revision": model.tokenizer_revision,
    }


def _sampling_from_config(
    values: Mapping[str, Any],
    *,
    resume_manifest: Mapping[str, Any] | None = None,
) -> tuple[SamplingConfig, str]:
    """Resolve config sampling into effective settings and a provenance policy."""

    temperature = float(values.get("temperature", 0.0))
    configured_seed = values.get("generation_seed")
    previous_sampling = _resume_sampling(resume_manifest)
    if configured_seed == "record-at-run-time":
        if temperature == 0.0:
            seed = None
            policy = "greedy-decoding-no-seed"
        elif previous_sampling is not None:
            seed = _required_resume_seed(previous_sampling)
            policy = "run-resolved-seed"
        else:
            seed = secrets.randbits(32)
            policy = "run-resolved-seed"
    elif configured_seed is None:
        seed = None
        policy = "greedy-decoding-no-seed" if temperature == 0.0 else "unseeded-sampling"
    elif isinstance(configured_seed, bool):
        raise ValueError("sampling generation_seed must be an integer or record-at-run-time")
    else:
        seed = int(configured_seed)
        policy = "configured-seed"
    sampling = SamplingConfig(
        max_new_tokens=int(values.get("max_new_tokens", 32)),
        temperature=temperature,
        top_p=float(values.get("top_p", 1.0)),
        top_k=(
            None
            if values.get("top_k") is None
            else int(values["top_k"])
        ),
        seed=seed,
    )
    if previous_sampling is not None:
        expected = dict(sampling.to_record())
        expected["generation_seed_policy"] = policy
        _validate_sampling_match(previous_sampling, expected)
    return sampling, policy


def _llama_cpp_options(
    config: Mapping[str, Any],
    *,
    max_context_tokens: int | None = None,
) -> dict[str, Any]:
    """Resolve and verify the operational Q8_0 llama.cpp reference artifact."""

    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("llama.cpp backend requires a runtime mapping")
    values = runtime.get("llama_cpp", runtime)
    if not isinstance(values, Mapping):
        raise ValueError("runtime.llama_cpp must be a mapping")
    configured_path = values.get("model_path") or os.environ.get(
        "EXP001_LLAMA_MODEL_PATH"
    )
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError(
            "llama.cpp backend requires runtime.llama_cpp.model_path or "
            "EXP001_LLAMA_MODEL_PATH"
        )
    model_path = _rooted(Path(configured_path))
    if not model_path.is_file():
        raise FileNotFoundError(f"llama.cpp model artifact is missing: {model_path}")
    expected_size = values.get("artifact_size_bytes")
    actual_size = model_path.stat().st_size
    if expected_size is not None and actual_size != int(expected_size):
        raise ValueError(
            "llama.cpp model artifact size mismatch: "
            f"expected {expected_size}, found {actual_size}"
        )
    expected_sha256 = values.get("artifact_sha256")
    actual_sha256 = _sha256(model_path)
    if expected_sha256 is not None:
        if actual_sha256.lower() != str(expected_sha256).lower():
            raise ValueError(
                "llama.cpp model artifact SHA-256 mismatch: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )
    configured_n_ctx = int(values.get("n_ctx", 131392))
    effective_n_ctx = configured_n_ctx
    if max_context_tokens is not None:
        context_config = config.get("context", {})
        declared_lengths = (
            context_config.get("lengths", ())
            if isinstance(context_config, Mapping)
            else ()
        )
        try:
            declared_max_context = max(int(value) for value in declared_lengths)
        except (TypeError, ValueError):
            declared_max_context = 0
        configured_overhead = max(0, configured_n_ctx - declared_max_context)
        effective_n_ctx = int(max_context_tokens) + configured_overhead
        if effective_n_ctx <= int(max_context_tokens):
            effective_n_ctx = int(max_context_tokens) + 1
    options: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": effective_n_ctx,
        "n_batch": int(values.get("n_batch", 512)),
        "n_gpu_layers": int(values.get("n_gpu_layers", -1)),
        "flash_attn": bool(values.get("flash_attn", True)),
        "verbose": bool(values.get("verbose", False)),
        "_artifact_uri": str(values.get("artifact_uri", configured_path)),
        "_artifact_sha256": actual_sha256,
        "_artifact_size_bytes": actual_size,
    }
    version = values.get("version")
    if version is not None:
        options["version"] = str(version)
    return options


def _load_resume_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"resume requires the existing run manifest with sampling provenance: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid resume manifest JSON: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("resume manifest must be a schema version 1 object")
    return value


def _validate_resume_checkpoint(
    manifest: Mapping[str, Any] | None,
    *,
    phase: str,
    backend: str,
    output_path: Path,
    model: ModelSpec,
    context_provenance: Mapping[str, Any],
) -> None:
    if manifest is None:
        raise ValueError("resume requires an existing run checkpoint")
    identity_fields = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "backend": backend,
        "status": "in_progress",
    }
    mismatches = [
        field
        for field, expected in identity_fields.items()
        if manifest.get(field) != expected
    ]
    raw_results = manifest.get("raw_results")
    if not isinstance(raw_results, str) or not raw_results.strip():
        mismatches.append("raw_results")
    elif _rooted(Path(raw_results)).resolve() != output_path.resolve():
        mismatches.append("raw_results")

    recorded_model = manifest.get("model")
    expected_model = _model_record(model)
    if not isinstance(recorded_model, Mapping):
        mismatches.append("model")
    else:
        for field in ("id", "revision", "tokenizer_id", "tokenizer_revision"):
            if recorded_model.get(field) != expected_model[field]:
                mismatches.append(f"model.{field}")

    recorded_context = manifest.get("context_provenance")
    if not isinstance(recorded_context, Mapping):
        mismatches.append("context_provenance")
    else:
        for field in (
            "source_revision",
            "fixture_seed",
            "config_path",
            "config_sha256",
            "task_catalog",
            "task_catalog_sha256",
            "task_ids",
            "task_types",
            "conditions",
            "repeats",
        ):
            if recorded_context.get(field) != context_provenance.get(field):
                mismatches.append(f"context_provenance.{field}")

    if mismatches:
        raise ValueError(
            "resume checkpoint identity mismatch: "
            f"{sorted(set(mismatches))}"
        )


def _resume_sampling(
    manifest: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if manifest is None:
        return None
    value = manifest.get("sampling")
    if not isinstance(value, Mapping):
        raise ValueError("resume manifest is missing sampling provenance")
    return value


def _required_resume_seed(sampling: Mapping[str, Any]) -> int:
    seed = sampling.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("resume manifest is missing the resolved generation seed")
    return seed


def _validate_sampling_match(
    recorded: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    fields = (
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "do_sample",
        "generation_seed_policy",
    )
    missing = [field for field in fields if field not in recorded]
    mismatches = [
        field
        for field in fields
        if field in recorded and recorded.get(field) != expected.get(field)
    ]
    if missing or mismatches:
        raise ValueError(
            "sampling provenance mismatch on resume: "
            f"missing={missing}, mismatched={mismatches}"
        )


def _validate_resume_sampling(
    results: Iterable[TrialResult],
    sampling: SamplingConfig,
) -> None:
    expected = sampling.to_record()
    for result in results:
        recorded = result.input.get("sampling")
        if not isinstance(recorded, Mapping):
            raise ValueError(
                f"existing trial {result.trial_id} is missing sampling provenance"
            )
        _validate_sampling_match(recorded, expected)


def _write_resume_checkpoint(
    *,
    manifest_path: Path,
    phase: str,
    backend: str,
    model: ModelSpec,
    output_path: Path,
    sampling: SamplingConfig,
    generation_seed_policy: str,
    context_provenance: Mapping[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sampling_record = sampling.to_record()
    sampling_record["generation_seed_policy"] = generation_seed_policy
    checkpoint = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "backend": backend,
        "status": "in_progress",
        "model": _model_record(model),
        "sampling": sampling_record,
        "context_provenance": dict(context_provenance),
        "raw_results": _relative_or_absolute(output_path),
    }
    manifest_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _exclusion_reason(
    *,
    expected_trial_n: int,
    trial_n: int,
    scored_n: int,
    statuses: Mapping[str, int],
) -> str | None:
    # A complete attempted cell is valid even when some trials fail at
    # runtime or produce invalid output. Those failures are part of the
    # end-to-end denominator and are reported by the aggregate counters.
    if trial_n == expected_trial_n:
        return None
    reasons: list[str] = []
    if trial_n != expected_trial_n:
        reasons.append(f"expected {expected_trial_n} trials, recorded {trial_n}")
    if scored_n != expected_trial_n:
        status_text = ", ".join(
            f"{status}={count}"
            for status, count in sorted(statuses.items())
            if status != TrialStatus.COMPLETED.value or scored_n != expected_trial_n
        )
        reasons.append(f"scored {scored_n}/{expected_trial_n}; statuses: {status_text or 'none'}")
    return "; ".join(reasons)


def runner_trial_id(
    experiment_id: str,
    task_id: str,
    condition_id: str,
    repeat_index: int,
) -> str:
    """Build the same ID as EvaluationRunner without coupling to its internals."""

    return f"{experiment_id}:{task_id}:{condition_id}:run{repeat_index:02d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(REPEATS), default="smoke")
    parser.add_argument(
        "--backend",
        choices=("fixture", "transformers", "llama.cpp"),
        default="fixture",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fixture-seed", type=int)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted model-backed raw JSONL run",
    )
    parser.add_argument(
        "--overwrite-smoke",
        action="store_true",
        help="atomically regenerate existing fixture smoke artifacts",
    )
    args = parser.parse_args(argv)
    output_path = args.output or ROOT / "experiments/exp_001-context_measurement/results/raw/{phase}-trials.jsonl".format(phase=args.phase)
    manifest_path = args.manifest or ROOT / "experiments/exp_001-context_measurement/results/manifests/{phase}.json".format(phase=args.phase)
    manifest = run_experiment(
        phase=args.phase,
        backend=args.backend,
        output_path=output_path,
        manifest_path=manifest_path,
        fixture_seed=args.fixture_seed,
        overwrite_smoke=args.overwrite_smoke,
        resume=args.resume,
        config_path=args.config,
    )
    print(json.dumps({key: manifest[key] for key in (
        "phase", "backend", "actual_trial_n", "raw_results", "manifest_path"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
