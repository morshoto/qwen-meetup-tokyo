"""Run the matched exp_003 context-length × quantization matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.analysis import aggregate_jsonl, write_summary_csv  # noqa: E402
from llm_lab.context import ContextTokenizer, Evidence, SyntheticContextGenerator  # noqa: E402
from llm_lab.datasets import TaskCatalog  # noqa: E402
from llm_lab.evaluation import (  # noqa: E402
    EvaluationRunner,
    EvaluationTask,
    ExpectedAnswerScorer,
    TrialResult,
    TrialStatus,
    load_trial_results,
    make_trial_id,
)
from llm_lab.generation import (  # noqa: E402
    GenerationRequest,
    GenerationResponse,
    GenerationTiming,
    RuntimeMetadata,
    SamplingConfig,
    TokenUsage,
)
from llm_lab.models import ModelSpec  # noqa: E402
from llm_lab.quantization import QuantizationManifest, QuantizationVariant  # noqa: E402
from llm_lab.runtimes import LlamaCppRuntime, RuntimeConfig  # noqa: E402
from llm_lab.telemetry import capture_environment  # noqa: E402


TASK_CATALOG = ROOT / "data/tasks/core.v001.jsonl"
EXPERIMENT_ID = "exp_003"
CONFIG_PATH = ROOT / "experiments/exp_003-context_x_quantization/config.yaml"
TASK_TYPES = ("literal_retrieval", "semantic_retrieval", "multi_hop")
RuntimeFactory = Callable[[], Any]


@dataclass(frozen=True)
class Condition:
    target_context_tokens: int
    evidence_position: float

    @property
    def condition_id(self) -> str:
        return (
            f"ctx{self.target_context_tokens:06d}:"
            f"p{int(self.evidence_position * 100):03d}"
        )


class FixtureRuntime:
    """Deterministic harness backend; it is not model evidence."""

    name = "llama.cpp"
    _answers = {
        "task.literal.000001": "ZX-4817",
        "task.semantic.000001": "Reliability Engineering",
        "task.multihop.000001": "8392",
    }

    def __init__(self) -> None:
        self._model: ModelSpec | None = None
        self._config: RuntimeConfig | None = None
        self.closed = False

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        self._model = model
        self._config = config

    def get_tokenizer(self) -> ContextTokenizer:
        return _FixtureTokenizer()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        task_id = str(request.metadata["task_id"])
        output = self._answers[task_id]
        return GenerationResponse(
            output_text=output,
            usage=TokenUsage(
                prompt_tokens=len(request.prompt.split()),
                completion_tokens=1,
            ),
            timing=GenerationTiming(
                ttft_seconds=0.001,
                prefill_seconds=0.002,
                decode_seconds=0.001,
                total_seconds=0.004,
            ),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version="fixture",
                model_id=self._model.model_id if self._model else request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config={"purpose": "harness-smoke-only"},
            ),
        )

    def close(self) -> None:
        self.closed = True


class _FixtureTokenizer:
    name = "fixture-llama-tokenizer-v1"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


def load_experiment_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the committed experiment protocol used by the runner."""

    path = _rooted(path)
    try:
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid experiment config: {path}") from error
    if not isinstance(record, dict):
        raise ValueError(f"experiment config must be a mapping: {path}")
    return record


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"experiment config section {name!r} must be a mapping")
    return value


def _default_phase(config: Mapping[str, Any]) -> str:
    value = _config_section(config, "experiment").get("default_phase")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("experiment config must declare a non-empty default_phase")
    return value


def load_manifest(path: Path) -> QuantizationManifest:
    """Load the resolved exp_002 manifest used as exp_003's artifact source."""

    path = _rooted(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("template") is True or _contains_placeholder(record):
        raise ValueError("a resolved source manifest without placeholders is required")
    manifest = QuantizationManifest.from_record(record)
    if manifest.experiment_id != "exp_002":
        raise ValueError(f"unsupported source experiment: {manifest.experiment_id!r}")
    if manifest.runtime_name != "llama.cpp":
        raise ValueError(f"unsupported source runtime: {manifest.runtime_name!r}")
    return manifest


def planned_conditions(
    phase: str,
    *,
    context_lengths: Iterable[int] | None = None,
    evidence_positions: Iterable[float] | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[Condition]:
    """Return the deterministic context/position grid for a run phase."""

    phase_controls = _phase_controls(phase, config)
    lengths = _select_context_lengths(
        phase_controls["context_lengths"]
        if context_lengths is None
        else context_lengths
    )
    positions = _select_evidence_positions(
        phase_controls["evidence_positions"]
        if evidence_positions is None
        else evidence_positions
    )
    return [
        Condition(length, position)
        for length in lengths
        for position in positions
    ]


def expected_trial_count(
    manifest: QuantizationManifest,
    *,
    phase: str = "main",
    condition_ids: Iterable[str] | None = None,
    context_lengths: Iterable[int] | None = None,
    evidence_positions: Iterable[float] | None = None,
    repeats: int | None = None,
    config: Mapping[str, Any] | None = None,
) -> int:
    """Return the selected matrix size before runtime exclusions."""

    variants = _select_variants(manifest, condition_ids, config)
    conditions = planned_conditions(
        phase,
        context_lengths=context_lengths,
        evidence_positions=evidence_positions,
        config=config,
    )
    run_repeats = _select_repeats(phase, repeats, config)
    return len(variants) * len(conditions) * len(manifest.task_ids) * run_repeats


def run_experiment(
    *,
    source_manifest_path: Path,
    output_path: Path,
    manifest_output_path: Path,
    processed_path: Path,
    phase: str,
    backend: str | None = None,
    condition_ids: Iterable[str] | None = None,
    context_lengths: Iterable[int] | None = None,
    evidence_positions: Iterable[float] | None = None,
    repeats: int | None = None,
    fixture_seed: int = 42,
    runtime_factory: RuntimeFactory | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run a selected matrix, safely resuming an append-only JSONL file."""

    source_manifest_path = _rooted(source_manifest_path)
    output_path = _rooted(output_path)
    manifest_output_path = _rooted(manifest_output_path)
    processed_path = _rooted(processed_path)
    config = load_experiment_config(
        CONFIG_PATH if config_path is None else config_path
    )
    phase_controls = _phase_controls(phase, config)
    selected_backend = phase_controls["backend"] if backend is None else backend
    if selected_backend not in ("fixture", "llama.cpp"):
        raise ValueError(f"unsupported backend: {selected_backend!r}")
    source_manifest = load_manifest(source_manifest_path)
    variants = _select_variants(source_manifest, condition_ids, config)
    conditions = planned_conditions(
        phase,
        context_lengths=context_lengths,
        evidence_positions=evidence_positions,
        config=config,
    )
    run_repeats = _select_repeats(phase, repeats, config)
    catalog = TaskCatalog.from_jsonl(TASK_CATALOG)
    if tuple(source_manifest.task_ids) != catalog.ids:
        raise ValueError("source manifest task_ids must exactly match the shared task catalog")
    sampling = _sampling_config(source_manifest.sampling)
    fingerprint = _run_fingerprint(
        source_manifest,
        phase=phase,
        backend=selected_backend,
        variant_ids=[variant.condition_id for variant in variants],
        conditions=conditions,
        repeats=run_repeats,
        fixture_seed=fixture_seed,
    )
    artifact_paths = (
        _verify_artifacts(source_manifest_path, variants)
        if selected_backend == "llama.cpp"
        else {}
    )
    model = ModelSpec(
        model_id=source_manifest.model_id,
        revision=source_manifest.model_revision,
        tokenizer_id=source_manifest.tokenizer_id,
        tokenizer_revision=source_manifest.tokenizer_revision,
    )
    existing = load_trial_results(output_path)
    expected_ids = {
        make_trial_id(
            EXPERIMENT_ID,
            task_id,
            condition_id=_execution_condition_id(variant, condition),
            repeat_index=repeat_index,
        )
        for variant in variants
        for condition in conditions
        for task_id in source_manifest.task_ids
        for repeat_index in range(1, run_repeats + 1)
    }
    _validate_existing(existing, expected_ids, fingerprint)
    existing_ids = {result.trial_id for result in existing}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_tasks_by_condition: dict[str, tuple[EvaluationTask, ...]] = {}
    total_results = len(existing)
    tokenizer: ContextTokenizer | None = None
    runtime_configs = {
        variant.condition_id: _runtime_config(
            source_manifest,
            variant,
            artifact_paths.get(variant.condition_id),
            conditions=conditions,
            sampling=sampling,
            backend=selected_backend,
        )
        for variant in variants
    }
    factory = runtime_factory or (
        FixtureRuntime if selected_backend == "fixture" else LlamaCppRuntime
    )

    for variant in variants:
        runtime = factory()
        try:
            runtime.load(
                model,
                runtime_configs[variant.condition_id],
            )
            if tokenizer is None:
                get_tokenizer = getattr(runtime, "get_tokenizer", None)
                if not callable(get_tokenizer):
                    raise TypeError("exp_003 runtime must expose get_tokenizer()")
                tokenizer = get_tokenizer()
                base_tasks_by_condition = {
                    condition.condition_id: tuple(
                        build_tasks(
                            catalog,
                            source_manifest.task_ids,
                            condition,
                            tokenizer=tokenizer,
                            prompt_id=source_manifest.prompt_id,
                            fixture_seed=fixture_seed,
                        )
                    )
                    for condition in conditions
                }
            evaluator = EvaluationRunner(
                runtime=runtime,
                model=model,
                scorer=ExpectedAnswerScorer(),
                experiment_id=EXPERIMENT_ID,
                output_path=output_path,
            )
            for condition in conditions:
                variant_tasks = tuple(
                    replace(
                        task,
                        metadata={
                            **dict(task.metadata),
                            "variant_condition_id": variant.condition_id,
                            "variant_label": variant.label,
                            "quantization_type": variant.quantization_type,
                            "artifact_uri": variant.artifact.artifact_uri,
                            "artifact_sha256": variant.artifact.artifact_sha256,
                            "artifact_size_bytes": variant.artifact.artifact_size_bytes,
                            "run_fingerprint": fingerprint,
                        },
                    )
                    for task in base_tasks_by_condition[condition.condition_id]
                )
                execution_condition_id = _execution_condition_id(variant, condition)
                for repeat_index in range(1, run_repeats + 1):
                    missing_tasks = [
                        task
                        for task in variant_tasks
                        if make_trial_id(
                            EXPERIMENT_ID,
                            task.task_id,
                            condition_id=execution_condition_id,
                            repeat_index=repeat_index,
                        )
                        not in existing_ids
                    ]
                    if not missing_tasks:
                        continue
                    new_results = evaluator.run(
                        missing_tasks,
                        repeats=1,
                        repeat_indices=(repeat_index,),
                        condition_id=execution_condition_id,
                        sampling=sampling,
                    )
                    total_results += len(new_results)
                    existing_ids.update(result.trial_id for result in new_results)
        finally:
            runtime.close()

    summaries = aggregate_jsonl(output_path)
    write_summary_csv(processed_path, summaries)
    manifest = _run_manifest(
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        output_path=output_path,
        manifest_output_path=manifest_output_path,
        phase=phase,
        backend=selected_backend,
        variants=variants,
        conditions=conditions,
        repeats=run_repeats,
        results=load_trial_results(output_path),
        fixture_seed=fixture_seed,
        catalog=catalog,
        fingerprint=fingerprint,
        effective_runtime_options_by_variant={
            condition_id: dict(runtime_config.options)
            for condition_id, runtime_config in runtime_configs.items()
        },
    )
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "expected_trial_n": len(expected_ids),
        "actual_trial_n": total_results,
        "skipped_trial_n": len(existing),
        "summary_row_n": len(summaries),
        "output_path": str(output_path),
        "manifest_output_path": str(manifest_output_path),
        "processed_path": str(processed_path),
        "run_fingerprint": fingerprint,
    }


def build_tasks(
    catalog: TaskCatalog,
    task_ids: Iterable[str],
    condition: Condition,
    *,
    tokenizer: ContextTokenizer,
    prompt_id: str,
    fixture_seed: int,
) -> list[EvaluationTask]:
    """Build one matched task set for a context/position condition."""

    generator = SyntheticContextGenerator(tokenizer=tokenizer)
    tasks: list[EvaluationTask] = []
    for task_id in task_ids:
        definition = catalog.get(task_id)
        task_seed = fixture_seed + int(definition.metadata["seed"])
        generated = generator.generate(
            [
                Evidence(id=str(item["id"]), text=str(item["text"]))
                for item in definition.evidence
            ],
            target_tokens=condition.target_context_tokens,
            evidence_position=condition.evidence_position,
            seed=task_seed,
        )
        spans = [
            {
                "id": span.id,
                "text": span.text,
                "token_start": span.token_start,
                "token_end": span.token_end,
                "requested_position": span.requested_position,
                "actual_position": span.actual_position,
            }
            for span in generated.evidence
        ]
        context_instance_id = _context_instance_id(
            task_id,
            condition.target_context_tokens,
            condition.evidence_position,
            task_seed,
        )
        tasks.append(
            EvaluationTask.from_definition(
                definition,
                context=generated.text,
                prompt_id=prompt_id,
                metadata={
                    "corpus_id": "corpus.synthetic.000001",
                    "fixture_seed": fixture_seed,
                    "task_seed": task_seed,
                    "context_instance_id": context_instance_id,
                    "context_sha256": _text_sha256(generated.text),
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
                        "tokenization_mode", "tokenizer"
                    ),
                    "target_unit": "input-tokenizer-tokens",
                    "matched_cell_key": f"{task_id}:{condition.condition_id}",
                },
            )
        )
    return tasks


def _runtime_config(
    manifest: QuantizationManifest,
    variant: QuantizationVariant,
    artifact_path: Path | None,
    *,
    conditions: Iterable[Condition],
    sampling: SamplingConfig,
    backend: str,
) -> RuntimeConfig:
    options = dict(manifest.runtime_options)
    max_context = max(condition.target_context_tokens for condition in conditions)
    options["n_ctx"] = max(
        int(options.get("n_ctx", 0)),
        max_context + sampling.max_new_tokens + manifest.context_overhead_tokens,
    )
    options["quantization_type"] = variant.quantization_type
    if backend == "fixture":
        options["purpose"] = "harness-smoke-only"
    elif artifact_path is not None:
        options["model_path"] = str(artifact_path)
    return RuntimeConfig(
        name=manifest.runtime_name,
        version=manifest.runtime_version,
        options=options,
    )


def _sampling_config(values: Mapping[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        max_new_tokens=int(values["max_new_tokens"]),
        temperature=float(values["temperature"]),
        top_p=float(values["top_p"]),
        top_k=None if values.get("top_k") is None else int(values["top_k"]),
        seed=None if values.get("seed") is None else int(values["seed"]),
    )


def _select_variants(
    manifest: QuantizationManifest,
    condition_ids: Iterable[str] | None,
    config: Mapping[str, Any] | None = None,
) -> tuple[QuantizationVariant, ...]:
    config = load_experiment_config() if config is None else config
    quantization = _config_section(config, "quantization")
    configured_variants = quantization.get("variants")
    if not isinstance(configured_variants, (list, tuple)):
        raise ValueError("experiment config quantization.variants must be a list")
    selected = tuple(
        str(value)
        for value in (
            configured_variants if condition_ids is None else tuple(condition_ids)
        )
    )
    if not selected:
        raise ValueError("at least one quantization variant must be selected")
    by_id = {variant.condition_id: variant for variant in manifest.variants}
    unknown = set(selected) - set(by_id)
    if unknown:
        raise ValueError(f"unknown variant condition IDs: {sorted(unknown)}")
    return tuple(by_id[condition_id] for condition_id in selected)


def _phase_controls(
    phase: str,
    config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    config = load_experiment_config() if config is None else config
    phases = _config_section(config, "phases")
    try:
        raw_controls = phases[phase]
    except KeyError as error:
        raise ValueError(f"unsupported phase: {phase!r}") from error
    if not isinstance(raw_controls, Mapping):
        raise ValueError(f"phase {phase!r} must be a mapping")
    try:
        lengths = raw_controls["lengths"]
        positions = raw_controls["evidence_positions"]
        repeats = raw_controls["repeats"]
        backend = raw_controls["backend"]
    except KeyError as error:
        raise ValueError(f"phase {phase!r} is missing {error.args[0]!r}") from error
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError(f"phase {phase!r} backend must be a non-empty string")
    return {
        "context_lengths": lengths,
        "evidence_positions": positions,
        "repeats": repeats,
        "backend": backend,
    }


def _select_context_lengths(values: Iterable[int]) -> tuple[int, ...]:
    selected = tuple(int(value) for value in values)
    if not selected or any(value < 1 for value in selected):
        raise ValueError("context lengths must contain positive values")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("context lengths must be strictly increasing")
    return selected


def _select_evidence_positions(values: Iterable[float]) -> tuple[float, ...]:
    selected = tuple(float(value) for value in values)
    if not selected or any(not 0.0 <= value <= 1.0 for value in selected):
        raise ValueError("evidence positions must be between 0 and 1")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("evidence positions must be strictly increasing")
    return selected


def _select_repeats(
    phase: str,
    repeats: int | None,
    config: Mapping[str, Any] | None = None,
) -> int:
    default = int(_phase_controls(phase, config)["repeats"])
    selected = default if repeats is None else int(repeats)
    if selected < 1 or selected > default:
        raise ValueError(f"repeats must be between 1 and {default} for {phase}")
    return selected


def _execution_condition_id(variant: QuantizationVariant, condition: Condition) -> str:
    return f"{variant.condition_id}:{condition.condition_id}"


def _context_instance_id(
    task_id: str,
    context_tokens: int,
    position: float,
    seed: int,
) -> str:
    return f"{task_id}:seed{seed}:ctx{context_tokens}:p{int(position * 100):03d}"


def _run_fingerprint(
    manifest: QuantizationManifest,
    *,
    phase: str,
    backend: str,
    variant_ids: Iterable[str],
    conditions: Iterable[Condition],
    repeats: int,
    fixture_seed: int,
) -> str:
    payload = {
        "source_manifest": manifest.to_record(),
        "phase": phase,
        "backend": backend,
        "variant_ids": list(variant_ids),
        "conditions": [
            {
                "target_context_tokens": condition.target_context_tokens,
                "evidence_position": condition.evidence_position,
            }
            for condition in conditions
        ],
        "repeats": repeats,
        "fixture_seed": fixture_seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_existing(
    existing: Iterable[TrialResult],
    expected_ids: set[str],
    fingerprint: str,
) -> None:
    for result in existing:
        if result.trial_id not in expected_ids:
            raise ValueError(f"existing trial is outside selected run: {result.trial_id}")
        if result.input.get("run_fingerprint") != fingerprint:
            raise ValueError("existing raw results do not match the selected run")


def _verify_artifacts(
    manifest_path: Path,
    variants: Iterable[QuantizationVariant],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for variant in variants:
        path = _artifact_path(manifest_path, variant.artifact.artifact_uri)
        if not path.is_file():
            raise FileNotFoundError(f"artifact is missing: {path}")
        if path.stat().st_size != variant.artifact.artifact_size_bytes:
            raise ValueError(f"artifact size mismatch for {variant.condition_id}")
        if _sha256(path).lower() != variant.artifact.artifact_sha256.lower():
            raise ValueError(f"artifact SHA-256 mismatch for {variant.condition_id}")
        paths[variant.condition_id] = path
    return paths


def _run_manifest(
    *,
    source_manifest_path: Path,
    source_manifest: QuantizationManifest,
    output_path: Path,
    manifest_output_path: Path,
    phase: str,
    backend: str,
    variants: Iterable[QuantizationVariant],
    conditions: Iterable[Condition],
    repeats: int,
    results: Iterable[TrialResult],
    fixture_seed: int,
    catalog: TaskCatalog,
    fingerprint: str,
    effective_runtime_options_by_variant: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result_list = list(results)
    variant_list = list(variants)
    condition_list = list(conditions)
    by_cell: dict[tuple[str, str, str], list[TrialResult]] = {}
    for result in result_list:
        by_cell.setdefault(
            (
                str(result.input["variant_condition_id"]),
                str(result.input["task_type"]),
                str(result.input["condition_id"]),
            ),
            [],
        ).append(result)

    coverage: list[dict[str, Any]] = []
    for variant in variant_list:
        for condition in condition_list:
            execution_id = _execution_condition_id(variant, condition)
            for task_type in TASK_TYPES:
                cell_results = by_cell.get((variant.condition_id, task_type, execution_id), [])
                statuses = Counter(result.status.value for result in cell_results)
                scored_n = sum(
                    result.score.get("correct") is not None
                    for result in cell_results
                )
                coverage.append(
                    {
                        "variant_condition_id": variant.condition_id,
                        "task_type": task_type,
                        "condition_id": execution_id,
                        "target_context_tokens": condition.target_context_tokens,
                        "requested_evidence_position": condition.evidence_position,
                        "trial_n": len(cell_results),
                        "scored_n": scored_n,
                        "statuses": dict(sorted(statuses.items())),
                        "status": "valid" if scored_n == repeats else "excluded",
                        "exclusion_reason": (
                            None
                            if scored_n == repeats
                            else "not all planned trials produced scored outputs; see raw results"
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "backend": backend,
        "fixture_seed": fixture_seed,
        "source_manifest": _display_path(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_experiment_id": source_manifest.experiment_id,
        "model": {
            "id": source_manifest.model_id,
            "revision": source_manifest.model_revision,
            "tokenizer_id": source_manifest.tokenizer_id,
            "tokenizer_revision": source_manifest.tokenizer_revision,
        },
        "runtime": {
            "name": source_manifest.runtime_name,
            "version": source_manifest.runtime_version,
            "source_options": dict(source_manifest.runtime_options),
            "effective_options_by_variant": {
                condition_id: dict(options)
                for condition_id, options in effective_runtime_options_by_variant.items()
            },
        },
        "task_catalog": _display_path(TASK_CATALOG),
        "prompt_id": source_manifest.prompt_id,
        "task_ids": list(catalog.ids),
        "task_types": list(TASK_TYPES),
        "quantization_variants": [variant.to_record() for variant in variant_list],
        "context_lengths": _ordered_unique(
            condition.target_context_tokens for condition in condition_list
        ),
        "evidence_positions": _ordered_unique(
            condition.evidence_position for condition in condition_list
        ),
        "repeats": repeats,
        "matching": "same task seed, generated context text, context length, and evidence position across variants",
        "planned_condition_n": len(condition_list),
        "planned_cell_n": len(coverage),
        "planned_trial_n": len(condition_list) * len(catalog.ids) * len(variant_list) * repeats,
        "actual_trial_n": len(result_list),
        "raw_results": _display_path(output_path),
        "raw_results_sha256": _sha256(output_path),
        "manifest_path": _display_path(manifest_output_path),
        "run_fingerprint": fingerprint,
        "environment": capture_environment(ROOT),
        "coverage": coverage,
        "excluded_cells": [row for row in coverage if row["status"] == "excluded"],
        "interpretation": (
            "Fixture backend validates matching, task construction, scoring, storage, and coverage only; it is not a Qwen measurement."
            if backend == "fixture"
            else "Results are model/runtime observations under the recorded environment."
        ),
    }


def _artifact_path(manifest_path: Path, artifact_uri: str) -> Path:
    if artifact_uri.startswith("file://"):
        return Path(artifact_uri.removeprefix("file://"))
    path = Path(artifact_uri)
    return path if path.is_absolute() else manifest_path.parent / path


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "REPLACE_WITH" in value or "SET_TO_" in value
    if isinstance(value, Mapping):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    config = load_experiment_config()
    phase_names = tuple(_config_section(config, "phases"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=phase_names)
    parser.add_argument("--backend", choices=("fixture", "llama.cpp"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--condition-id", action="append")
    parser.add_argument("--context-length", action="append", type=int)
    parser.add_argument("--evidence-position", action="append", type=float)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--fixture-seed", type=int, default=42)
    args = parser.parse_args(argv)
    phase = args.phase or _default_phase(config)
    experiment_root = ROOT / "experiments/exp_003-context_x_quantization/results"
    output_path = args.output or experiment_root / "raw" / f"{phase}-trials.jsonl"
    manifest_output_path = args.manifest_output or experiment_root / "manifests" / f"{phase}.json"
    processed_path = args.processed or experiment_root / "processed" / f"{phase}-summary.csv"
    result = run_experiment(
        source_manifest_path=args.source_manifest,
        output_path=output_path,
        manifest_output_path=manifest_output_path,
        processed_path=processed_path,
        phase=phase,
        backend=args.backend,
        condition_ids=args.condition_id,
        context_lengths=args.context_length,
        evidence_positions=args.evidence_position,
        repeats=args.repeats,
        fixture_seed=args.fixture_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
