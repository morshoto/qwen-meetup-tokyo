"""Run the matched exp_004 agent trajectory matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.agents import AgentHarness, AgentTask, TrajectoryControl  # noqa: E402
from llm_lab.analysis import aggregate_jsonl, write_summary_csv  # noqa: E402
from llm_lab.evaluation import (  # noqa: E402
    TrialResult,
    TrialStatus,
    JsonlResultWriter,
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
from llm_lab.runtimes import LlamaCppRuntime, Runtime, RuntimeConfig  # noqa: E402
from llm_lab.telemetry import capture_environment  # noqa: E402


EXPERIMENT_ID = "exp_004"
CONFIG_PATH = ROOT / "experiments/exp_004-agent_context_growth/config.yaml"
TASK_CATALOG = ROOT / "data/tasks/agent.v001.jsonl"
RuntimeFactory = Callable[[], Runtime]


class FixtureAgentRuntime:
    """Deterministic backend for smoke validation; never model evidence."""

    name = "fixture-agent"

    def __init__(self) -> None:
        self._model: ModelSpec | None = None
        self._config: RuntimeConfig | None = None
        self.closed = False

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        self._model = model
        self._config = config

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.metadata["agent_stage"] == "discovery":
            output = '{"action":"tool","name":"discover_fact","arguments":{}}'
        else:
            output = json.dumps(
                {
                    "action": "answer",
                    "value": str(request.metadata["fixture_expected_answer"]),
                }
            )
        return GenerationResponse(
            output_text=output,
            usage=TokenUsage(
                prompt_tokens=len(request.prompt.split()),
                completion_tokens=len(output.split()),
            ),
            timing=GenerationTiming(
                ttft_seconds=0.001,
                prefill_seconds=0.002,
                decode_seconds=0.001,
                total_seconds=0.004,
            ),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version="1.0",
                model_id=self._model.model_id if self._model else request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config={"purpose": "harness-smoke-only"},
            ),
        )

    def close(self) -> None:
        self.closed = True


def load_experiment_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the committed JSON-compatible YAML protocol."""

    path = _rooted(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"experiment config must be readable JSON-compatible YAML: {path}"
        ) from error
    if not isinstance(record, dict):
        raise ValueError("experiment config must be a mapping")
    return record


def load_source_manifest(path: Path) -> QuantizationManifest:
    """Load the resolved exp_003 manifest supplying selected artifacts.

    exp_003 records its resolved artifact selection as a run manifest under
    ``results/manifests`` and points back to the resolved exp_002 manifest for
    the canonical artifact provenance.  Accept both that recorded format and
    the canonical ``QuantizationManifest`` format used by the unit tests.
    """

    path = _rooted(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid resolved source manifest: {path}") from error

    try:
        if "variants" in record:
            manifest = QuantizationManifest.from_record(record)
        else:
            manifest = _manifest_from_exp003_run(path, record)
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ValueError(f"invalid resolved source manifest: {path}") from error
    if manifest.experiment_id != "exp_003":
        raise ValueError(
            "exp_004 requires a resolved exp_003 source manifest, got "
            f"{manifest.experiment_id!r}"
        )
    return manifest


def _manifest_from_exp003_run(
    path: Path,
    record: Mapping[str, Any],
) -> QuantizationManifest:
    """Adapt exp003's recorded run manifest to the shared source contract."""

    if record.get("experiment_id") != "exp_003":
        raise ValueError("source run manifest must describe exp_003")
    if record.get("backend") != "llama.cpp":
        raise ValueError("exp_003 source run must use the llama.cpp backend")
    source_reference = record.get("source_manifest")
    if not isinstance(source_reference, str) or not source_reference.strip():
        raise ValueError("exp_003 source run must name its resolved source manifest")
    source_path = _rooted(Path(source_reference))
    source_record = json.loads(source_path.read_text(encoding="utf-8"))
    source_manifest = QuantizationManifest.from_record(source_record)
    if source_manifest.experiment_id != "exp_002":
        raise ValueError(
            "exp_003 source run must point to an exp_002 resolved manifest"
        )
    declared_sha256 = record.get("source_manifest_sha256")
    actual_sha256 = _sha256_file(source_path)
    if declared_sha256 != actual_sha256:
        raise ValueError(
            "exp_003 source manifest digest mismatch: "
            f"manifest={declared_sha256!r}, actual={actual_sha256!r}"
        )

    runtime = record.get("runtime")
    model = record.get("model")
    if not isinstance(runtime, Mapping) or not isinstance(model, Mapping):
        raise ValueError("exp_003 source run must contain model and runtime records")
    variant_records = record.get("quantization_variants")
    if not isinstance(variant_records, list) or not variant_records:
        raise ValueError("exp_003 source run must contain quantization variants")
    variants: list[QuantizationVariant] = []
    for variant_record in variant_records:
        if not isinstance(variant_record, Mapping):
            raise ValueError("exp_003 quantization variants must be mappings")
        normalized = dict(variant_record)
        artifact = normalized.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("exp_003 quantization variant must contain an artifact")
        normalized_artifact = dict(artifact)
        normalized_artifact["artifact_uri"] = str(
            _artifact_path(source_path, str(normalized_artifact["artifact_uri"]))
        )
        normalized["artifact"] = normalized_artifact
        variants.append(QuantizationVariant.from_record(normalized))

    task_ids = tuple(str(value) for value in record["task_ids"])
    context_lengths = tuple(int(value) for value in record["context_lengths"])
    return QuantizationManifest(
        experiment_id="exp_003",
        model_id=str(model["id"]),
        model_revision=str(model["revision"]),
        tokenizer_id=str(model["tokenizer_id"]),
        tokenizer_revision=str(model["tokenizer_revision"]),
        runtime_name=str(runtime["name"]),
        runtime_version=str(runtime["version"]),
        prompt_id=str(record["prompt_id"]),
        task_ids=task_ids,
        context_lengths=context_lengths,
        sampling=dict(source_manifest.sampling),
        variants=tuple(variants),
        repeats=int(record.get("repeats", source_manifest.repeats)),
        context_length_semantics="input_tokens",
        context_overhead_tokens=source_manifest.context_overhead_tokens,
        runtime_options=dict(
            runtime.get("source_options", source_manifest.runtime_options)
        ),
    )


def load_tasks(path: Path = TASK_CATALOG) -> tuple[AgentTask, ...]:
    path = _rooted(path)
    tasks: list[AgentTask] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"agent task catalog is unreadable: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            task = AgentTask.from_record(record)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid agent task on line {line_number}") from error
        if task.task_id in seen:
            raise ValueError(f"duplicate agent task ID: {task.task_id}")
        seen.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError("agent task catalog must contain at least one task")
    return tuple(tasks)


def planned_conditions(
    phase: str,
    *,
    config: Mapping[str, Any] | None = None,
    trajectory_lengths: Iterable[int] | None = None,
    critical_positions: Iterable[float] | None = None,
) -> list[TrajectoryControl]:
    """Return the deterministic trajectory-length × position grid."""

    experiment_config = load_experiment_config() if config is None else config
    phase_config = _phase_config(experiment_config, phase)
    lengths = tuple(
        int(value)
        for value in (
            phase_config["lengths"]
            if trajectory_lengths is None
            else trajectory_lengths
        )
    )
    positions = tuple(
        float(value)
        for value in (
            phase_config["critical_positions"]
            if critical_positions is None
            else critical_positions
        )
    )
    if tuple(sorted(set(lengths))) != lengths or any(value < 1 for value in lengths):
        raise ValueError("trajectory lengths must be unique, increasing, and positive")
    if tuple(sorted(set(positions))) != positions or any(
        not 0.0 <= value <= 1.0 for value in positions
    ):
        raise ValueError(
            "critical positions must be unique, increasing, and between 0 and 1"
        )
    return [
        TrajectoryControl(length, position)
        for length in lengths
        for position in positions
    ]


def expected_trial_count(
    source_manifest: QuantizationManifest,
    tasks: Iterable[AgentTask],
    *,
    phase: str,
    config: Mapping[str, Any] | None = None,
    condition_ids: Iterable[str] | None = None,
    trajectory_lengths: Iterable[int] | None = None,
    critical_positions: Iterable[float] | None = None,
    repeats: int | None = None,
) -> int:
    variants = _select_variants(source_manifest, condition_ids, config)
    controls = planned_conditions(
        phase,
        config=config,
        trajectory_lengths=trajectory_lengths,
        critical_positions=critical_positions,
    )
    run_repeats = _select_repeats(phase, repeats, config)
    return len(variants) * len(tuple(tasks)) * len(controls) * run_repeats


def run_experiment(
    *,
    source_manifest_path: Path,
    output_path: Path,
    manifest_output_path: Path,
    processed_path: Path,
    phase: str,
    backend: str | None = None,
    condition_ids: Iterable[str] | None = None,
    trajectory_lengths: Iterable[int] | None = None,
    critical_positions: Iterable[float] | None = None,
    repeats: int | None = None,
    fixture_seed: int | None = None,
    runtime_factory: RuntimeFactory | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run or resume a selected matrix and write its resolved manifest."""

    source_manifest_path = _rooted(source_manifest_path)
    output_path = _rooted(output_path)
    manifest_output_path = _rooted(manifest_output_path)
    processed_path = _rooted(processed_path)
    config = load_experiment_config(CONFIG_PATH if config_path is None else config_path)
    phase_config = _phase_config(config, phase)
    selected_backend = str(phase_config["backend"] if backend is None else backend)
    if selected_backend not in {"fixture", "llama.cpp"}:
        raise ValueError(f"unsupported backend: {selected_backend!r}")
    source_manifest = load_source_manifest(source_manifest_path)
    variants = _select_variants(source_manifest, condition_ids, config)
    tasks = load_tasks(
        ROOT / str(_section(config, "experiment")["task_catalog"])
    )
    controls = planned_conditions(
        phase,
        config=config,
        trajectory_lengths=trajectory_lengths,
        critical_positions=critical_positions,
    )
    run_repeats = _select_repeats(phase, repeats, config)
    seed = (
        int(_section(config, "experiment")["fixture_seed"])
        if fixture_seed is None
        else int(fixture_seed)
    )
    sampling = _sampling_config(_section(config, "sampling"))
    run_fingerprint = _run_fingerprint(
        config=config,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        phase=phase,
        backend=selected_backend,
        variants=variants,
        controls=controls,
        repeats=run_repeats,
        fixture_seed=seed,
    )
    expected_ids = {
        make_trial_id(
            EXPERIMENT_ID,
            task.task_id,
            condition_id=f"{variant.condition_id}:{control.condition_id}",
            repeat_index=repeat_index,
        )
        for variant in variants
        for control in controls
        for task in tasks
        for repeat_index in range(1, run_repeats + 1)
    }
    existing = load_trial_results(output_path)
    _validate_existing(existing, expected_ids, run_fingerprint)
    existing_ids = {result.trial_id for result in existing}
    writer = JsonlResultWriter(output_path)
    observed_runtime: dict[str, Any] = {
        "name": selected_backend,
        "version": None,
        "options": {},
    }
    if existing:
        observed_runtime = {
            "name": existing[0].runtime.get("name"),
            "version": existing[0].runtime.get("version"),
            "options": dict(existing[0].runtime.get("config", {})),
        }
    try:
        for variant in variants:
            missing = [
                (control, task, repeat_index)
                for control in controls
                for task in tasks
                for repeat_index in range(1, run_repeats + 1)
                if make_trial_id(
                    EXPERIMENT_ID,
                    task.task_id,
                    condition_id=f"{variant.condition_id}:{control.condition_id}",
                    repeat_index=repeat_index,
                )
                not in existing_ids
            ]
            if not missing:
                continue
            runtime, runtime_config = _make_runtime(
                selected_backend,
                source_manifest,
                variant,
                source_manifest_path,
                runtime_factory=runtime_factory,
                tasks=tasks,
            )
            observed_runtime = {
                "name": runtime.name,
                "version": runtime_config.version,
                "options": dict(runtime_config.options),
            }
            model = _model_for_variant(source_manifest, variant)
            runtime.load(model, runtime_config)
            try:
                harness = AgentHarness(
                    runtime=runtime,
                    model=model,
                    max_action_attempts=int(
                        _section(config, "runtime")["max_action_attempts"]
                    ),
                )
                for control, task, repeat_index in missing:
                    context_instance_id = _context_instance_id(
                        task,
                        control,
                        fixture_seed=seed,
                    )
                    trial_id = make_trial_id(
                        EXPERIMENT_ID,
                        task.task_id,
                        condition_id=f"{variant.condition_id}:{control.condition_id}",
                        repeat_index=repeat_index,
                    )
                    run_metadata: dict[str, Any] = {
                        "variant_condition_id": variant.condition_id,
                        "context_instance_id": context_instance_id,
                        "environment_fingerprint": task.environment().fingerprint,
                        "run_fingerprint": run_fingerprint,
                        "fixture_seed": seed,
                    }
                    if selected_backend == "fixture":
                        run_metadata["fixture_expected_answer"] = task.expected_answer
                    agent_run = harness.run(
                        task,
                        control,
                        sampling=sampling,
                        metadata=run_metadata,
                    )
                    result = _trial_result(
                        trial_id=trial_id,
                        task=task,
                        variant=variant,
                        control=control,
                        repeat_index=repeat_index,
                        run=agent_run,
                        model=model,
                        runtime=runtime,
                        backend=selected_backend,
                        context_instance_id=context_instance_id,
                        run_fingerprint=run_fingerprint,
                    )
                    writer.append(result)
                    existing_ids.add(trial_id)
            finally:
                runtime.close()
    finally:
        all_results = load_trial_results(output_path)
    summaries = aggregate_jsonl(output_path)
    write_summary_csv(processed_path, summaries)
    manifest = _build_manifest(
        config=config,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=_sha256_file(source_manifest_path),
        phase=phase,
        backend=selected_backend,
        variants=variants,
        controls=controls,
        tasks=tasks,
        repeats=run_repeats,
        fixture_seed=seed,
        run_fingerprint=run_fingerprint,
        existing_n=len(existing),
        results=all_results,
        effective_runtime=observed_runtime,
        raw_result_path=output_path,
    )
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "expected_trial_n": len(expected_ids),
        "actual_trial_n": len(all_results),
        "skipped_trial_n": len(existing),
        "run_fingerprint": run_fingerprint,
        "manifest_path": str(manifest_output_path),
        "processed_path": str(processed_path),
    }


def _trial_result(
    *,
    trial_id: str,
    task: AgentTask,
    variant: QuantizationVariant,
    control: TrajectoryControl,
    repeat_index: int,
    run: Any,
    model: ModelSpec,
    runtime: Runtime,
    backend: str,
    context_instance_id: str,
    run_fingerprint: str,
) -> TrialResult:
    last_response = run.last_response
    correct = (
        run.final_answer is not None
        and " ".join(run.final_answer.strip().lower().split())
        == " ".join(task.expected_answer.strip().lower().split())
    )
    status: TrialStatus
    if run.status == "completed":
        status = TrialStatus.COMPLETED
    elif run.error and run.error.get("type") not in {"ActionParseError", None}:
        status = TrialStatus.RUNTIME_ERROR
    else:
        status = TrialStatus.INVALID_OUTPUT
    runtime_record = {
        "name": runtime.name,
        "version": last_response.runtime.runtime_version
        if last_response is not None and last_response.runtime is not None
        else None,
        "config": dict(last_response.runtime.config)
        if last_response is not None and last_response.runtime is not None
        else {},
    }
    input_record = {
        "task_type": task.task_type,
        "condition_id": f"{variant.condition_id}:{control.condition_id}",
        "variant_condition_id": variant.condition_id,
        "repeat_index": repeat_index,
        "context_instance_id": context_instance_id,
        "environment_fingerprint": task.environment().fingerprint,
        "run_fingerprint": run_fingerprint,
        "trajectory_control": control.to_record(),
        "trajectory": run.trajectory.to_records(),
        "trajectory_length": control.trajectory_length,
        "requested_critical_position": control.critical_position,
        "actual_critical_position": control.actual_critical_position,
        "trajectory_context_tokens": run.metrics.get("max_input_tokens"),
        "total_input_tokens": run.metrics.get("total_input_tokens"),
        "fixture_only": backend == "fixture",
        "metrics": dict(run.metrics),
    }
    timing = {
        "total_s": sum(
            response.timing.total_seconds or 0.0 for response in run.responses
        ),
        "ttft_s": sum(
            response.timing.ttft_seconds or 0.0 for response in run.responses
        ),
    }
    return TrialResult(
        trial_id=trial_id,
        experiment_id=EXPERIMENT_ID,
        task_id=task.task_id,
        status=status,
        model=_model_record(model, variant),
        runtime=runtime_record,
        input=input_record,
        generation={
            "output_text": last_response.output_text if last_response else None,
            "final_answer": run.final_answer,
            "output_tokens": (
                last_response.usage.completion_tokens if last_response else None
            ),
            "responses": [
                {
                    "output_text": response.output_text,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "timing": {
                        "ttft_s": response.timing.ttft_seconds,
                        "total_s": response.timing.total_seconds,
                    },
                }
                for response in run.responses
            ],
        },
        score={
            "correct": correct if run.final_answer is not None else None,
            "value": float(correct) if run.final_answer is not None else None,
            "scorer": "agent_exact.v1",
            "details": {
                "failure_category": run.metrics.get("failure_category"),
                "critical_fact_reused": run.metrics.get("critical_fact_reused"),
            },
        },
        timing=timing,
        environment={
            **capture_environment(ROOT),
            "backend": backend,
            "purpose": "harness_smoke_only" if backend == "fixture" else "measurement",
        },
        error=run.error,
    )


def _make_runtime(
    backend: str,
    source_manifest: QuantizationManifest,
    variant: QuantizationVariant,
    source_manifest_path: Path,
    *,
    runtime_factory: RuntimeFactory | None,
    tasks: Iterable[AgentTask],
) -> tuple[Runtime, RuntimeConfig]:
    del tasks
    if runtime_factory is not None:
        runtime = runtime_factory()
        return runtime, RuntimeConfig(
            name=runtime.name,
            version="fixture" if backend == "fixture" else source_manifest.runtime_version,
            options={"purpose": "harness-smoke-only"}
            if backend == "fixture"
            else {"variant_condition_id": variant.condition_id},
        )
    if backend == "fixture":
        return FixtureAgentRuntime(), RuntimeConfig(
            name="fixture-agent",
            version="1.0",
            options={"purpose": "harness-smoke-only"},
        )
    artifact_path = _artifact_path(source_manifest_path, variant.artifact.artifact_uri)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"selected artifact for {variant.condition_id} is unavailable: {artifact_path}"
        )
    actual_sha256 = _sha256_file(artifact_path)
    if actual_sha256 != variant.artifact.artifact_sha256:
        raise ValueError(
            f"artifact digest mismatch for {variant.condition_id}: "
            f"manifest={variant.artifact.artifact_sha256}, actual={actual_sha256}"
        )
    actual_size = artifact_path.stat().st_size
    if actual_size != variant.artifact.artifact_size_bytes:
        raise ValueError(
            f"artifact size mismatch for {variant.condition_id}: "
            f"manifest={variant.artifact.artifact_size_bytes}, actual={actual_size}"
        )
    options = dict(source_manifest.runtime_options)
    options["model_path"] = str(artifact_path)
    return LlamaCppRuntime(), RuntimeConfig(
        name=source_manifest.runtime_name,
        version=source_manifest.runtime_version,
        options=options,
    )


def _build_manifest(
    *,
    config: Mapping[str, Any],
    source_manifest: QuantizationManifest,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    phase: str,
    backend: str,
    variants: Iterable[QuantizationVariant],
    controls: Iterable[TrajectoryControl],
    tasks: Iterable[AgentTask],
    repeats: int,
    fixture_seed: int,
    run_fingerprint: str,
    existing_n: int,
    results: Iterable[TrialResult],
    effective_runtime: Mapping[str, Any],
    raw_result_path: Path,
) -> dict[str, Any]:
    variant_list = list(variants)
    result_list = list(results)
    controls_list = list(controls)
    task_list = list(tasks)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "backend": backend,
        "fixture_only": backend == "fixture",
        "protocol": {
            "config": str(CONFIG_PATH.relative_to(ROOT)),
            "task_catalog": str(_section(config, "experiment")["task_catalog"]),
            "task_ids": [task.task_id for task in task_list],
            "task_types": sorted({task.task_type for task in task_list}),
            "fixture_seed": fixture_seed,
            "repeats": repeats,
            "trajectory_lengths": sorted(
                {control.trajectory_length for control in controls_list}
            ),
            "critical_positions": sorted(
                {control.critical_position for control in controls_list}
            ),
            "condition_ids": [control.condition_id for control in controls_list],
        },
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": source_manifest_sha256,
            "experiment_id": source_manifest.experiment_id,
            "model": {
                "id": source_manifest.model_id,
                "revision": source_manifest.model_revision,
                "tokenizer_id": source_manifest.tokenizer_id,
                "tokenizer_revision": source_manifest.tokenizer_revision,
            },
            "runtime": {
                "name": source_manifest.runtime_name,
                "version": source_manifest.runtime_version,
                "options": dict(source_manifest.runtime_options),
            },
            "variants": [variant.to_record() for variant in variant_list],
        },
        "effective_runtime": dict(effective_runtime),
        "run_fingerprint": run_fingerprint,
        "expected_trial_n": len(variant_list)
        * len(task_list)
        * len(controls_list)
        * repeats,
        "actual_trial_n": len(result_list),
        "skipped_trial_n": existing_n,
        "status_counts": _status_counts(result_list),
        "raw_result_sha256": _sha256_file(raw_result_path),
    }


def _status_counts(results: Iterable[TrialResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status.value] = counts.get(result.status.value, 0) + 1
    return counts


def _select_variants(
    source_manifest: QuantizationManifest,
    condition_ids: Iterable[str] | None,
    config: Mapping[str, Any] | None,
) -> tuple[QuantizationVariant, ...]:
    requested = tuple(
        str(value)
        for value in (
            _section(_load_or_empty_config(config), "quantization")["variants"]
            if condition_ids is None
            else condition_ids
        )
    )
    by_id = {variant.condition_id: variant for variant in source_manifest.variants}
    unknown = set(requested) - set(by_id)
    if unknown:
        raise ValueError(f"selected variants are absent from source manifest: {sorted(unknown)}")
    if not requested:
        raise ValueError("at least one quantization variant must be selected")
    return tuple(by_id[value] for value in requested)


def _select_repeats(
    phase: str,
    repeats: int | None,
    config: Mapping[str, Any] | None,
) -> int:
    selected = repeats
    if selected is None:
        selected = int(_phase_config(_load_or_empty_config(config), phase)["repeats"])
    if selected < 1:
        raise ValueError("repeats must be positive")
    return selected


def _phase_config(config: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    phases = _section(config, "phases")
    value = phases.get(phase)
    if not isinstance(value, Mapping):
        raise ValueError(f"unsupported experiment phase: {phase!r}")
    return value


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"experiment config section {name!r} must be a mapping")
    return value


def _load_or_empty_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return load_experiment_config() if config is None else config


def _sampling_config(record: Mapping[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        max_new_tokens=int(record["max_new_tokens"]),
        temperature=float(record["temperature"]),
        top_p=float(record.get("top_p", 1.0)),
        top_k=(None if record.get("top_k") is None else int(record["top_k"])),
        seed=(
            None
            if record.get("generation_seed") is None
            else int(record["generation_seed"])
        ),
    )


def _model_for_variant(
    source_manifest: QuantizationManifest,
    variant: QuantizationVariant,
) -> ModelSpec:
    return ModelSpec(
        model_id=source_manifest.model_id,
        revision=source_manifest.model_revision,
        tokenizer_id=source_manifest.tokenizer_id,
        tokenizer_revision=source_manifest.tokenizer_revision,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "quantization_condition_id": variant.condition_id,
            "quantization_type": variant.quantization_type,
        },
    )


def _model_record(
    model: ModelSpec,
    variant: QuantizationVariant,
) -> dict[str, Any]:
    return {
        "id": model.model_id,
        "revision": model.revision,
        "tokenizer_id": model.tokenizer_id,
        "tokenizer_revision": model.tokenizer_revision,
        "quantization_condition_id": variant.condition_id,
        "quantization_type": variant.quantization_type,
        "artifact": variant.artifact.to_record(),
    }


def _context_instance_id(
    task: AgentTask,
    control: TrajectoryControl,
    *,
    fixture_seed: int,
) -> str:
    payload = {
        "task_id": task.task_id,
        "fixture_seed": fixture_seed,
        "control": control.to_record(),
        "environment_fingerprint": task.environment().fingerprint,
    }
    return _sha256_bytes(_canonical_json(payload))


def _run_fingerprint(
    *,
    config: Mapping[str, Any],
    source_manifest_path: Path,
    source_manifest: QuantizationManifest,
    phase: str,
    backend: str,
    variants: Iterable[QuantizationVariant],
    controls: Iterable[TrajectoryControl],
    repeats: int,
    fixture_seed: int,
) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest": source_manifest.to_record(),
        "phase": phase,
        "backend": backend,
        "variants": [variant.condition_id for variant in variants],
        "controls": [control.to_record() for control in controls],
        "repeats": repeats,
        "fixture_seed": fixture_seed,
    }
    return _sha256_bytes(_canonical_json(payload))


def _validate_existing(
    existing: Iterable[TrialResult],
    expected_ids: set[str],
    run_fingerprint: str,
) -> None:
    for result in existing:
        if result.trial_id not in expected_ids:
            raise ValueError(f"existing trial is outside selected run: {result.trial_id}")
        existing_fingerprint = result.input.get("run_fingerprint")
        if existing_fingerprint != run_fingerprint:
            raise ValueError("existing raw results do not match the selected run")


def _artifact_path(manifest_path: Path, artifact_uri: str) -> Path:
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"unsupported local artifact URI: {artifact_uri}")
    path = Path(artifact_uri)
    return path if path.is_absolute() else manifest_path.parent / path


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return "unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--phase", default="main")
    parser.add_argument("--backend")
    parser.add_argument("--condition-id", action="append", dest="condition_ids")
    parser.add_argument("--trajectory-length", type=int, action="append")
    parser.add_argument("--critical-position", type=float, action="append")
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    run_experiment(
        source_manifest_path=args.source_manifest,
        output_path=args.output,
        manifest_output_path=args.manifest,
        processed_path=args.processed,
        phase=args.phase,
        backend=args.backend,
        condition_ids=args.condition_ids,
        trajectory_lengths=args.trajectory_length,
        critical_positions=args.critical_position,
        repeats=args.repeats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
