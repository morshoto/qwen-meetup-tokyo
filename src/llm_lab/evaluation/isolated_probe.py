"""Process-isolated probe execution with fail-visible wall-clock timeouts."""

from __future__ import annotations

import importlib
import hashlib
import multiprocessing
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Mapping


@dataclass(frozen=True)
class ProbeOutcome:
    """Result returned by a bounded child-process probe."""

    value: Mapping[str, Any] | None
    timed_out: bool
    exit_code: int | None
    peak_memory_bytes: int | None
    memory_measurement: str | None
    termination_reason: str | None
    error: Mapping[str, Any] | None = None
    alive: bool = False


def run_isolated_probe(
    worker: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float,
    memory_sample_interval: float = 0.25,
) -> ProbeOutcome:
    """Run an importable worker in a fresh process and enforce a hard timeout.

    ``worker`` is a ``module:function`` path rather than a live callable. This
    keeps the boundary serializable under macOS ``spawn`` and prevents a live
    llama.cpp client from being forked into a process that the parent may later
    kill. The parent samples child RSS and owns termination decisions.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if memory_sample_interval <= 0:
        raise ValueError("memory_sample_interval must be positive")

    context = multiprocessing.get_context("spawn")
    messages = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker_entry,
        args=(worker, dict(payload), messages),
        name="llm-lab-feasibility-probe",
    )
    process.start()
    started = time.monotonic()
    peak_memory: int | None = None
    memory_measurement: str | None = None
    timed_out = False
    termination_reason: str | None = None

    while process.is_alive():
        sample, measurement = _read_child_rss(process.pid)
        if sample is not None:
            peak_memory = sample if peak_memory is None else max(peak_memory, sample)
        if measurement is not None:
            memory_measurement = measurement
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            termination_reason = "timeout"
            process.terminate()
            process.join(timeout=min(5.0, max(0.1, timeout_seconds)))
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            break
        process.join(timeout=min(memory_sample_interval, remaining))

    process.join(timeout=0.1)
    sample, measurement = _read_child_rss(process.pid)
    if sample is not None:
        peak_memory = sample if peak_memory is None else max(peak_memory, sample)
    if measurement is not None:
        memory_measurement = measurement

    message: Mapping[str, Any] | None = None
    try:
        message = messages.get(timeout=0.2)
    except Empty:
        message = None
    finally:
        messages.close()
        messages.join_thread()

    if timed_out:
        return ProbeOutcome(
            value=None,
            timed_out=True,
            exit_code=process.exitcode,
            peak_memory_bytes=peak_memory,
            memory_measurement=memory_measurement,
            termination_reason=termination_reason,
            error={"type": "TimeoutError", "message": "probe exceeded timeout"},
            alive=process.is_alive(),
        )

    if not isinstance(message, Mapping):
        return ProbeOutcome(
            value=None,
            timed_out=False,
            exit_code=process.exitcode,
            peak_memory_bytes=peak_memory,
            memory_measurement=memory_measurement,
            termination_reason="missing_worker_result",
            error={
                "type": "ProbeWorkerError",
                "message": "worker exited without a result",
            },
            alive=process.is_alive(),
        )

    if message.get("status") == "error":
        error_value = message.get("error")
        error = error_value if isinstance(error_value, Mapping) else None
        return ProbeOutcome(
            value=None,
            timed_out=False,
            exit_code=process.exitcode,
            peak_memory_bytes=peak_memory,
            memory_measurement=memory_measurement,
            termination_reason="worker_error",
            error=error,
            alive=process.is_alive(),
        )

    value = message.get("value")
    if not isinstance(value, Mapping):
        return ProbeOutcome(
            value=None,
            timed_out=False,
            exit_code=process.exitcode,
            peak_memory_bytes=peak_memory,
            memory_measurement=memory_measurement,
            termination_reason="invalid_worker_result",
            error={
                "type": "ProbeWorkerError",
                "message": "worker result must be a mapping",
            },
            alive=process.is_alive(),
        )
    return ProbeOutcome(
        value=dict(value),
        timed_out=False,
        exit_code=process.exitcode,
        peak_memory_bytes=peak_memory,
        memory_measurement=memory_measurement,
        termination_reason=None,
        alive=process.is_alive(),
    )


def _worker_entry(
    worker_path: str,
    payload: Mapping[str, Any],
    messages: Any,
) -> None:
    try:
        module_name, separator, function_name = worker_path.partition(":")
        if not separator or not module_name or not function_name:
            raise ValueError("worker must use module:function notation")
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        value = function(payload)
        if not isinstance(value, Mapping):
            raise TypeError("worker must return a mapping")
        messages.put({"status": "ok", "value": dict(value)})
    except BaseException as error:  # pragma: no cover - exercised in child
        messages.put(
            {
                "status": "error",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )


def _read_child_rss(pid: int | None) -> tuple[int | None, str | None]:
    if pid is None:
        return None, None
    try:
        import psutil
    except ImportError:
        return None, None
    try:
        return int(psutil.Process(pid).memory_info().rss), "psutil.child_rss_sampled"
    except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
        return None, None


def run_exp001_qwen_probe(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one exp_001 Q8 task inside the isolated child process.

    The payload contains only JSON-compatible values. Loading the model and
    constructing the inference-tokenized context in the child ensures that a
    parent timeout never leaves a live llama.cpp client behind.
    """

    from llm_lab.context import Evidence, SyntheticContextGenerator
    from llm_lab.datasets import TaskCatalog
    from llm_lab.evaluation import (
        CalibratedAnswerScorer,
        EvaluationRunner,
        EvaluationTask,
    )
    from llm_lab.generation import SamplingConfig
    from llm_lab.models import qwen38_model_spec
    from llm_lab.runtimes import RuntimeConfig
    from llm_lab.runtimes.llama_cpp import LlamaCppRuntime

    definition_payload = payload.get("task_definition")
    if not isinstance(definition_payload, Mapping):
        raise ValueError("exp_001 probe requires task_definition")
    catalog = TaskCatalog.from_records([definition_payload])
    definition = catalog.tasks[0]
    condition = payload.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("exp_001 probe requires condition")
    target_tokens = int(condition["target_context_tokens"])
    evidence_position = float(condition["evidence_position"])
    namespace = str(condition.get("namespace", "feasibility"))
    fixture_seed = int(payload["fixture_seed"])
    task_seed = fixture_seed + int(definition.metadata["seed"])

    model_payload = payload.get("model")
    if not isinstance(model_payload, Mapping):
        raise ValueError("exp_001 probe requires model identity")
    model = qwen38_model_spec(
        revision=model_payload.get("revision"),
        tokenizer_revision=model_payload.get("tokenizer_revision"),
    )
    runtime_options = payload.get("runtime_options")
    if not isinstance(runtime_options, Mapping):
        raise ValueError("exp_001 probe requires runtime options")
    runtime = LlamaCppRuntime()
    runtime.load(
        model,
        RuntimeConfig(
            name="llama.cpp",
            version=payload.get("runtime_version"),
            options=dict(runtime_options),
        ),
    )
    try:
        tokenizer = runtime.get_tokenizer()
        generated = SyntheticContextGenerator(tokenizer=tokenizer).generate(
            [
                Evidence(id=str(item["id"]), text=str(item["text"]))
                for item in definition.evidence
            ],
            target_tokens=target_tokens,
            evidence_position=evidence_position,
            seed=task_seed,
        )
        spans = [
            {
                "id": span.id,
                "token_start": span.token_start,
                "token_end": span.token_end,
                "requested_position": span.requested_position,
                "actual_position": span.actual_position,
            }
            for span in generated.evidence
        ]
        metadata = {
            "corpus_id": "corpus.synthetic.000001",
            "fixture_seed": fixture_seed,
            "task_seed": task_seed,
            "target_context_tokens": target_tokens,
            "actual_context_tokens": generated.token_count,
            "requested_evidence_position": evidence_position,
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
            "target_unit": "inference-tokenizer-tokens",
            "context_instance_id": (
                f"{definition.task_id}:{namespace}:ctx{target_tokens:06d}:"
                f"p{int(evidence_position * 100):03d}:seed{task_seed}"
            ),
            "context_sha256": hashlib.sha256(
                generated.text.encode("utf-8")
            ).hexdigest(),
        }
        task = EvaluationTask.from_definition(
            definition,
            context=generated.text,
            metadata=metadata,
        )
        sampling_payload = payload.get("sampling")
        if not isinstance(sampling_payload, Mapping):
            raise ValueError("exp_001 probe requires sampling")
        sampling = SamplingConfig(
            max_new_tokens=int(sampling_payload["max_new_tokens"]),
            temperature=float(sampling_payload["temperature"]),
            top_p=float(sampling_payload["top_p"]),
            top_k=(
                None
                if sampling_payload.get("top_k") is None
                else int(sampling_payload["top_k"])
            ),
            seed=sampling_payload.get("seed"),
        )
        result = EvaluationRunner(
            runtime=runtime,
            model=model,
            scorer=CalibratedAnswerScorer(),
            experiment_id="exp_001",
        ).run(
            [task],
            repeats=1,
            condition_id=(
                f"{namespace}:ctx{target_tokens:06d}:"
                f"p{int(evidence_position * 100):03d}"
            ),
            sampling=sampling,
        )[0]
        return result.to_record()
    finally:
        runtime.close()
