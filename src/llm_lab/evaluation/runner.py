"""Repeated-trial execution with fail-visible result recording."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from llm_lab.generation import GenerationRequest, GenerationResponse, SamplingConfig
from llm_lab.models import ModelSpec
from llm_lab.runtimes import Runtime
from llm_lab.telemetry import TelemetryCollector, TelemetryRecord

from .contracts import ScoreResult, Scorer, Task
from .results import TrialResult, TrialStatus, make_trial_id
from .storage import JsonlResultWriter


class EvaluationRunner:
    def __init__(
        self,
        *,
        runtime: Runtime,
        model: ModelSpec,
        scorer: Scorer,
        experiment_id: str,
        output_path: str | None = None,
        writer: JsonlResultWriter | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        if writer is not None and output_path is not None:
            raise ValueError("provide writer or output_path, not both")
        self.runtime = runtime
        self.model = model
        self.scorer = scorer
        self.experiment_id = experiment_id
        self.writer = writer or (JsonlResultWriter(output_path) if output_path else None)
        self.telemetry = telemetry or TelemetryCollector()

    def run(
        self,
        tasks: Iterable[Task],
        *,
        repeats: int = 1,
        repeat_indices: Iterable[int] | None = None,
        condition_id: str = "default",
        sampling: SamplingConfig | None = None,
    ) -> list[TrialResult]:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if repeat_indices is None:
            selected_repeat_indices = tuple(range(1, repeats + 1))
        else:
            selected_repeat_indices = tuple(repeat_indices)
            if not selected_repeat_indices:
                raise ValueError("repeat_indices must not be empty")
            if any(index < 1 for index in selected_repeat_indices):
                raise ValueError("repeat_indices must be positive")
        sampling_config = sampling or SamplingConfig()
        results: list[TrialResult] = []
        for task in tasks:
            for repeat_index in selected_repeat_indices:
                result = self._run_one(
                    task,
                    repeat_index=repeat_index,
                    condition_id=condition_id,
                    sampling=sampling_config,
                )
                if self.writer is not None:
                    self.writer.append(result)
                results.append(result)
        return results

    def _run_one(
        self,
        task: Task,
        *,
        repeat_index: int,
        condition_id: str,
        sampling: SamplingConfig,
    ) -> TrialResult:
        trial_id = make_trial_id(
            self.experiment_id,
            task.task_id,
            condition_id=condition_id,
            repeat_index=repeat_index,
        )
        handle = self.telemetry.start()
        request: GenerationRequest | None = None
        response: GenerationResponse | None = None
        try:
            request = task.build_request(self.model, sampling)
        except Exception as error:
            telemetry = self.telemetry.finish(handle, None)
            return self._record(
                trial_id,
                task,
                condition_id,
                repeat_index,
                status=TrialStatus.INVALID_INPUT,
                telemetry=telemetry,
                error=error,
            )

        try:
            response = self.runtime.generate(request)
        except Exception as error:
            telemetry = self.telemetry.finish(handle, None)
            return self._record(
                trial_id,
                task,
                condition_id,
                repeat_index,
                status=TrialStatus.RUNTIME_ERROR,
                request=request,
                telemetry=telemetry,
                error=error,
            )

        telemetry = self.telemetry.finish(handle, response)
        try:
            score = self.scorer.score(task, response)
        except Exception as error:
            return self._record(
                trial_id,
                task,
                condition_id,
                repeat_index,
                status=TrialStatus.SCORER_ERROR,
                request=request,
                response=response,
                telemetry=telemetry,
                error=error,
            )

        status = TrialStatus.INVALID_OUTPUT if score.correct is None else TrialStatus.COMPLETED
        return self._record(
            trial_id,
            task,
            condition_id,
            repeat_index,
            status=status,
            request=request,
            response=response,
            score=_score_record(score),
            telemetry=telemetry,
        )

    def _record(
        self,
        trial_id: str,
        task: Task,
        condition_id: str,
        repeat_index: int,
        *,
        status: TrialStatus,
        telemetry: TelemetryRecord,
        request: GenerationRequest | None = None,
        response: GenerationResponse | None = None,
        score: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> TrialResult:
        timing = telemetry.to_dict()
        if response is not None:
            timing.update(
                {
                    "prefill_s": response.timing.prefill_seconds,
                    "decode_s": response.timing.decode_seconds,
                }
            )
            if response.timing.post_first_chunk_seconds is not None:
                timing.update(
                    {
                        "stream_ttft_s": response.timing.ttft_seconds,
                        "post_first_chunk_s": response.timing.post_first_chunk_seconds,
                        "prompt_throughput_proxy_tok_s": _rate(
                            response.usage.prompt_tokens,
                            response.timing.ttft_seconds,
                        ),
                        "post_first_chunk_output_tok_s": _rate(
                            response.usage.completion_tokens,
                            response.timing.post_first_chunk_seconds,
                        ),
                    }
                )
        runtime = _runtime_record(self.runtime, response)
        input_metadata = dict(request.metadata) if request is not None else {}
        input_metadata.update(
            {
                "task_type": task.task_type,
                "condition_id": condition_id,
                "repeat_index": repeat_index,
                "prompt_tokens": response.usage.prompt_tokens
                if response is not None
                else None,
            }
        )
        score_record = dict(score or {})
        score_record.setdefault("scorer", self.scorer.name)
        return TrialResult(
            trial_id=trial_id,
            experiment_id=self.experiment_id,
            task_id=task.task_id,
            status=status,
            model=_model_record(self.model),
            runtime=runtime,
            input=input_metadata,
            generation=_generation_record(response),
            score=score_record,
            timing=timing,
            memory={
                "peak_bytes": telemetry.peak_memory_bytes,
                "measurement": telemetry.memory_measurement,
            },
            environment=dict(telemetry.environment),
            error=_error_record(error),
        )


def _model_record(model: ModelSpec) -> dict[str, Any]:
    return {
        "id": model.model_id,
        "revision": model.revision,
        "tokenizer_id": model.tokenizer_id,
        "tokenizer_revision": model.tokenizer_revision,
        "capabilities": asdict(model.capabilities),
    }


def _score_record(score: ScoreResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "correct": score.correct,
        "value": score.value,
        "scorer": score.scorer,
        "details": dict(score.details),
    }
    for field_name in ("exact_correct", "answer_bearing_correct", "format_valid"):
        metric = getattr(score, field_name)
        if metric is not None:
            record[field_name] = metric
    return record


def _runtime_record(runtime: Runtime, response: GenerationResponse | None) -> dict[str, Any]:
    if response is not None and response.runtime is not None:
        return {
            "name": response.runtime.runtime_name,
            "version": response.runtime.runtime_version,
            "model_id": response.runtime.model_id,
            "tokenizer_id": response.runtime.tokenizer_id,
            "config": dict(response.runtime.config),
        }
    return {"name": runtime.name, "version": None, "config": {}}


def _generation_record(response: GenerationResponse | None) -> dict[str, Any]:
    if response is None:
        return {}
    return {
        "output_text": response.output_text,
        "output_tokens": response.usage.completion_tokens,
        "finish_reason": response.finish_reason,
    }


def _error_record(error: Exception | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {"type": type(error).__name__, "message": str(error)}


def _rate(tokens: int | None, seconds: float | None) -> float | None:
    if tokens is None or seconds is None or seconds <= 0:
        return None
    return tokens / seconds
