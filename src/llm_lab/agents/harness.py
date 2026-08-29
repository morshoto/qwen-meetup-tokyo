"""Minimal multi-turn agent harness for controlled reliability experiments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from llm_lab.generation import GenerationRequest, GenerationResponse, SamplingConfig
from llm_lab.models import ModelSpec
from llm_lab.runtimes import Runtime

from .trajectory import (
    AgentAction,
    AgentTrajectory,
    DeterministicToolEnvironment,
    ToolDefinition,
    parse_action,
)


@dataclass(frozen=True)
class AgentTask:
    """One deterministic state-tracking task definition."""

    task_id: str
    objective: str
    expected_answer: str
    critical_observation_id: str
    critical_observation: str
    distractor_outputs: tuple[str, ...]
    task_type: str = "agent_state_tracking"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "task_id",
            "objective",
            "expected_answer",
            "critical_observation_id",
            "critical_observation",
            "task_type",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not self.distractor_outputs or any(
            not output.strip() for output in self.distractor_outputs
        ):
            raise ValueError("distractor_outputs must contain non-empty outputs")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "AgentTask":
        expected = record.get("expected")
        critical = record.get("critical_observation")
        if not isinstance(expected, Mapping) or expected.get("type") != "exact":
            raise ValueError("agent task expected must be an exact answer mapping")
        if not isinstance(critical, Mapping):
            raise ValueError("agent task critical_observation must be a mapping")
        distractors = record.get("distractor_outputs")
        if not isinstance(distractors, list):
            raise ValueError("agent task distractor_outputs must be a list")
        return cls(
            task_id=str(record["id"]),
            objective=str(record["objective"]),
            expected_answer=str(expected["value"]),
            critical_observation_id=str(critical["id"]),
            critical_observation=str(critical["content"]),
            distractor_outputs=tuple(str(value) for value in distractors),
            task_type=str(record.get("type", "agent_state_tracking")),
            metadata=dict(record.get("metadata", {})),
        )

    def environment(self) -> DeterministicToolEnvironment:
        return DeterministicToolEnvironment(
            critical_observation=self.critical_observation,
            distractor_outputs=self.distractor_outputs,
        )


@dataclass(frozen=True)
class TrajectoryControl:
    """Controls the number and relative position of tool observations."""

    trajectory_length: int
    critical_position: float

    def __post_init__(self) -> None:
        if self.trajectory_length < 1:
            raise ValueError("trajectory_length must be positive")
        if not 0.0 <= self.critical_position <= 1.0:
            raise ValueError("critical_position must be between 0 and 1")

    @property
    def pre_discovery_steps(self) -> int:
        return round((self.trajectory_length - 1) * self.critical_position)

    @property
    def post_discovery_steps(self) -> int:
        return self.trajectory_length - 1 - self.pre_discovery_steps

    @property
    def actual_critical_position(self) -> float:
        return self.pre_discovery_steps / self.trajectory_length

    @property
    def condition_id(self) -> str:
        return (
            f"traj{self.trajectory_length:04d}:"
            f"p{int(self.critical_position * 100):03d}"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "trajectory_length": self.trajectory_length,
            "critical_position": self.critical_position,
            "pre_discovery_steps": self.pre_discovery_steps,
            "post_discovery_steps": self.post_discovery_steps,
            "actual_critical_position": self.actual_critical_position,
            "condition_id": self.condition_id,
        }


@dataclass(frozen=True)
class AgentRun:
    """Outcome of one controlled agent trajectory."""

    status: str
    trajectory: AgentTrajectory
    metrics: Mapping[str, Any]
    responses: tuple[GenerationResponse, ...] = ()
    final_answer: str | None = None
    error: Mapping[str, str] | None = None

    @property
    def last_response(self) -> GenerationResponse | None:
        return self.responses[-1] if self.responses else None


class AgentHarness:
    """Run a two-stage agent task with deterministic pre/post observations."""

    def __init__(
        self,
        *,
        runtime: Runtime,
        model: ModelSpec,
        max_action_attempts: int = 3,
    ) -> None:
        if max_action_attempts < 1:
            raise ValueError("max_action_attempts must be positive")
        self.runtime = runtime
        self.model = model
        self.max_action_attempts = max_action_attempts

    def run(
        self,
        task: AgentTask,
        control: TrajectoryControl,
        *,
        sampling: SamplingConfig,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentRun:
        trajectory = AgentTrajectory()
        environment = task.environment()
        trajectory.add_message(
            "system",
            "You are a tool-using agent. Emit exactly one JSON action and no markdown.",
        )
        trajectory.add_message("user", task.objective)
        for index in range(control.pre_discovery_steps):
            result = environment.execute(
                "inspect_noise",
                {"index": index % len(task.distractor_outputs)},
            )
            trajectory.add_tool_result(
                result.tool_name,
                _tool_content(result.output, result.error),
                source="controller",
                metadata={
                    "controlled": True,
                    "phase": "pre_discovery",
                    "observation_index": index,
                    "ok": result.ok,
                },
            )

        responses: list[GenerationResponse] = []
        action_signatures: list[str] = []
        generated_action_n = 0
        parsed_action_n = 0
        tool_attempt_n = 0
        valid_tool_call_n = 0
        repeated_action_n = 0
        recovery_n = 0
        planning_error_n = 0
        critical_discovered = False
        started_at = time.perf_counter()

        discovery_result = self._run_stage(
            trajectory,
            task,
            stage="discovery",
            sampling=sampling,
            metadata=metadata,
            responses=responses,
            action_signatures=action_signatures,
        )
        generated_action_n += discovery_result["generated_action_n"]
        parsed_action_n += discovery_result["parsed_action_n"]
        tool_attempt_n += discovery_result["tool_attempt_n"]
        valid_tool_call_n += discovery_result["valid_tool_call_n"]
        repeated_action_n += discovery_result["repeated_action_n"]
        recovery_n += discovery_result["recovery_n"]
        planning_error_n += discovery_result["planning_error_n"]
        critical_discovered = bool(discovery_result["critical_discovered"])
        if not critical_discovered:
            return self._finish(
                status="invalid_output",
                trajectory=trajectory,
                responses=responses,
                final_answer=None,
                task=task,
                control=control,
                started_at=started_at,
                generated_action_n=generated_action_n,
                parsed_action_n=parsed_action_n,
                tool_attempt_n=tool_attempt_n,
                valid_tool_call_n=valid_tool_call_n,
                repeated_action_n=repeated_action_n,
                recovery_n=recovery_n,
                planning_error_n=planning_error_n,
                critical_discovered=False,
                error=discovery_result.get("error"),
            )

        for offset in range(control.post_discovery_steps):
            index = control.pre_discovery_steps + offset
            result = environment.execute(
                "inspect_noise",
                {"index": index % len(task.distractor_outputs)},
            )
            trajectory.add_tool_result(
                result.tool_name,
                _tool_content(result.output, result.error),
                source="controller",
                metadata={
                    "controlled": True,
                    "phase": "post_discovery",
                    "observation_index": index,
                    "ok": result.ok,
                },
            )

        answer_result = self._run_stage(
            trajectory,
            task,
            stage="answer",
            sampling=sampling,
            metadata=metadata,
            responses=responses,
            action_signatures=action_signatures,
        )
        generated_action_n += answer_result["generated_action_n"]
        parsed_action_n += answer_result["parsed_action_n"]
        tool_attempt_n += answer_result["tool_attempt_n"]
        valid_tool_call_n += answer_result["valid_tool_call_n"]
        repeated_action_n += answer_result["repeated_action_n"]
        recovery_n += answer_result["recovery_n"]
        planning_error_n += answer_result["planning_error_n"]
        final_answer = answer_result.get("final_answer")
        status = "completed" if final_answer is not None else "invalid_output"
        return self._finish(
            status=status,
            trajectory=trajectory,
            responses=responses,
            final_answer=final_answer,
            task=task,
            control=control,
            started_at=started_at,
            generated_action_n=generated_action_n,
            parsed_action_n=parsed_action_n,
            tool_attempt_n=tool_attempt_n,
            valid_tool_call_n=valid_tool_call_n,
            repeated_action_n=repeated_action_n,
            recovery_n=recovery_n,
            planning_error_n=planning_error_n,
            critical_discovered=critical_discovered,
            error=answer_result.get("error"),
        )

    def _run_stage(
        self,
        trajectory: AgentTrajectory,
        task: AgentTask,
        *,
        stage: str,
        sampling: SamplingConfig,
        metadata: Mapping[str, Any] | None,
        responses: list[GenerationResponse],
        action_signatures: list[str],
    ) -> dict[str, Any]:
        generated_action_n = 0
        parsed_action_n = 0
        tool_attempt_n = 0
        valid_tool_call_n = 0
        repeated_action_n = 0
        planning_error_n = 0
        first_failure = None
        for attempt in range(1, self.max_action_attempts + 1):
            request_metadata = dict(metadata or {})
            request_metadata.update(
                {
                    "task_id": task.task_id,
                    "agent_stage": stage,
                    "agent_attempt": attempt,
                    "trajectory_events": len(trajectory.events),
                    "environment_fingerprint": task.environment().fingerprint,
                }
            )
            if "fixture_expected_answer" in request_metadata:
                request_metadata["fixture_mode"] = True
            try:
                response = self.runtime.generate(
                    GenerationRequest(
                        prompt=self._prompt(task, trajectory, stage),
                        model=self.model,
                        sampling=sampling,
                        metadata=request_metadata,
                    )
                )
            except Exception as error:
                return {
                    "generated_action_n": generated_action_n,
                    "parsed_action_n": parsed_action_n,
                    "tool_attempt_n": tool_attempt_n,
                    "valid_tool_call_n": valid_tool_call_n,
                    "repeated_action_n": repeated_action_n,
                    "recovery_n": max(0, attempt - 1),
                    "planning_error_n": planning_error_n,
                    "critical_discovered": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            responses.append(response)
            generated_action_n += 1
            trajectory.add_message(
                "assistant",
                response.output_text or "[empty model response]",
                source="model",
                kind="action",
                metadata={
                    "stage": stage,
                    "attempt": attempt,
                    "empty_output": not bool(response.output_text.strip()),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )
            try:
                action = parse_action(response.output_text)
                parsed_action_n += 1
            except ValueError as error:
                first_failure = first_failure or {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                planning_error_n += 1
                trajectory.add_tool_result(
                    "action_parser",
                    str(error),
                    source="harness",
                    metadata={"ok": False, "stage": stage, "attempt": attempt},
                )
                continue

            signature = _action_signature(action)
            if signature in action_signatures:
                repeated_action_n += 1
            action_signatures.append(signature)
            if action.action == "tool":
                tool_attempt_n += 1
                result = task.environment().execute(
                    action.name or "", action.arguments
                )
                valid_tool_call_n += int(result.ok)
                trajectory.add_tool_result(
                    result.tool_name,
                    _tool_content(result.output, result.error),
                    source="environment",
                    metadata={
                        "ok": result.ok,
                        "stage": stage,
                        "attempt": attempt,
                        "requested_action": action.to_record()
                        if hasattr(action, "to_record")
                        else _action_record(action),
                        "critical_observation": (
                            stage == "discovery"
                            and action.name == "discover_fact"
                            and result.ok
                        ),
                    },
                )
                if stage == "discovery" and action.name == "discover_fact" and result.ok:
                    return {
                        "generated_action_n": generated_action_n,
                        "parsed_action_n": parsed_action_n,
                        "tool_attempt_n": tool_attempt_n,
                        "valid_tool_call_n": valid_tool_call_n,
                        "repeated_action_n": repeated_action_n,
                        "recovery_n": max(0, attempt - 1),
                        "planning_error_n": planning_error_n,
                        "critical_discovered": True,
                    }
                planning_error_n += int(
                    not result.ok
                    or stage == "answer"
                    or (stage == "discovery" and action.name != "discover_fact")
                )
                continue

            if stage == "answer":
                return {
                    "generated_action_n": generated_action_n,
                    "parsed_action_n": parsed_action_n,
                    "tool_attempt_n": tool_attempt_n,
                    "valid_tool_call_n": valid_tool_call_n,
                    "repeated_action_n": repeated_action_n,
                    "recovery_n": max(0, attempt - 1),
                    "planning_error_n": planning_error_n,
                    "critical_discovered": False,
                    "final_answer": action.value,
                }
            planning_error_n += 1
        return {
            "generated_action_n": generated_action_n,
            "parsed_action_n": parsed_action_n,
            "tool_attempt_n": tool_attempt_n,
            "valid_tool_call_n": valid_tool_call_n,
            "repeated_action_n": repeated_action_n,
            "recovery_n": max(0, self.max_action_attempts - 1),
            "planning_error_n": planning_error_n,
            "critical_discovered": False,
            "error": first_failure,
        }

    def _finish(
        self,
        *,
        status: str,
        trajectory: AgentTrajectory,
        responses: list[GenerationResponse],
        final_answer: str | None,
        task: AgentTask,
        control: TrajectoryControl,
        started_at: float,
        generated_action_n: int,
        parsed_action_n: int,
        tool_attempt_n: int,
        valid_tool_call_n: int,
        repeated_action_n: int,
        recovery_n: int,
        planning_error_n: int,
        critical_discovered: bool,
        error: Mapping[str, str] | None,
    ) -> AgentRun:
        total_input_tokens = sum(
            response.usage.prompt_tokens or 0 for response in responses
        )
        context_tokens = [
            response.usage.prompt_tokens
            for response in responses
            if response.usage.prompt_tokens is not None
        ]
        correct = (
            final_answer is not None
            and _normalize(final_answer) == _normalize(task.expected_answer)
        )
        critical_reused = bool(correct and critical_discovered)
        if error and error.get("type") not in {"ActionParseError", None}:
            failure_category = "runtime"
        elif correct:
            failure_category = "success"
        elif planning_error_n or not final_answer:
            failure_category = "tool_planning"
        elif not critical_discovered:
            failure_category = "retrieval"
        else:
            failure_category = "state_tracking"
        metrics = {
            "trajectory_length": control.trajectory_length,
            "requested_critical_position": control.critical_position,
            "actual_critical_position": control.actual_critical_position,
            "critical_observation_index": control.pre_discovery_steps,
            "critical_fact_discovered": critical_discovered,
            "critical_fact_reused": critical_reused,
            "tool_call_n": tool_attempt_n,
            "valid_tool_call_n": valid_tool_call_n,
            "tool_call_validity": (
                valid_tool_call_n / tool_attempt_n if tool_attempt_n else None
            ),
            "generated_action_n": generated_action_n,
            "parsed_action_n": parsed_action_n,
            "repeated_action_n": repeated_action_n,
            "recovery_n": recovery_n,
            "planning_error_n": planning_error_n,
            "total_input_tokens": total_input_tokens,
            "max_input_tokens": max(context_tokens) if context_tokens else None,
            "elapsed_s": time.perf_counter() - started_at,
            "failure_category": failure_category,
        }
        return AgentRun(
            status=status,
            trajectory=trajectory,
            metrics=metrics,
            responses=tuple(responses),
            final_answer=final_answer,
            error=None if error is None else dict(error),
        )

    @staticmethod
    def _prompt(task: AgentTask, trajectory: AgentTrajectory, stage: str) -> str:
        tools = [
            {
                "name": definition.name,
                "description": definition.description,
                "parameters": dict(definition.parameters),
            }
            for definition in task.environment().tools
        ]
        if stage == "discovery":
            instruction = (
                "The next action must discover the critical fact. Return a tool "
                "action calling discover_fact."
            )
        else:
            instruction = (
                "Use the earlier critical fact and return an answer action. "
                "The value must be the requested answer."
            )
        return (
            "Agent task objective:\n"
            f"{task.objective}\n\n"
            "Available tools:\n"
            f"{json.dumps(tools, sort_keys=True)}\n\n"
            f"Current stage: {stage}\n{instruction}\n\n"
            "Accumulated trajectory:\n"
            f"{trajectory.render()}\n\n"
            'Return exactly one JSON object: {"action":"tool",...} or '
            '{"action":"answer","value":"..."}. '
        )


def _action_signature(action: AgentAction) -> str:
    return json.dumps(_action_record(action), sort_keys=True, separators=(",", ":"))


def _action_record(action: AgentAction) -> dict[str, Any]:
    return {
        "action": action.action,
        "name": action.name,
        "arguments": dict(action.arguments),
        "value": action.value,
    }


def _tool_content(output: str, error: str | None) -> str:
    return output if error is None else f"ERROR: {error}"


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())
