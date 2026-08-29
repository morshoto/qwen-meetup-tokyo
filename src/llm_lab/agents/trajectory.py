"""Serializable agent trajectories and deterministic tool fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


_ROLES = frozenset({"system", "user", "assistant", "tool"})
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


class ActionParseError(ValueError):
    """Raised when a model response is not a supported agent action."""


@dataclass(frozen=True)
class AgentAction:
    """One structured action emitted by an agent model."""

    action: str
    name: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    value: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"tool", "answer"}:
            raise ActionParseError(f"unsupported action: {self.action!r}")
        if self.action == "tool":
            if not self.name or not self.name.strip():
                raise ActionParseError("tool action requires a non-empty name")
            if not isinstance(self.arguments, Mapping):
                raise ActionParseError("tool action arguments must be an object")
        elif self.value is None or not self.value.strip():
            raise ActionParseError("answer action requires a non-empty value")

    def to_record(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "name": self.name,
            "arguments": dict(self.arguments),
            "value": self.value,
        }


def parse_action(text: str) -> AgentAction:
    """Parse canonical actions and Qwen's equivalent thinking/tool aliases."""

    candidate = _THINK_BLOCK_RE.sub("", text.strip()).strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        record = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ActionParseError("agent response must be a JSON object") from error
    if not isinstance(record, Mapping):
        raise ActionParseError("agent response must be a JSON object")
    action = record.get("action")
    if not isinstance(action, str):
        raise ActionParseError("agent response requires a string action")
    if action == "tool":
        if "name" in record and "tool" in record and record["name"] != record["tool"]:
            raise ActionParseError("tool action name aliases must agree")
        if (
            "arguments" in record
            and "parameters" in record
            and record["arguments"] != record["parameters"]
        ):
            raise ActionParseError("tool action argument aliases must agree")
        name = record.get("name", record.get("tool"))
        arguments = record.get("arguments", record.get("parameters", {}))
        if not isinstance(arguments, Mapping):
            raise ActionParseError("tool action arguments must be an object")
        return AgentAction(
            action=action,
            name=_optional_text(name),
            arguments=dict(arguments),
        )
    if action == "answer":
        return AgentAction(action=action, value=_optional_text(record.get("value")))
    raise ActionParseError(f"unsupported action: {action!r}")


@dataclass(frozen=True)
class TrajectoryEvent:
    """One ordered, machine-readable item in an agent history."""

    index: int
    role: str
    content: str
    source: str = "harness"
    kind: str = "message"
    tool_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("trajectory event index must not be negative")
        if self.role not in _ROLES:
            raise ValueError(f"unsupported trajectory role: {self.role!r}")
        if not self.content.strip():
            raise ValueError("trajectory event content must be non-empty")
        if not self.source.strip() or not self.kind.strip():
            raise ValueError("trajectory event source and kind must be non-empty")
        if self.role == "tool" and (self.tool_name is None or not self.tool_name.strip()):
            raise ValueError("tool events require a tool_name")

    def to_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "kind": self.kind,
            "tool_name": self.tool_name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TrajectoryEvent":
        return cls(
            index=int(record["index"]),
            role=str(record["role"]),
            content=str(record["content"]),
            source=str(record.get("source", "harness")),
            kind=str(record.get("kind", "message")),
            tool_name=(
                None if record.get("tool_name") is None else str(record["tool_name"])
            ),
            metadata=_mapping(record.get("metadata", {})),
        )


class AgentTrajectory:
    """Mutable builder for an ordered trajectory with canonical serialization."""

    def __init__(self, events: tuple[TrajectoryEvent, ...] = ()) -> None:
        self._events = list(events)
        self._validate_indexes()

    @property
    def events(self) -> tuple[TrajectoryEvent, ...]:
        return tuple(self._events)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_records(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def add_message(
        self,
        role: str,
        content: str,
        *,
        source: str = "harness",
        kind: str = "message",
        tool_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrajectoryEvent:
        event = TrajectoryEvent(
            index=len(self._events),
            role=role,
            content=content,
            source=source,
            kind=kind,
            tool_name=tool_name,
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        return event

    def add_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        source: str = "environment",
        metadata: Mapping[str, Any] | None = None,
    ) -> TrajectoryEvent:
        return self.add_message(
            "tool",
            content,
            source=source,
            kind="tool_result",
            tool_name=tool_name,
            metadata=metadata,
        )

    def to_records(self) -> list[dict[str, Any]]:
        return [event.to_record() for event in self._events]

    @classmethod
    def from_records(cls, records: list[Mapping[str, Any]]) -> "AgentTrajectory":
        events = tuple(TrajectoryEvent.from_record(record) for record in records)
        return cls(events)

    def render(self) -> str:
        """Render the full history for a backend-neutral generation prompt."""

        lines = []
        for event in self._events:
            label = event.role if event.tool_name is None else f"tool:{event.tool_name}"
            lines.append(f"[{event.index}] {label}: {event.content}")
        return "\n".join(lines)

    def _validate_indexes(self) -> None:
        if [event.index for event in self._events] != list(range(len(self._events))):
            raise ValueError("trajectory event indexes must be contiguous from zero")


@dataclass(frozen=True)
class ToolDefinition:
    """Backend-neutral tool metadata included in agent prompts."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("tool name and description must be non-empty")

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ToolResult:
    """Deterministic tool output, including failures as first-class records."""

    tool_name: str
    output: str
    ok: bool = True
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "output": self.output,
            "ok": self.ok,
            "error": self.error,
        }


class DeterministicToolEnvironment:
    """Small deterministic environment for controlled agent trajectories."""

    def __init__(
        self,
        *,
        critical_observation: str,
        distractor_outputs: tuple[str, ...],
    ) -> None:
        if not critical_observation.strip():
            raise ValueError("critical_observation must be non-empty")
        if not distractor_outputs or any(not output.strip() for output in distractor_outputs):
            raise ValueError("distractor_outputs must contain non-empty outputs")
        self.critical_observation = critical_observation
        self.distractor_outputs = tuple(distractor_outputs)

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                name="discover_fact",
                description="Return the task's critical fact.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            ToolDefinition(
                name="inspect_noise",
                description="Return one deterministic unrelated observation by index.",
                parameters={
                    "type": "object",
                    "properties": {"index": {"type": "integer", "minimum": 0}},
                    "required": ["index"],
                    "additionalProperties": False,
                },
            ),
        )

    @property
    def fingerprint(self) -> str:
        record = {
            "critical_observation": self.critical_observation,
            "distractor_outputs": list(self.distractor_outputs),
            "tools": [tool.to_record() for tool in self.tools],
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        args = dict(arguments or {})
        if tool_name == "discover_fact":
            if args:
                return ToolResult(
                    tool_name=tool_name,
                    output="",
                    ok=False,
                    error="discover_fact does not accept arguments",
                )
            return ToolResult(tool_name=tool_name, output=self.critical_observation)
        if tool_name == "inspect_noise":
            index = args.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                return ToolResult(
                    tool_name=tool_name,
                    output="",
                    ok=False,
                    error="inspect_noise requires an integer index",
                )
            if index < 0 or index >= len(self.distractor_outputs):
                return ToolResult(
                    tool_name=tool_name,
                    output="",
                    ok=False,
                    error=f"distractor index out of range: {index}",
                )
            return ToolResult(tool_name=tool_name, output=self.distractor_outputs[index])
        return ToolResult(
            tool_name=tool_name,
            output="",
            ok=False,
            error=f"unknown tool: {tool_name}",
        )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ActionParseError("action text fields must be strings")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("trajectory metadata must be an object")
    return dict(value)
