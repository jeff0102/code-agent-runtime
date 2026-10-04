"""Machine-readable contracts between Supervisor and Executor agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


SCHEMA_VERSION = 1


class ProtocolError(ValueError):
    """Raised when an agent message violates the runtime protocol."""


class SupervisorDecisionType(StrEnum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    BLOCK = "BLOCK"


class ExecutorStatus(StrEnum):
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


def _require_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ProtocolError(f"{name} is missing fields: {sorted(missing)}")
    if unknown:
        raise ProtocolError(f"{name} contains unknown fields: {sorted(unknown)}")


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    return value


def _require_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_require_string(item, f"{name}[{index}]"))
    return result


@dataclass(frozen=True, slots=True)
class SupervisorDecision:
    """Validated decision returned by the Supervisor."""

    decision: SupervisorDecisionType
    task_complete: bool
    instructions: list[str]
    blocking_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, SupervisorDecisionType):
            raise ProtocolError("decision must be a SupervisorDecisionType")
        if not isinstance(self.task_complete, bool):
            raise ProtocolError("task_complete must be a boolean")
        _require_string_list(self.instructions, "instructions")
        if self.blocking_reason is not None:
            _require_string(self.blocking_reason, "blocking_reason")

        if self.decision is SupervisorDecisionType.ACCEPT:
            if not self.task_complete:
                raise ProtocolError("ACCEPT requires task_complete=true")
            if self.instructions:
                raise ProtocolError("ACCEPT must not contain revision instructions")
            if self.blocking_reason is not None:
                raise ProtocolError("ACCEPT must not contain blocking_reason")

        elif self.decision is SupervisorDecisionType.REVISE:
            if self.task_complete:
                raise ProtocolError("REVISE requires task_complete=false")
            if not self.instructions:
                raise ProtocolError("REVISE requires at least one instruction")
            if self.blocking_reason is not None:
                raise ProtocolError("REVISE must not contain blocking_reason")

        elif self.decision is SupervisorDecisionType.BLOCK:
            if self.task_complete:
                raise ProtocolError("BLOCK requires task_complete=false")
            if not self.blocking_reason or not self.blocking_reason.strip():
                raise ProtocolError("BLOCK requires blocking_reason")
            if self.instructions:
                raise ProtocolError("BLOCK must not contain revision instructions")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical JSON-compatible representation."""
        return {
            "schema_version": SCHEMA_VERSION,
            "message_type": "supervisor_decision",
            "decision": self.decision.value,
            "task_complete": self.task_complete,
            "instructions": list(self.instructions),
            "blocking_reason": self.blocking_reason,
        }

    def to_json(self) -> str:
        """Serialize to deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorDecision":
        data = _require_object(payload, "SupervisorDecision")
        _require_exact_keys(
            data,
            {
                "schema_version",
                "message_type",
                "decision",
                "task_complete",
                "instructions",
                "blocking_reason",
            },
            "SupervisorDecision",
        )

        if data["schema_version"] != SCHEMA_VERSION:
            raise ProtocolError(
                f"Unsupported SupervisorDecision schema_version: {data['schema_version']!r}"
            )
        if data["message_type"] != "supervisor_decision":
            raise ProtocolError("Invalid SupervisorDecision message_type")

        try:
            decision = SupervisorDecisionType(data["decision"])
        except ValueError as exc:
            raise ProtocolError(f"Invalid supervisor decision: {data['decision']!r}") from exc

        if not isinstance(data["task_complete"], bool):
            raise ProtocolError("task_complete must be a boolean")

        instructions = _require_string_list(data["instructions"], "instructions")

        blocking_reason = data["blocking_reason"]
        if blocking_reason is not None:
            blocking_reason = _require_string(blocking_reason, "blocking_reason")

        return cls(
            decision=decision,
            task_complete=data["task_complete"],
            instructions=instructions,
            blocking_reason=blocking_reason,
        )

    @classmethod
    def from_json(cls, payload: str) -> "SupervisorDecision":
        """Parse and validate a JSON-encoded Supervisor decision."""
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"Invalid SupervisorDecision JSON: {exc}") from exc
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class ExecutorReport:
    """Validated completion report returned by the Executor."""

    status: ExecutorStatus
    summary: str
    changed_files: list[str]
    tests_executed: list[str]
    validation_summary: str
    blockers: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutorStatus):
            raise ProtocolError("status must be an ExecutorStatus")
        _require_string(self.summary, "summary")
        _require_string_list(self.changed_files, "changed_files")
        _require_string_list(self.tests_executed, "tests_executed")
        _require_string(self.validation_summary, "validation_summary")
        _require_string_list(self.blockers, "blockers")

        if self.status is ExecutorStatus.COMPLETED and self.blockers:
            raise ProtocolError("COMPLETED must not contain blockers")

        if self.status in {ExecutorStatus.BLOCKED, ExecutorStatus.FAILED} and not self.blockers:
            raise ProtocolError(
                f"{self.status.value} requires at least one blocker"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical JSON-compatible representation."""
        return {
            "schema_version": SCHEMA_VERSION,
            "message_type": "executor_report",
            "status": self.status.value,
            "summary": self.summary,
            "changed_files": list(self.changed_files),
            "tests_executed": list(self.tests_executed),
            "validation_summary": self.validation_summary,
            "blockers": list(self.blockers),
        }

    def to_json(self) -> str:
        """Serialize to deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutorReport":
        data = _require_object(payload, "ExecutorReport")
        _require_exact_keys(
            data,
            {
                "schema_version",
                "message_type",
                "status",
                "summary",
                "changed_files",
                "tests_executed",
                "validation_summary",
                "blockers",
            },
            "ExecutorReport",
        )

        if data["schema_version"] != SCHEMA_VERSION:
            raise ProtocolError(
                f"Unsupported ExecutorReport schema_version: {data['schema_version']!r}"
            )
        if data["message_type"] != "executor_report":
            raise ProtocolError("Invalid ExecutorReport message_type")

        try:
            status = ExecutorStatus(data["status"])
        except ValueError as exc:
            raise ProtocolError(f"Invalid executor status: {data['status']!r}") from exc

        return cls(
            status=status,
            summary=_require_string(data["summary"], "summary"),
            changed_files=_require_string_list(data["changed_files"], "changed_files"),
            tests_executed=_require_string_list(data["tests_executed"], "tests_executed"),
            validation_summary=_require_string(
                data["validation_summary"],
                "validation_summary",
            ),
            blockers=_require_string_list(data["blockers"], "blockers"),
        )

    @classmethod
    def from_json(cls, payload: str) -> "ExecutorReport":
        """Parse and validate a JSON-encoded Executor report."""
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"Invalid ExecutorReport JSON: {exc}") from exc
        return cls.from_dict(decoded)
