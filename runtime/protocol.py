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
