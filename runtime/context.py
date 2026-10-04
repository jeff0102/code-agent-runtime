"""Typed, bounded context packages for Supervisor and Executor agents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runtime.models import Task
from runtime.protocol import ExecutorReport, SupervisorDecision
from runtime.validation import ValidationResult
from runtime.workspace import WorkspaceSnapshot


class ContextError(ValueError):
    """Raised when an agent context cannot be assembled safely."""


@dataclass(frozen=True, slots=True)
class ContextLimits:
    """Maximum text sizes included in an agent context."""

    scope_chars: int = 50_000
    agents_chars: int = 30_000
    diff_chars: int = 40_000
    git_status_chars: int = 10_000
    revision_instructions_chars: int = 10_000
    validation_summary_chars: int = 20_000
    executor_report_chars: int = 20_000
    prior_decision_chars: int = 10_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ContextError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class ScopeContext:
    """Target project scope and agent contract content."""

    scope_hash: str
    scope_text: str
    agents_text: str | None


@dataclass(frozen=True, slots=True)
class GitContext:
    """Read-only Git state presented to an agent."""

    branch: str
    base_commit: str
    status: str
    diff: str


@dataclass(frozen=True, slots=True)
class ExecutorContext:
    """Implementation context presented to the Executor."""

    repository: str
    session_id: str
    task: Task
    scope: ScopeContext
    git: GitContext
    previous_revision_instructions: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible context."""
        return {
            "context_type": "executor",
            "schema_version": 1,
            "repository": self.repository,
            "session_id": self.session_id,
            "task": _task_dict(self.task),
            "scope": {
                "scope_hash": self.scope.scope_hash,
                "scope_text": self.scope.scope_text,
                "agents_text": self.scope.agents_text,
            },
            "git": {
                "branch": self.git.branch,
                "base_commit": self.git.base_commit,
                "status": self.git.status,
                "diff": self.git.diff,
            },
            "previous_revision_instructions": list(
                self.previous_revision_instructions
            ),
        }

    def to_json(self) -> str:
        """Serialize context deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class SupervisorContext:
    """Read-only review context presented to the Supervisor."""

    repository: str
    session_id: str
    task: Task
    scope: ScopeContext
    git: GitContext
    validation: dict[str, Any]
    executor_report: dict[str, Any] | None
    previous_decision: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible context."""
        return {
            "context_type": "supervisor",
            "schema_version": 1,
            "repository": self.repository,
            "session_id": self.session_id,
            "task": _task_dict(self.task),
            "scope": {
                "scope_hash": self.scope.scope_hash,
                "scope_text": self.scope.scope_text,
                "agents_text": self.scope.agents_text,
            },
            "git": {
                "branch": self.git.branch,
                "base_commit": self.git.base_commit,
                "status": self.git.status,
                "diff": self.git.diff,
            },
            "validation": self.validation,
            "executor_report": self.executor_report,
            "previous_decision": self.previous_decision,
        }

    def to_json(self) -> str:
        """Serialize context deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)




@dataclass(frozen=True, slots=True)
class PlannerContext:
    """Read-only context used to select the next implementation task."""

    repository: str
    session_id: str
    scope: ScopeContext
    git: GitContext
    completed_tasks: list[dict[str, Any]]
    next_sequence: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_type": "supervisor_planner",
            "schema_version": 1,
            "repository": self.repository,
            "session_id": self.session_id,
            "scope": {
                "scope_hash": self.scope.scope_hash,
                "scope_text": self.scope.scope_text,
                "agents_text": self.scope.agents_text,
            },
            "git": {
                "branch": self.git.branch,
                "base_commit": self.git.base_commit,
                "status": self.git.status,
                "diff": self.git.diff,
            },
            "completed_tasks": list(self.completed_tasks),
            "next_sequence": self.next_sequence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)


def build_scope_context(
    workspace_path: str | Path,
    *,
    scope_hash: str,
    limits: ContextLimits | None = None,
) -> ScopeContext:
    """Load the target scope and optional agent contract."""
    limits = limits or ContextLimits()
    workspace = Path(workspace_path)

    scope_path = workspace / "SCOPE.md"
    if not scope_path.is_file():
        raise ContextError(f"Required scope file not found: {scope_path}")

    scope_text = _read_bounded(scope_path, limits.scope_chars)

    agents_path = workspace / "AGENTS.md"
    agents_text = (
        _read_bounded(agents_path, limits.agents_chars)
        if agents_path.is_file()
        else None
    )

    return ScopeContext(
        scope_hash=scope_hash,
        scope_text=scope_text,
        agents_text=agents_text,
    )


def build_git_context(
    snapshot: WorkspaceSnapshot,
    *,
    base_commit: str,
    limits: ContextLimits | None = None,
) -> GitContext:
    """Build a bounded read-only Git context."""
    limits = limits or ContextLimits()
    return GitContext(
        branch=snapshot.branch,
        base_commit=base_commit,
        status=_bounded_text(snapshot.status, limits.git_status_chars),
        diff=_bounded_text(snapshot.diff, limits.diff_chars),
    )




def build_planner_context(
    *,
    repository: str,
    session_id: str,
    scope: ScopeContext,
    git: GitContext,
    completed_tasks: list[Task],
    next_sequence: int,
    limits: ContextLimits | None = None,
) -> PlannerContext:
    """Build a bounded context for selecting the next project task."""
    if next_sequence < 1:
        raise ContextError("next_sequence must be greater than zero")
    return PlannerContext(
        repository=repository,
        session_id=session_id,
        scope=scope,
        git=git,
        completed_tasks=[
            {
                "sequence": task.sequence,
                "title": _bounded_text(task.title, 2_000).strip(),
                "status": task.status.value,
                "attempt_count": task.attempt_count,
            }
            for task in completed_tasks
        ],
        next_sequence=next_sequence,
    )


def build_executor_context(
    *,
    repository: str,
    session_id: str,
    task: Task,
    scope: ScopeContext,
    git: GitContext,
    previous_revision_instructions: list[str] | None = None,
    limits: ContextLimits | None = None,
) -> ExecutorContext:
    """Build the implementation context for an Executor attempt."""
    limits = limits or ContextLimits()
    instructions = [
        _bounded_text(item, limits.revision_instructions_chars)
        for item in (previous_revision_instructions or [])
    ]
    return ExecutorContext(
        repository=repository,
        session_id=session_id,
        task=task,
        scope=scope,
        git=git,
        previous_revision_instructions=instructions,
    )


def build_supervisor_context(
    *,
    repository: str,
    session_id: str,
    task: Task,
    scope: ScopeContext,
    git: GitContext,
    validation: ValidationResult,
    executor_report: ExecutorReport | None,
    previous_decision: SupervisorDecision | None = None,
    limits: ContextLimits | None = None,
) -> SupervisorContext:
    """Build the read-only review context for a Supervisor attempt."""
    limits = limits or ContextLimits()

    validation_dict = _validation_dict(validation)
    validation_json = json.dumps(validation_dict, sort_keys=True, ensure_ascii=False)
    if len(validation_json) > limits.validation_summary_chars:
        validation_dict = {
            "truncated": True,
            "summary": _bounded_text(
                validation_json,
                limits.validation_summary_chars,
            ),
        }

    report_dict = executor_report.to_dict() if executor_report else None
    if report_dict is not None:
        report_json = json.dumps(report_dict, sort_keys=True, ensure_ascii=False)
        if len(report_json) > limits.executor_report_chars:
            report_dict = {
                "truncated": True,
                "summary": _bounded_text(
                    report_json,
                    limits.executor_report_chars,
                ),
            }

    decision_dict = previous_decision.to_dict() if previous_decision else None
    if decision_dict is not None:
        decision_json = json.dumps(
            decision_dict,
            sort_keys=True,
            ensure_ascii=False,
        )
        if len(decision_json) > limits.prior_decision_chars:
            decision_dict = {
                "truncated": True,
                "summary": _bounded_text(
                    decision_json,
                    limits.prior_decision_chars,
                ),
            }

    return SupervisorContext(
        repository=repository,
        session_id=session_id,
        task=task,
        scope=scope,
        git=git,
        validation=validation_dict,
        executor_report=report_dict,
        previous_decision=decision_dict,
    )


def _task_dict(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "session_id": task.session_id,
        "sequence": task.sequence,
        "title": task.title,
        "objective": task.objective,
        "instructions": task.instructions,
        "acceptance_criteria": task.acceptance_criteria,
        "status": task.status.value,
        "attempt_count": task.attempt_count,
        "created_at": task.created_at,
        "completed_at": task.completed_at,
    }


def _validation_dict(validation: ValidationResult) -> dict[str, Any]:
    return {
        "success": validation.success,
        "commands": [
            {
                "name": command.name,
                "argv": list(command.argv),
                "required": command.required,
                "exit_code": command.exit_code,
                "timed_out": command.timed_out,
                "duration_seconds": round(command.duration_seconds, 6),
                "stdout_artifact": command.stdout_artifact,
                "stderr_artifact": command.stderr_artifact,
            }
            for command in validation.commands
        ],
    }


def _read_bounded(path: Path, limit: int) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContextError(f"Unable to read context file: {path}") from exc
    return _bounded_text(content, limit)


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n\n[TRUNCATED: {remaining} characters omitted]\n\n"
    remaining = len(value) - limit
    marker_text = marker.format(remaining=remaining)

    if len(marker_text) >= limit:
        return value[:limit]

    available = limit - len(marker_text)
    prefix_size = (available + 1) // 2
    suffix_size = available // 2

    return (
        value[:prefix_size]
        + marker_text
        + value[-suffix_size:]
    )
