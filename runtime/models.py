"""Domain models and state enums used by the orchestration runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SessionStatus(StrEnum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    FAILED = "FAILED"


class TaskStatus(StrEnum):
    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    REVIEWING = "REVIEWING"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class Decision(StrEnum):
    PENDING = "PENDING"
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    BLOCK = "BLOCK"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    repository: str
    workspace_path: str
    branch: str
    scope_hash: str
    status: SessionStatus
    current_task_id: str | None
    max_iterations: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    session_id: str
    sequence: int
    title: str
    objective: str
    instructions: str
    acceptance_criteria: str
    status: TaskStatus
    attempt_count: int
    created_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class Iteration:
    iteration_id: str
    task_id: str
    attempt_number: int
    executor_conversation_id: str | None
    supervisor_conversation_id: str | None
    base_commit: str
    decision: Decision
    started_at: str
    completed_at: str | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class Checkpoint:
    checkpoint_id: str
    task_id: str
    iteration_id: str
    commit_sha: str
    created_at: str
