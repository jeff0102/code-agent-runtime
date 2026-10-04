"""SQLite-backed durable state for autonomous coding sessions."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from runtime.models import (
    Checkpoint,
    Decision,
    Iteration,
    Session,
    SessionStatus,
    Task,
    TaskStatus,
)


SESSION_TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.RUNNING: frozenset(
        {SessionStatus.PAUSED, SessionStatus.BLOCKED, SessionStatus.DONE, SessionStatus.FAILED}
    ),
    SessionStatus.PAUSED: frozenset({SessionStatus.RUNNING, SessionStatus.BLOCKED, SessionStatus.FAILED}),
    SessionStatus.BLOCKED: frozenset({SessionStatus.RUNNING, SessionStatus.FAILED}),
    SessionStatus.DONE: frozenset(),
    SessionStatus.FAILED: frozenset(),
}

TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PLANNED: frozenset({TaskStatus.EXECUTING, TaskStatus.BLOCKED, TaskStatus.FAILED}),
    TaskStatus.EXECUTING: frozenset(
        {TaskStatus.REVIEWING, TaskStatus.REVISION_REQUIRED, TaskStatus.BLOCKED, TaskStatus.FAILED}
    ),
    TaskStatus.REVIEWING: frozenset(
        {TaskStatus.ACCEPTED, TaskStatus.REVISION_REQUIRED, TaskStatus.BLOCKED, TaskStatus.FAILED}
    ),
    TaskStatus.REVISION_REQUIRED: frozenset(
        {TaskStatus.EXECUTING, TaskStatus.BLOCKED, TaskStatus.FAILED}
    ),
    TaskStatus.ACCEPTED: frozenset(),
    TaskStatus.BLOCKED: frozenset({TaskStatus.EXECUTING, TaskStatus.FAILED}),
    TaskStatus.FAILED: frozenset(),
}


class StateError(RuntimeError):
    """Raised when a persisted state operation violates runtime invariants."""


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


class StateStore:
    """Durable store for sessions, tasks, iterations, checkpoints, and events."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured SQLite connection."""
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the runtime schema when it does not already exist."""
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    repository TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    scope_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_task_id TEXT,
                    max_iterations INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    acceptance_criteria TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(session_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS iterations (
                    iteration_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    attempt_number INTEGER NOT NULL,
                    executor_conversation_id TEXT,
                    supervisor_conversation_id TEXT,
                    base_commit TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    failure_reason TEXT,
                    UNIQUE(task_id, attempt_number)
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    iteration_id TEXT NOT NULL REFERENCES iterations(iteration_id) ON DELETE CASCADE,
                    commit_sha TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, iteration_id),
                    UNIQUE(commit_sha)
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
                    iteration_id TEXT REFERENCES iterations(iteration_id) ON DELETE SET NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
                    iteration_id TEXT REFERENCES iterations(iteration_id) ON DELETE SET NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS leases (
                    workspace_path TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_session_status
                    ON tasks(session_id, status);
                CREATE INDEX IF NOT EXISTS idx_iterations_task
                    ON iterations(task_id, attempt_number);
                CREATE INDEX IF NOT EXISTS idx_events_session_created
                    ON events(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_artifacts_iteration
                    ON artifacts(iteration_id);
                """
            )

    def create_session(
        self,
        repository: str,
        workspace_path: str,
        branch: str,
        scope_hash: str,
        max_iterations: int,
        session_id: str | None = None,
    ) -> Session:
        """Create a new active development session."""
        if max_iterations < 1:
            raise ValueError("max_iterations must be greater than zero")
        session_id = session_id or str(uuid.uuid4())
        timestamp = utc_now()

        with self.connection() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO sessions (
                        session_id, repository, workspace_path, branch, scope_hash,
                        status, current_task_id, max_iterations, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        session_id,
                        repository,
                        workspace_path,
                        branch,
                        scope_hash,
                        SessionStatus.RUNNING.value,
                        max_iterations,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise StateError(f"Session already exists: {session_id}") from exc

        self.append_event(session_id, None, None, "SESSION_CREATED", {})
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Session:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise StateError(f"Unknown session: {session_id}")
        return Session(
            session_id=row["session_id"],
            repository=row["repository"],
            workspace_path=row["workspace_path"],
            branch=row["branch"],
            scope_hash=row["scope_hash"],
            status=SessionStatus(row["status"]),
            current_task_id=row["current_task_id"],
            max_iterations=row["max_iterations"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def set_session_status(self, session_id: str, status: SessionStatus) -> Session:
        """Transition a session to a valid next state."""
        session = self.get_session(session_id)
        if status == session.status:
            return session
        if status not in SESSION_TRANSITIONS[session.status]:
            raise StateError(
                f"Invalid session transition: {session.status.value} -> {status.value}"
            )

        timestamp = utc_now()
        with self.connection() as connection:
            connection.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (status.value, timestamp, session_id),
            )
            connection.commit()

        self.append_event(session_id, None, None, "SESSION_STATUS_CHANGED", {
            "from": session.status.value,
            "to": status.value,
        })
        return self.get_session(session_id)

    def create_task(
        self,
        session_id: str,
        sequence: int,
        title: str,
        objective: str,
        instructions: str,
        acceptance_criteria: str,
        task_id: str | None = None,
    ) -> Task:
        """Create one atomic task for a session."""
        self.get_session(session_id)
        if sequence < 1:
            raise ValueError("sequence must be greater than zero")
        task_id = task_id or str(uuid.uuid4())
        timestamp = utc_now()

        with self.connection() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_id, session_id, sequence, title, objective,
                        instructions, acceptance_criteria, status, attempt_count,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL)
                    """,
                    (
                        task_id,
                        session_id,
                        sequence,
                        title,
                        objective,
                        instructions,
                        acceptance_criteria,
                        TaskStatus.PLANNED.value,
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    UPDATE sessions
                    SET current_task_id = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (task_id, timestamp, session_id),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise StateError(
                    f"Task sequence already exists for session: {sequence}"
                ) from exc

        self.append_event(session_id, task_id, None, "TASK_CREATED", {
            "sequence": sequence,
            "title": title,
        })
        return self.get_task(task_id)

    def list_tasks(self, session_id: str) -> list[Task]:
        """Return tasks for a session in deterministic sequence order."""
        self.get_session(session_id)
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM tasks
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()
        return [
            Task(
                task_id=row["task_id"],
                session_id=row["session_id"],
                sequence=row["sequence"],
                title=row["title"],
                objective=row["objective"],
                instructions=row["instructions"],
                acceptance_criteria=row["acceptance_criteria"],
                status=TaskStatus(row["status"]),
                attempt_count=row["attempt_count"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
            )
            for row in rows
        ]

    def next_task_sequence(self, session_id: str) -> int:
        """Return the next task sequence number for a session."""
        self.get_session(session_id)
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM tasks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["next_sequence"])

    def latest_iteration(self, task_id: str) -> Iteration | None:
        """Return the most recent iteration for a task."""
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM iterations
                WHERE task_id = ?
                ORDER BY attempt_number DESC
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return Iteration(
            iteration_id=row["iteration_id"],
            task_id=row["task_id"],
            attempt_number=row["attempt_number"],
            executor_conversation_id=row["executor_conversation_id"],
            supervisor_conversation_id=row["supervisor_conversation_id"],
            base_commit=row["base_commit"],
            decision=Decision(row["decision"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            failure_reason=row["failure_reason"],
        )

    def get_task_by_sequence(self, session_id: str, sequence: int) -> Task | None:
        """Return a task by session sequence number, if it exists."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE session_id = ? AND sequence = ?",
                (session_id, sequence),
            ).fetchone()
        if row is None:
            return None
        return Task(
            task_id=row["task_id"],
            session_id=row["session_id"],
            sequence=row["sequence"],
            title=row["title"],
            objective=row["objective"],
            instructions=row["instructions"],
            acceptance_criteria=row["acceptance_criteria"],
            status=TaskStatus(row["status"]),
            attempt_count=row["attempt_count"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    def get_task(self, task_id: str) -> Task:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise StateError(f"Unknown task: {task_id}")
        return Task(
            task_id=row["task_id"],
            session_id=row["session_id"],
            sequence=row["sequence"],
            title=row["title"],
            objective=row["objective"],
            instructions=row["instructions"],
            acceptance_criteria=row["acceptance_criteria"],
            status=TaskStatus(row["status"]),
            attempt_count=row["attempt_count"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    def set_task_status(self, task_id: str, status: TaskStatus) -> Task:
        """Transition a task to a valid next state."""
        task = self.get_task(task_id)
        if status == task.status:
            return task
        if status not in TASK_TRANSITIONS[task.status]:
            raise StateError(
                f"Invalid task transition: {task.status.value} -> {status.value}"
            )

        completed_at = utc_now() if status in {
            TaskStatus.ACCEPTED,
            TaskStatus.BLOCKED,
            TaskStatus.FAILED,
        } else task.completed_at
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, completed_at = ?
                WHERE task_id = ?
                """,
                (status.value, completed_at, task_id),
            )
            connection.commit()

        self.append_event(task.session_id, task_id, None, "TASK_STATUS_CHANGED", {
            "from": task.status.value,
            "to": status.value,
        })
        return self.get_task(task_id)

    def start_iteration(
        self,
        task_id: str,
        base_commit: str,
        executor_conversation_id: str | None = None,
        supervisor_conversation_id: str | None = None,
    ) -> Iteration:
        """Start the next Executor attempt for a task."""
        task = self.get_task(task_id)
        if task.status not in {TaskStatus.PLANNED, TaskStatus.REVISION_REQUIRED}:
            raise StateError(f"Cannot start iteration from task status {task.status.value}")

        attempt_number = task.attempt_count + 1
        iteration_id = str(uuid.uuid4())
        timestamp = utc_now()

        with self.connection() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, attempt_count = ?
                WHERE task_id = ?
                """,
                (TaskStatus.EXECUTING.value, attempt_number, task_id),
            )
            connection.execute(
                """
                INSERT INTO iterations (
                    iteration_id, task_id, attempt_number,
                    executor_conversation_id, supervisor_conversation_id,
                    base_commit, decision, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iteration_id,
                    task_id,
                    attempt_number,
                    executor_conversation_id,
                    supervisor_conversation_id,
                    base_commit,
                    Decision.PENDING.value,
                    timestamp,
                ),
            )
            connection.commit()

        self.append_event(
            task.session_id,
            task_id,
            iteration_id,
            "ITERATION_STARTED",
            {"attempt_number": attempt_number, "base_commit": base_commit},
        )
        return self.get_iteration(iteration_id)

    def get_pending_iteration(self, task_id: str) -> Iteration | None:
        """Return the task's active pending iteration, if one exists."""
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM iterations
                WHERE task_id = ? AND decision = ?
                ORDER BY attempt_number DESC
                LIMIT 1
                """,
                (task_id, Decision.PENDING.value),
            ).fetchone()
        if row is None:
            return None
        return Iteration(
            iteration_id=row["iteration_id"],
            task_id=row["task_id"],
            attempt_number=row["attempt_number"],
            executor_conversation_id=row["executor_conversation_id"],
            supervisor_conversation_id=row["supervisor_conversation_id"],
            base_commit=row["base_commit"],
            decision=Decision(row["decision"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            failure_reason=row["failure_reason"],
        )

    def latest_pending_plan(self, session_id: str) -> dict[str, Any] | None:
        """Return the last planning decision not yet followed by task creation."""
        events = self.list_events(session_id)
        last_task_event = -1
        for index, event in enumerate(events):
            if event["event_type"] == "TASK_CREATED":
                last_task_event = index
        for event in reversed(events[last_task_event + 1:]):
            if event["event_type"] == "SUPERVISOR_PLANNING_DECISION":
                payload = event["payload"]
                planned_sequence = payload.get("next_sequence")
                if isinstance(planned_sequence, int) and self.get_task_by_sequence(
                    session_id,
                    planned_sequence,
                ) is not None:
                    return None
                return payload
            if event["event_type"] in {"SESSION_PLANNING_FAILED", "SESSION_PLANNING_BLOCKED"}:
                return None
        return None

    def latest_supervisor_decision(self, iteration_id: str) -> dict[str, Any] | None:
        """Return the latest persisted Supervisor decision event for an iteration."""
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT payload
                FROM events
                WHERE iteration_id = ? AND event_type = ?
                ORDER BY created_at DESC, event_id DESC
                LIMIT 1
                """,
                (iteration_id, "SUPERVISOR_DECISION"),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])

    def list_active_sessions(self) -> list[Session]:
        """Return sessions that may require startup recovery."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM sessions
                WHERE status IN (?, ?)
                ORDER BY created_at
                """,
                (SessionStatus.RUNNING.value, SessionStatus.PAUSED.value),
            ).fetchall()
        return [
            Session(
                session_id=row["session_id"],
                repository=row["repository"],
                workspace_path=row["workspace_path"],
                branch=row["branch"],
                scope_hash=row["scope_hash"],
                status=SessionStatus(row["status"]),
                current_task_id=row["current_task_id"],
                max_iterations=row["max_iterations"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_iteration(self, iteration_id: str) -> Iteration:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM iterations WHERE iteration_id = ?", (iteration_id,)
            ).fetchone()
        if row is None:
            raise StateError(f"Unknown iteration: {iteration_id}")
        return Iteration(
            iteration_id=row["iteration_id"],
            task_id=row["task_id"],
            attempt_number=row["attempt_number"],
            executor_conversation_id=row["executor_conversation_id"],
            supervisor_conversation_id=row["supervisor_conversation_id"],
            base_commit=row["base_commit"],
            decision=Decision(row["decision"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            failure_reason=row["failure_reason"],
        )

    def set_iteration_conversations(
        self,
        iteration_id: str,
        executor_conversation_id: str | None,
        supervisor_conversation_id: str | None,
    ) -> Iteration:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE iterations
                SET executor_conversation_id = ?,
                    supervisor_conversation_id = ?
                WHERE iteration_id = ?
                """,
                (executor_conversation_id, supervisor_conversation_id, iteration_id),
            )
            connection.commit()
        return self.get_iteration(iteration_id)

    def complete_iteration(
        self,
        iteration_id: str,
        decision: Decision,
        *,
        failure_reason: str | None = None,
    ) -> Iteration:
        """Persist a Supervisor decision and advance the task state."""
        iteration = self.get_iteration(iteration_id)
        if iteration.decision != Decision.PENDING:
            raise StateError("Iteration has already been completed")

        task = self.get_task(iteration.task_id)
        timestamp = utc_now()

        if decision == Decision.ACCEPT:
            next_status = TaskStatus.ACCEPTED
        elif decision == Decision.REVISE:
            next_status = TaskStatus.REVISION_REQUIRED
        elif decision == Decision.BLOCK:
            next_status = TaskStatus.BLOCKED
        elif decision == Decision.FAIL:
            next_status = TaskStatus.FAILED
        else:
            raise StateError(f"Invalid terminal iteration decision: {decision.value}")

        with self.connection() as connection:
            connection.execute(
                """
                UPDATE iterations
                SET decision = ?, completed_at = ?, failure_reason = ?
                WHERE iteration_id = ?
                """,
                (decision.value, timestamp, failure_reason, iteration_id),
            )
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, completed_at = ?
                WHERE task_id = ?
                """,
                (
                    next_status.value,
                    timestamp if next_status in {
                        TaskStatus.ACCEPTED,
                        TaskStatus.BLOCKED,
                        TaskStatus.FAILED,
                    } else None,
                    task.task_id,
                ),
            )
            if next_status in {TaskStatus.BLOCKED, TaskStatus.FAILED}:
                connection.execute(
                    "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                    (
                        SessionStatus.BLOCKED.value if next_status == TaskStatus.BLOCKED else SessionStatus.FAILED.value,
                        timestamp,
                        task.session_id,
                    ),
                )
            connection.commit()

        self.append_event(
            task.session_id,
            task.task_id,
            iteration_id,
            "ITERATION_COMPLETED",
            {"decision": decision.value, "failure_reason": failure_reason},
        )
        return self.get_iteration(iteration_id)

    def create_checkpoint(
        self,
        task_id: str,
        iteration_id: str,
        commit_sha: str,
        checkpoint_id: str | None = None,
    ) -> Checkpoint:
        """Record an accepted Git checkpoint."""
        task = self.get_task(task_id)
        iteration = self.get_iteration(iteration_id)
        if task.status != TaskStatus.ACCEPTED:
            raise StateError("A checkpoint requires an accepted task")
        if iteration.decision != Decision.ACCEPT:
            raise StateError("A checkpoint requires an accepted iteration")

        checkpoint_id = checkpoint_id or str(uuid.uuid4())
        timestamp = utc_now()

        with self.connection() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO checkpoints (
                        checkpoint_id, task_id, iteration_id, commit_sha, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (checkpoint_id, task_id, iteration_id, commit_sha, timestamp),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise StateError("Checkpoint already recorded") from exc

        self.append_event(
            task.session_id,
            task_id,
            iteration_id,
            "CHECKPOINT_CREATED",
            {"commit_sha": commit_sha},
        )
        return self.get_checkpoint(checkpoint_id)

    def checkpoint_for_iteration(self, iteration_id: str) -> Checkpoint | None:
        """Return the checkpoint recorded for an iteration, if any."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE iteration_id = ?",
                (iteration_id,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            task_id=row["task_id"],
            iteration_id=row["iteration_id"],
            commit_sha=row["commit_sha"],
            created_at=row["created_at"],
        )

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            ).fetchone()
        if row is None:
            raise StateError(f"Unknown checkpoint: {checkpoint_id}")
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            task_id=row["task_id"],
            iteration_id=row["iteration_id"],
            commit_sha=row["commit_sha"],
            created_at=row["created_at"],
        )

    def latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT c.*
                FROM checkpoints c
                JOIN tasks t ON t.task_id = c.task_id
                WHERE t.session_id = ?
                ORDER BY c.created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            task_id=row["task_id"],
            iteration_id=row["iteration_id"],
            commit_sha=row["commit_sha"],
            created_at=row["created_at"],
        )

    def append_event(
        self,
        session_id: str,
        task_id: str | None,
        iteration_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> str:
        """Append an immutable event and return its identifier."""
        event_id = str(uuid.uuid4())
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    event_id, session_id, task_id, iteration_id,
                    event_type, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    task_id,
                    iteration_id,
                    event_type,
                    json.dumps(payload, sort_keys=True),
                    utc_now(),
                ),
            )
            connection.commit()
        return event_id

    def list_events(self, session_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT event_id, task_id, iteration_id, event_type, payload, created_at
                FROM events
                WHERE session_id = ?
                ORDER BY created_at, event_id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "task_id": row["task_id"],
                "iteration_id": row["iteration_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_artifact(
        self,
        session_id: str,
        artifact_type: str,
        path: str,
        sha256: str,
        size_bytes: int,
        task_id: str | None = None,
        iteration_id: str | None = None,
    ) -> str:
        """Record an artifact reference without storing its content in SQLite."""
        artifact_id = str(uuid.uuid4())
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, session_id, task_id, iteration_id,
                    artifact_type, path, sha256, size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    session_id,
                    task_id,
                    iteration_id,
                    artifact_type,
                    path,
                    sha256,
                    size_bytes,
                    utc_now(),
                ),
            )
            connection.commit()
        return artifact_id

    def acquire_lease(
        self,
        workspace_path: str,
        owner_id: str,
        ttl_seconds: float,
    ) -> None:
        """Acquire or refresh a workspace lease if it is available."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")

        now = time.time()
        expires = now + ttl_seconds

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_id, expires_at FROM leases WHERE workspace_path = ?",
                (workspace_path,),
            ).fetchone()
            if row is not None and row["owner_id"] != owner_id and row["expires_at"] > now:
                connection.rollback()
                raise StateError(
                    f"Workspace is locked by active owner {row['owner_id']}"
                )

            connection.execute(
                """
                INSERT INTO leases (
                    workspace_path, owner_id, acquired_at, heartbeat_at, expires_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace_path) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    acquired_at = excluded.acquired_at,
                    heartbeat_at = excluded.heartbeat_at,
                    expires_at = excluded.expires_at
                """,
                (workspace_path, owner_id, now, now, expires),
            )
            connection.commit()

    def heartbeat_lease(self, workspace_path: str, owner_id: str, ttl_seconds: float) -> None:
        """Extend an existing workspace lease."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        now = time.time()
        expires = now + ttl_seconds
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE leases
                SET heartbeat_at = ?, expires_at = ?
                WHERE workspace_path = ?
                  AND owner_id = ?
                  AND expires_at > ?
                """,
                (now, expires, workspace_path, owner_id, now),
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise StateError("Cannot heartbeat a missing or expired lease")

    def release_lease(self, workspace_path: str, owner_id: str) -> None:
        """Release a lease owned by the caller."""
        with self.connection() as connection:
            connection.execute(
                """
                DELETE FROM leases
                WHERE workspace_path = ? AND owner_id = ?
                """,
                (workspace_path, owner_id),
            )
            connection.commit()

    def cleanup_expired_leases(self) -> int:
        """Remove stale leases and return the number deleted."""
        with self.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM leases WHERE expires_at <= ?", (time.time(),)
            )
            connection.commit()
        return cursor.rowcount
