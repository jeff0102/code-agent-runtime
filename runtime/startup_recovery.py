"""Coordinate safe startup recovery for active development sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from runtime.models import Checkpoint, Decision, Iteration, Session, SessionStatus, Task
from runtime.recovery import RecoveryAction, Reconciliation, reconcile_workspace
from runtime.scope import ScopeError, fingerprint_scope
from runtime.state import StateError, StateStore
from runtime.workspace import GitWorkspace, WorkspaceError


class StartupRecoveryOutcome(StrEnum):
    READY_FOR_EXECUTION = "READY_FOR_EXECUTION"
    RESUME_EXECUTION = "RESUME_EXECUTION"
    ALREADY_CHECKPOINTED = "ALREADY_CHECKPOINTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class StartupRecoveryResult:
    """Outcome of recovering one persisted session."""

    session_id: str
    task_id: str | None
    iteration_id: str | None
    outcome: StartupRecoveryOutcome
    reason: str
    reconciliation: Reconciliation | None


class StartupRecovery:
    """Reconcile active sessions without modifying their workspaces."""

    def __init__(self, state: StateStore) -> None:
        self.state = state

    def recover_session(self, session_id: str) -> StartupRecoveryResult:
        """Recover one session and block it when its workspace is unexpected."""
        session = self.state.get_session(session_id)

        try:
            current_scope = fingerprint_scope(session.workspace_path)
        except ScopeError as exc:
            return self._block_scope(session=session, reason=str(exc))

        if current_scope.sha256 != session.scope_hash:
            return self._block_scope(
                session=session,
                reason=(
                    "SCOPE.md has changed since the session started: "
                    f"expected {session.scope_hash}, found {current_scope.sha256}."
                ),
            )

        if session.current_task_id is None:
            return self._record_result(
                session,
                StartupRecoveryResult(
                    session_id=session.session_id,
                    task_id=None,
                    iteration_id=None,
                    outcome=StartupRecoveryOutcome.READY_FOR_EXECUTION,
                    reason="Session has no active task; Supervisor may plan the next task.",
                    reconciliation=None,
                ),
            )

        task = self.state.get_task(session.current_task_id)
        pending_iteration = self.state.get_pending_iteration(task.task_id)
        latest_checkpoint = self.state.latest_checkpoint(session.session_id)
        latest_iteration = self.state.latest_iteration(task.task_id)

        workspace = GitWorkspace(session.workspace_path)

        repaired = self._repair_checkpoint_boundary(
            session=session,
            task=task,
            latest_iteration=latest_iteration,
            latest_checkpoint=latest_checkpoint,
            workspace=workspace,
        )
        if repaired is not None:
            return repaired

        if pending_iteration is not None:
            base_commit = pending_iteration.base_commit
            active_iteration = True
        elif latest_checkpoint is not None:
            base_commit = latest_checkpoint.commit_sha
            active_iteration = False
        else:
            base_commit = workspace.current_commit()
            active_iteration = False

        try:
            reconciliation = reconcile_workspace(
                workspace,
                expected_branch=session.branch,
                base_commit=base_commit,
                latest_checkpoint_commit=latest_checkpoint.commit_sha
                if latest_checkpoint is not None
                else None,
                active_iteration=active_iteration,
            )
        except WorkspaceError as exc:
            return self._block(
                session=session,
                task=task,
                iteration=pending_iteration,
                reason=str(exc),
            )

        outcome = self._map_outcome(
            task=task,
            pending_iteration=pending_iteration,
            latest_checkpoint=latest_checkpoint,
            reconciliation=reconciliation,
        )

        result = StartupRecoveryResult(
            session_id=session.session_id,
            task_id=task.task_id,
            iteration_id=pending_iteration.iteration_id if pending_iteration else None,
            outcome=outcome,
            reason=reconciliation.reason,
            reconciliation=reconciliation,
        )

        if outcome is StartupRecoveryOutcome.BLOCKED:
            return self._block(
                session=session,
                task=task,
                iteration=pending_iteration,
                reason=reconciliation.reason,
            )

        return self._record_result(session, result)

    def _repair_checkpoint_boundary(
        self,
        *,
        session: Session,
        task: Task,
        latest_iteration: Iteration | None,
        latest_checkpoint: Checkpoint | None,
        workspace: GitWorkspace,
    ) -> StartupRecoveryResult | None:
        """Repair a crash between an accepted checkpoint commit and DB persistence."""
        if latest_iteration is None:
            return None

        expected_message = f"runtime-checkpoint:{latest_iteration.iteration_id}"
        current_message = workspace.commit_message()

        if current_message != expected_message:
            return None
        if workspace.status().strip():
            return self._block(
                session=session,
                task=task,
                iteration=latest_iteration,
                reason="Checkpoint candidate commit is not clean.",
            )

        if latest_iteration.decision is not Decision.ACCEPT:
            try:
                self.state.complete_iteration(latest_iteration.iteration_id, Decision.ACCEPT)
            except StateError as exc:
                return self._block(
                    session=session,
                    task=task,
                    iteration=latest_iteration,
                    reason=f"Unable to finalize accepted iteration: {exc}",
                )
            task = self.state.get_task(task.task_id)

        checkpoint = self.state.checkpoint_for_iteration(latest_iteration.iteration_id)
        if checkpoint is None:
            self.state.create_checkpoint(
                task.task_id,
                latest_iteration.iteration_id,
                commit_sha=workspace.current_commit(),
            )
        elif checkpoint.commit_sha != workspace.current_commit():
            return self._block(
                session=session,
                task=task,
                iteration=latest_iteration,
                reason="Persisted checkpoint SHA does not match checkpoint commit.",
            )

        repaired_task = self.state.get_task(task.task_id)
        return self._record_result(
            session,
            StartupRecoveryResult(
                session_id=session.session_id,
                task_id=task.task_id,
                iteration_id=latest_iteration.iteration_id,
                outcome=StartupRecoveryOutcome.ALREADY_CHECKPOINTED,
                reason="Recovered checkpoint commit persisted before the runtime state.",
                reconciliation=None,
            ),
        )

    def recover_active_sessions(self) -> list[StartupRecoveryResult]:
        """Recover all active sessions persisted by the runtime."""
        return [
            self.recover_session(session.session_id)
            for session in self.state.list_active_sessions()
        ]

    def _map_outcome(
        self,
        *,
        task: Task,
        pending_iteration: Iteration | None,
        latest_checkpoint: Checkpoint | None,
        reconciliation: Reconciliation,
    ) -> StartupRecoveryOutcome:
        if reconciliation.action == RecoveryAction.BLOCK_UNEXPECTED:
            return StartupRecoveryOutcome.BLOCKED

        if reconciliation.action == RecoveryAction.RESUME_UNCOMMITTED:
            return StartupRecoveryOutcome.RESUME_EXECUTION

        if reconciliation.action == RecoveryAction.CHECKPOINT_ALREADY_APPLIED:
            return StartupRecoveryOutcome.ALREADY_CHECKPOINTED

        if reconciliation.action == RecoveryAction.CLEAN_EXPECTED_BASE:
            if pending_iteration is not None:
                return StartupRecoveryOutcome.READY_FOR_EXECUTION
            if latest_checkpoint is not None and latest_checkpoint.task_id == task.task_id:
                return StartupRecoveryOutcome.ALREADY_CHECKPOINTED
            return StartupRecoveryOutcome.READY_FOR_EXECUTION

        raise StateError(f"Unhandled recovery action: {reconciliation.action.value}")

    def _block_scope(
        self,
        *,
        session: Session,
        reason: str,
    ) -> StartupRecoveryResult:
        self.state.set_session_status(session.session_id, SessionStatus.BLOCKED)
        self.state.append_event(
            session.session_id,
            None,
            None,
            "SCOPE_INTEGRITY_VIOLATION",
            {"reason": reason, "expected_scope_hash": session.scope_hash},
        )
        return StartupRecoveryResult(
            session_id=session.session_id,
            task_id=session.current_task_id,
            iteration_id=None,
            outcome=StartupRecoveryOutcome.BLOCKED,
            reason=reason,
            reconciliation=None,
        )

    def _block(
        self,
        *,
        session: Session,
        task: Task,
        iteration: Iteration | None,
        reason: str,
    ) -> StartupRecoveryResult:
        self.state.set_session_status(session.session_id, SessionStatus.BLOCKED)
        self.state.append_event(
            session.session_id,
            task.task_id,
            iteration.iteration_id if iteration else None,
            "SESSION_RECOVERY_BLOCKED",
            {"reason": reason},
        )
        return StartupRecoveryResult(
            session_id=session.session_id,
            task_id=task.task_id,
            iteration_id=iteration.iteration_id if iteration else None,
            outcome=StartupRecoveryOutcome.BLOCKED,
            reason=reason,
            reconciliation=None,
        )

    def _record_result(
        self,
        session: Session,
        result: StartupRecoveryResult,
    ) -> StartupRecoveryResult:
        self.state.append_event(
            session.session_id,
            result.task_id,
            result.iteration_id,
            "SESSION_RECOVERY_COMPLETED",
            {
                "outcome": result.outcome.value,
                "reason": result.reason,
            },
        )
        return result
