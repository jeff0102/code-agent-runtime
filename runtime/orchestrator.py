"""Durable Supervisor/Executor orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from runtime.artifacts import ArtifactStore
from runtime.context import (
    ContextLimits,
    build_executor_context,
    build_git_context,
    build_planner_context,
    build_scope_context,
    build_supervisor_context,
)
from runtime.lease import WorkspaceLease
from runtime.models import Decision, SessionStatus, TaskStatus
from runtime.openhands_executor import OpenHandsExecutionResult
from runtime.openhands_supervisor import OpenHandsSupervisorPlanResult, OpenHandsSupervisorResult
from runtime.protocol import (
    ExecutorReport,
    ExecutorStatus,
    SupervisorDecision,
    SupervisorDecisionType,
    SupervisorPlan,
    SupervisorPlanType,
)
from runtime.startup_recovery import StartupRecovery, StartupRecoveryOutcome
from runtime.state import StateError, StateStore
from runtime.validation import ValidationCommand, ValidationResult, ValidationRunner
from runtime.workspace import GitWorkspace, WorkspaceError


class OrchestrationError(RuntimeError):
    """Raised when an orchestration step cannot safely continue."""


class ExecutorConversationLike(Protocol):
    @property
    def conversation_id(self) -> str: ...

    def send_and_run(self, message: str) -> OpenHandsExecutionResult: ...

    def interrupt(self) -> None: ...

    def close(self) -> None: ...


class ExecutorFactoryLike(Protocol):
    def create(
        self,
        *,
        workspace_path: str | Path,
        conversation_id: str | None = None,
    ) -> ExecutorConversationLike: ...


class SupervisorConversationLike(Protocol):
    @property
    def conversation_id(self) -> str: ...

    def review(self, prompt: str) -> OpenHandsSupervisorResult: ...

    def plan(self, prompt: str) -> OpenHandsSupervisorPlanResult: ...

    def interrupt(self) -> None: ...

    def close(self) -> None: ...


class SupervisorFactoryLike(Protocol):
    def create(
        self,
        *,
        reviewer_workspace: str | Path,
        conversation_id: str | None = None,
    ) -> SupervisorConversationLike: ...


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Runtime controls for one autonomous task loop."""

    validation_commands: tuple[ValidationCommand, ...] = ()
    lease_ttl_seconds: float = 3600.0
    context_limits: ContextLimits = ContextLimits()
    reviewer_workspace: str | Path | None = None

    def __post_init__(self) -> None:
        if self.lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be greater than zero")


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    """Terminal result of one task execution loop."""

    session_id: str
    task_id: str
    task_status: TaskStatus
    session_status: SessionStatus
    iterations: int
    checkpoint_sha: str | None = None
    failure_reason: str | None = None


class Orchestrator:
    """Run one durable Executor -> validation -> Supervisor loop."""

    def __init__(
        self,
        state: StateStore,
        artifact_store: ArtifactStore,
        executor_factory: ExecutorFactoryLike,
        supervisor_factory: SupervisorFactoryLike,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.state = state
        self.artifact_store = artifact_store
        self.executor_factory = executor_factory
        self.supervisor_factory = supervisor_factory
        self.config = config or OrchestratorConfig()

    def run_task(
        self,
        session_id: str,
        task_id: str | None = None,
    ) -> TaskRunResult:
        """Recover a session and execute its current task to a terminal state."""
        recovery = StartupRecovery(self.state).recover_session(session_id)
        if recovery.outcome is StartupRecoveryOutcome.BLOCKED:
            session = self.state.get_session(session_id)
            resolved_task_id = task_id or session.current_task_id
            if resolved_task_id is None:
                raise OrchestrationError(
                    f"Session {session_id} is blocked and has no current task."
                )
            task = self.state.get_task(resolved_task_id)
            return TaskRunResult(
                session_id=session_id,
                task_id=resolved_task_id,
                task_status=task.status,
                session_status=session.status,
                iterations=task.attempt_count,
                failure_reason=recovery.reason,
            )

        session = self.state.get_session(session_id)
        resolved_task_id = task_id or session.current_task_id
        if resolved_task_id is None:
            raise OrchestrationError(
                "Session has no current task. Task planning is intentionally "
                "kept outside this first orchestration core."
            )

        if self.state.get_task(resolved_task_id).session_id != session_id:
            raise OrchestrationError("Task does not belong to the requested session.")

        reviewer_workspace = self._prepare_reviewer_workspace(session_id)
        workspace = GitWorkspace(session.workspace_path)

        with WorkspaceLease(
            self.state,
            session.workspace_path,
            owner_id=f"orchestrator:{session_id}",
            ttl_seconds=self.config.lease_ttl_seconds,
        ):
            return self._run_locked(
                session_id=session_id,
                task_id=resolved_task_id,
                workspace=workspace,
                reviewer_workspace=reviewer_workspace,
                recovery_outcome=recovery.outcome,
            )

    def _run_locked(
        self,
        *,
        session_id: str,
        task_id: str,
        workspace: GitWorkspace,
        reviewer_workspace: Path,
        recovery_outcome: StartupRecoveryOutcome,
    ) -> TaskRunResult:
        session = self.state.get_session(session_id)
        task = self.state.get_task(task_id)

        if session.status is not SessionStatus.RUNNING:
            return TaskRunResult(
                session_id=session_id,
                task_id=task_id,
                task_status=task.status,
                session_status=session.status,
                iterations=task.attempt_count,
                failure_reason="Session is not RUNNING.",
            )

        if task.status is TaskStatus.ACCEPTED:
            self.state.set_session_status(session_id, SessionStatus.DONE)
            return TaskRunResult(
                session_id=session_id,
                task_id=task_id,
                task_status=TaskStatus.ACCEPTED,
                session_status=SessionStatus.DONE,
                iterations=task.attempt_count,
                checkpoint_sha=self._latest_checkpoint_sha(session_id),
            )

        if task.status in {TaskStatus.BLOCKED, TaskStatus.FAILED}:
            return TaskRunResult(
                session_id=session_id,
                task_id=task_id,
                task_status=task.status,
                session_status=self.state.get_session(session_id).status,
                iterations=task.attempt_count,
                failure_reason="Task is already terminal.",
            )

        previous_revision_instructions: list[str] = []
        previous_decision: SupervisorDecision | None = None

        while True:
            session = self.state.get_session(session_id)
            task = self.state.get_task(task_id)

            if session.status is not SessionStatus.RUNNING:
                return TaskRunResult(
                    session_id=session_id,
                    task_id=task_id,
                    task_status=task.status,
                    session_status=session.status,
                    iterations=task.attempt_count,
                    checkpoint_sha=self._latest_checkpoint_sha(session_id),
                )

            pending = self.state.get_pending_iteration(task_id)
            if pending is not None:
                iteration = pending
                base_commit = pending.base_commit
                self.state.append_event(
                    session_id,
                    task_id,
                    iteration.iteration_id,
                    "ITERATION_RESUMED",
                    {
                        "attempt_number": iteration.attempt_number,
                        "recovery_outcome": recovery_outcome.value,
                    },
                )
            else:
                if task.attempt_count >= session.max_iterations:
                    return self._block_max_iterations(
                        session_id=session_id,
                        task_id=task_id,
                        attempts=task.attempt_count,
                    )

                base_commit = workspace.current_commit()
                iteration = self.state.start_iteration(
                    task_id,
                    base_commit=base_commit,
                )

            try:
                workspace.assert_branch(session.branch)
                scope = build_scope_context(
                    session.workspace_path,
                    scope_hash=session.scope_hash,
                    limits=self.config.context_limits,
                )
                snapshot = workspace.snapshot()
                executor_context = build_executor_context(
                    repository=session.repository,
                    session_id=session_id,
                    task=task,
                    scope=scope,
                    git=build_git_context(
                        snapshot,
                        base_commit=base_commit,
                        limits=self.config.context_limits,
                    ),
                    previous_revision_instructions=previous_revision_instructions,
                    limits=self.config.context_limits,
                )
                executor_prompt = self._executor_prompt(executor_context.to_json())

                executor = self.executor_factory.create(
                    workspace_path=session.workspace_path,
                    conversation_id=iteration.executor_conversation_id,
                )
                self.state.set_iteration_conversations(
                    iteration.iteration_id,
                    executor.conversation_id,
                    iteration.supervisor_conversation_id,
                )
                self.state.append_event(
                    session_id,
                    task_id,
                    iteration.iteration_id,
                    "EXECUTOR_STARTED",
                    {"conversation_id": executor.conversation_id},
                )

                try:
                    execution = executor.send_and_run(executor_prompt)
                finally:
                    executor.close()

                self.state.set_iteration_conversations(
                    iteration.iteration_id,
                    execution.conversation_id,
                    iteration.supervisor_conversation_id,
                )
                self.state.append_event(
                    session_id,
                    task_id,
                    iteration.iteration_id,
                    "EXECUTOR_COMPLETED",
                    {
                        "conversation_id": execution.conversation_id,
                        "execution_status": execution.execution_status,
                        "error": execution.error,
                    },
                )

                if execution.error is not None:
                    return self._fail_iteration(
                        session_id=session_id,
                        task_id=task_id,
                        iteration_id=iteration.iteration_id,
                        reason=execution.error,
                    )

                self._heartbeat(session.workspace_path, session_id)

                validation = ValidationRunner(
                    session.workspace_path,
                    self.artifact_store,
                ).run(
                    self.config.validation_commands,
                    session_id=session_id,
                    task_id=task_id,
                    iteration_id=iteration.iteration_id,
                )
                validation_summary = self._validation_summary(validation)
                self.state.append_event(
                    session_id,
                    task_id,
                    iteration.iteration_id,
                    "VALIDATION_COMPLETED",
                    {
                        "success": validation.success,
                        "summary": validation_summary,
                    },
                )

                snapshot = workspace.snapshot()
                executor_report = self._build_executor_report(
                    snapshot=snapshot,
                    execution_status=execution.execution_status,
                    validation_success=validation.success,
                    validation_summary=validation_summary,
                )
                report_path, report_hash, report_size = self.artifact_store.write_text(
                    session_id,
                    task_id,
                    iteration.iteration_id,
                    "executor-report.json",
                    executor_report.to_json(),
                )
                self.state.record_artifact(
                    session_id,
                    "executor-report",
                    report_path,
                    report_hash,
                    report_size,
                    task_id=task_id,
                    iteration_id=iteration.iteration_id,
                )

                self.state.set_task_status(task_id, TaskStatus.REVIEWING)
                supervisor_scope = build_scope_context(
                    session.workspace_path,
                    scope_hash=session.scope_hash,
                    limits=self.config.context_limits,
                )
                supervisor_context = build_supervisor_context(
                    repository=session.repository,
                    session_id=session_id,
                    task=task,
                    scope=supervisor_scope,
                    git=build_git_context(
                        snapshot,
                        base_commit=base_commit,
                        limits=self.config.context_limits,
                    ),
                    validation=validation,
                    executor_report=executor_report,
                    previous_decision=previous_decision,
                    limits=self.config.context_limits,
                )

                supervisor = self.supervisor_factory.create(
                    reviewer_workspace=reviewer_workspace,
                    conversation_id=iteration.supervisor_conversation_id,
                )
                self.state.set_iteration_conversations(
                    iteration.iteration_id,
                    execution.conversation_id,
                    supervisor.conversation_id,
                )
                supervisor_prompt = self._supervisor_prompt(
                    supervisor_context.to_json()
                )
                self.state.append_event(
                    session_id,
                    task_id,
                    iteration.iteration_id,
                    "SUPERVISOR_STARTED",
                    {"conversation_id": supervisor.conversation_id},
                )
                review = supervisor.review(supervisor_prompt)
                self.state.set_iteration_conversations(
                    iteration.iteration_id,
                    execution.conversation_id,
                    review.conversation_id,
                )
                self.state.append_event(
                    session_id,
                    task_id,
                    iteration.iteration_id,
                    "SUPERVISOR_DECISION",
                    {
                        "conversation_id": review.conversation_id,
                        "decision": review.decision.to_dict(),
                    },
                )
                review_path, review_hash, review_size = self.artifact_store.write_text(
                    session_id,
                    task_id,
                    iteration.iteration_id,
                    "supervisor-review.json",
                    review.decision.to_json(),
                )
                self.state.record_artifact(
                    session_id,
                    "supervisor-review",
                    review_path,
                    review_hash,
                    review_size,
                    task_id=task_id,
                    iteration_id=iteration.iteration_id,
                )
                supervisor.close()

                previous_decision = review.decision
                if review.decision.decision is SupervisorDecisionType.REVISE:
                    previous_revision_instructions = list(review.decision.instructions)

                decision = Decision(review.decision.decision.value)
                self.state.complete_iteration(
                    iteration.iteration_id,
                    decision,
                    failure_reason=review.decision.blocking_reason,
                )

                self._heartbeat(session.workspace_path, session_id)

                if decision is Decision.ACCEPT:
                    checkpoint_sha = (
                        workspace.checkpoint(
                            f"checkpoint: task {task.sequence} {task.title}"
                        )
                        if workspace.status().strip()
                        else workspace.current_commit()
                    )
                    self.state.create_checkpoint(
                        task_id,
                        iteration.iteration_id,
                        commit_sha=checkpoint_sha,
                    )
                    self.state.set_session_status(session_id, SessionStatus.DONE)
                    return TaskRunResult(
                        session_id=session_id,
                        task_id=task_id,
                        task_status=TaskStatus.ACCEPTED,
                        session_status=SessionStatus.DONE,
                        iterations=self.state.get_task(task_id).attempt_count,
                        checkpoint_sha=checkpoint_sha,
                    )

                if decision is Decision.BLOCK:
                    return TaskRunResult(
                        session_id=session_id,
                        task_id=task_id,
                        task_status=TaskStatus.BLOCKED,
                        session_status=self.state.get_session(session_id).status,
                        iterations=self.state.get_task(task_id).attempt_count,
                        failure_reason=review.decision.blocking_reason,
                    )

                if self.state.get_task(task_id).attempt_count >= session.max_iterations:
                    return self._block_max_iterations(
                        session_id=session_id,
                        task_id=task_id,
                        attempts=self.state.get_task(task_id).attempt_count,
                    )

                task = self.state.get_task(task_id)

            except (StateError, WorkspaceError, OSError, ValueError) as exc:
                return self._fail_iteration(
                    session_id=session_id,
                    task_id=task_id,
                    iteration_id=iteration.iteration_id,
                    reason=str(exc),
                )
            except Exception as exc:  # noqa: BLE001
                return self._fail_iteration(
                    session_id=session_id,
                    task_id=task_id,
                    iteration_id=iteration.iteration_id,
                    reason=f"Unexpected orchestration error: {exc}",
                )

    def _prepare_reviewer_workspace(self, session_id: str) -> Path:
        workspace = (
            Path(self.config.reviewer_workspace)
            if self.config.reviewer_workspace is not None
            else self.state.database_path.parent / "reviewer" / session_id
        )
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _heartbeat(self, workspace_path: str, session_id: str) -> None:
        try:
            self.state.heartbeat_lease(
                workspace_path,
                f"orchestrator:{session_id}",
                self.config.lease_ttl_seconds,
            )
        except StateError:
            pass

    def _fail_iteration(
        self,
        *,
        session_id: str,
        task_id: str,
        iteration_id: str,
        reason: str,
    ) -> TaskRunResult:
        try:
            self.state.complete_iteration(
                iteration_id,
                Decision.FAIL,
                failure_reason=reason,
            )
        except StateError:
            session = self.state.get_session(session_id)
            task = self.state.get_task(task_id)
            return TaskRunResult(
                session_id=session_id,
                task_id=task_id,
                task_status=task.status,
                session_status=session.status,
                iterations=task.attempt_count,
                failure_reason=reason,
            )

        session = self.state.get_session(session_id)
        task = self.state.get_task(task_id)
        return TaskRunResult(
            session_id=session_id,
            task_id=task_id,
            task_status=task.status,
            session_status=session.status,
            iterations=task.attempt_count,
            failure_reason=reason,
        )

    def _block_max_iterations(
        self,
        *,
        session_id: str,
        task_id: str,
        attempts: int,
    ) -> TaskRunResult:
        task = self.state.get_task(task_id)
        session = self.state.get_session(session_id)
        if task.status is not TaskStatus.BLOCKED:
            self.state.set_task_status(task_id, TaskStatus.BLOCKED)
        if session.status is not SessionStatus.BLOCKED:
            self.state.set_session_status(session_id, SessionStatus.BLOCKED)
        reason = f"Maximum iterations reached: {attempts}."
        self.state.append_event(
            session_id,
            task_id,
            None,
            "MAX_ITERATIONS_REACHED",
            {"attempts": attempts, "limit": session.max_iterations},
        )
        return TaskRunResult(
            session_id=session_id,
            task_id=task_id,
            task_status=TaskStatus.BLOCKED,
            session_status=SessionStatus.BLOCKED,
            iterations=attempts,
            failure_reason=reason,
        )

    def _latest_checkpoint_sha(self, session_id: str) -> str | None:
        checkpoint = self.state.latest_checkpoint(session_id)
        return checkpoint.commit_sha if checkpoint is not None else None

    @staticmethod
    def _executor_prompt(context_json: str) -> str:
        return (
            "Execute the assigned coding task in the provided workspace. "
            "You are the only agent allowed to modify source files. "
            "Stay within SCOPE.md and AGENTS.md. Implement the task, run the "
            "available validation commands when practical, and stop when the "
            "task is complete or you are blocked.\n\n"
            "Executor context (JSON):\n"
            f"{context_json}"
        )

    @staticmethod
    def _supervisor_prompt(context_json: str) -> str:
        return (
            "Review the Executor work as a read-only Supervisor. "
            "Use only the supplied evidence and do not modify any repository. "
            "Return exactly one SupervisorDecision JSON object. "
            "ACCEPT only when the acceptance criteria are satisfied; "
            "REVISE with concrete instructions when more work is required; "
            "BLOCK only when safe progress cannot continue.\n\n"
            "Supervisor context (JSON):\n"
            f"{context_json}"
        )

    @staticmethod
    def _validation_summary(validation: ValidationResult) -> str:
        if not validation.commands:
            return "No validation commands configured."
        parts = []
        for result in validation.commands:
            state = "passed" if result.success else "failed"
            if result.timed_out:
                state = "timed out"
            parts.append(f"{result.name}: {state} (exit={result.exit_code})")
        return "; ".join(parts)

    @staticmethod
    def _build_executor_report(
        *,
        snapshot,
        execution_status: str,
        validation_success: bool,
        validation_summary: str,
    ) -> ExecutorReport:
        changed_files = _parse_changed_files(snapshot.status)
        tests = [
            line.split(":", 1)[0].strip()
            for line in validation_summary.split("; ")
            if line.strip()
        ]
        blockers: list[str] = []
        status = ExecutorStatus.COMPLETED
        normalized_execution_status = execution_status.upper()
        if normalized_execution_status not in {"COMPLETED", "FINISHED", "SUCCESS", "DONE"}:
            if normalized_execution_status == "BLOCKED":
                status = ExecutorStatus.BLOCKED
            else:
                status = ExecutorStatus.FAILED
            blockers.append(f"OpenHands execution status: {execution_status}.")
        if not validation_success and status is ExecutorStatus.COMPLETED:
            blockers.append("Deterministic validation did not fully pass.")
        if blockers and status is ExecutorStatus.COMPLETED:
            status = ExecutorStatus.BLOCKED

        return ExecutorReport(
            status=status,
            summary=(
                "OpenHands Executor completed the run; runtime generated the "
                "report from execution state and workspace evidence."
            ),
            changed_files=changed_files,
            tests_executed=tests,
            validation_summary=validation_summary,
            blockers=blockers,
        )


def _parse_changed_files(status: str) -> list[str]:
    files: list[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[-1]
        if path and path not in files:
            files.append(path)
    return files
