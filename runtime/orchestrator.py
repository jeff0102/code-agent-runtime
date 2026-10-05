"""Durable Supervisor/Executor orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from runtime.artifacts import ArtifactStore
from runtime.git_integration import GitRemoteIntegration, RemoteIntegrationConfig
from runtime.context import (
    ContextLimits,
    build_executor_context,
    build_git_context,
    build_planner_context,
    build_scope_context,
    build_supervisor_context,
)
from runtime.lease import WorkspaceLease
from runtime.models import Decision, Session, SessionStatus, TaskStatus
from runtime.openhands_executor import OpenHandsExecutionResult
from runtime.openhands_supervisor import (
    OpenHandsSupervisorError,
    OpenHandsSupervisorPlanResult,
    OpenHandsSupervisorResult,
)
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
from runtime.scope import fingerprint_scope


class OrchestrationError(RuntimeError):
    """Raised when an orchestration step cannot safely continue."""


class ExecutorConversationLike(Protocol):
    @property
    def conversation_id(self) -> str: ...

    def send_and_run(self, message: str) -> OpenHandsExecutionResult: ...

    def run(self) -> OpenHandsExecutionResult: ...

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
    max_tasks_per_session: int = 100
    context_limits: ContextLimits = ContextLimits()
    reviewer_workspace: str | Path | None = None
    remote_integration: RemoteIntegrationConfig = RemoteIntegrationConfig()

    def __post_init__(self) -> None:
        if self.lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be greater than zero")
        if self.max_tasks_per_session < 1:
            raise ValueError("max_tasks_per_session must be greater than zero")


@dataclass(frozen=True, slots=True)
class SessionRunResult:
    """Terminal result of an autonomous session."""

    session_id: str
    session_status: SessionStatus
    tasks_completed: int
    checkpoint_sha: str | None = None
    failure_reason: str | None = None


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

    def start_session(
        self,
        *,
        repository: str,
        workspace_path: str | Path,
        max_iterations: int,
        branch: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create an isolated runtime-owned branch and durable session."""
        workspace = GitWorkspace(workspace_path)
        if workspace.status().strip():
            raise OrchestrationError("Cannot start a session from a dirty workspace.")

        scope = fingerprint_scope(workspace.path)
        session_id = session_id or str(uuid4())
        branch = branch or f"agent/{session_id}"
        base_commit = workspace.current_commit()

        workspace.create_branch(branch, base_commit=base_commit)
        try:
            return self.state.create_session(
                repository=repository,
                workspace_path=str(workspace.path),
                branch=branch,
                scope_hash=scope.sha256,
                max_iterations=max_iterations,
                session_id=session_id,
            )
        except Exception:
            try:
                workspace.run("switch", "-")
            except WorkspaceError:
                pass
            raise

    def run_task(
        self,
        session_id: str,
        task_id: str | None = None,
    ) -> TaskRunResult:
        """Recover a session and execute one task to a terminal state."""
        return self._run_task_internal(
            session_id,
            task_id,
            finalize_session=True,
        )

    def run_session(self, session_id: str) -> SessionRunResult:
        """Run task planning and execution until the Supervisor declares DONE."""
        session = self.state.get_session(session_id)
        reviewer_workspace = self._prepare_reviewer_workspace(session_id)
        workspace = GitWorkspace(session.workspace_path)
        with WorkspaceLease(
            self.state,
            session.workspace_path,
            owner_id=f"orchestrator:{session_id}",
            ttl_seconds=self.config.lease_ttl_seconds,
        ):
            self._prepare_target_branch(self.state.get_session(session_id))
            recovery = StartupRecovery(self.state).recover_session(session_id)
            push_recovery_error = self._push_recovered_checkpoint(
                session_id=session_id,
                workspace=workspace,
            )
            if push_recovery_error is not None:
                current = self.state.get_session(session_id)
                return SessionRunResult(
                    session_id=session_id,
                    session_status=current.status,
                    tasks_completed=self._accepted_task_count(session_id),
                    checkpoint_sha=self._latest_checkpoint_sha(session_id),
                    failure_reason=push_recovery_error,
                )
            recovery_outcome = recovery.outcome
            if recovery.outcome is StartupRecoveryOutcome.BLOCKED:
                current = self.state.get_session(session_id)
                return SessionRunResult(
                    session_id=session_id,
                    session_status=current.status,
                    tasks_completed=self._accepted_task_count(session_id),
                    checkpoint_sha=self._latest_checkpoint_sha(session_id),
                    failure_reason=recovery.reason,
                )
            while True:
                session = self.state.get_session(session_id)
                if session.status is not SessionStatus.RUNNING:
                    return SessionRunResult(
                        session_id=session_id,
                        session_status=session.status,
                        tasks_completed=self._accepted_task_count(session_id),
                        checkpoint_sha=self._latest_checkpoint_sha(session_id),
                        failure_reason="Session is not RUNNING.",
                    )

                current_task = self.state.get_task(session.current_task_id) if session.current_task_id else None
                if current_task is not None and current_task.status is not TaskStatus.ACCEPTED:
                    result = self._run_task_locked(
                        session_id=session_id,
                        task_id=current_task.task_id,
                        workspace=workspace,
                        reviewer_workspace=reviewer_workspace,
                        recovery_outcome=recovery_outcome,
                        finalize_session=False,
                    )
                    if result.task_status is not TaskStatus.ACCEPTED:
                        return SessionRunResult(
                            session_id=session_id,
                            session_status=self.state.get_session(session_id).status,
                            tasks_completed=self._accepted_task_count(session_id),
                            checkpoint_sha=self._latest_checkpoint_sha(session_id),
                            failure_reason=result.failure_reason,
                        )

                try:
                    plan = self._plan_next_task(
                        session_id=session_id,
                        workspace=workspace,
                        reviewer_workspace=reviewer_workspace,
                    )
                except Exception as exc:
                    reason = f"Supervisor planning failed: {exc}"
                    self.state.set_session_status(session_id, SessionStatus.FAILED)
                    self.state.append_event(
                        session_id,
                        None,
                        None,
                        "SESSION_PLANNING_FAILED",
                        {"reason": reason},
                    )
                    return SessionRunResult(
                        session_id=session_id,
                        session_status=SessionStatus.FAILED,
                        tasks_completed=self._accepted_task_count(session_id),
                        checkpoint_sha=self._latest_checkpoint_sha(session_id),
                        failure_reason=reason,
                    )
                if plan.action is SupervisorPlanType.DONE:
                    self.state.set_session_status(session_id, SessionStatus.DONE)
                    return SessionRunResult(
                        session_id=session_id,
                        session_status=SessionStatus.DONE,
                        tasks_completed=self._accepted_task_count(session_id),
                        checkpoint_sha=self._latest_checkpoint_sha(session_id),
                    )
                if (
                    plan.action is SupervisorPlanType.NEXT_TASK
                    and self._accepted_task_count(session_id)
                    >= self.config.max_tasks_per_session
                ):
                    reason = (
                        "Maximum tasks per session reached: "
                        f"{self.config.max_tasks_per_session}."
                    )
                    self.state.set_session_status(session_id, SessionStatus.BLOCKED)
                    self.state.append_event(
                        session_id,
                        None,
                        None,
                        "MAX_TASKS_REACHED",
                        {
                            "accepted_tasks": self._accepted_task_count(session_id),
                            "limit": self.config.max_tasks_per_session,
                        },
                    )
                    return SessionRunResult(
                        session_id=session_id,
                        session_status=SessionStatus.BLOCKED,
                        tasks_completed=self._accepted_task_count(session_id),
                        checkpoint_sha=self._latest_checkpoint_sha(session_id),
                        failure_reason=reason,
                    )

                if plan.action is SupervisorPlanType.BLOCK:
                    self.state.set_session_status(session_id, SessionStatus.BLOCKED)
                    self.state.append_event(
                        session_id,
                        None,
                        None,
                        "SESSION_PLANNING_BLOCKED",
                        {"reason": plan.blocking_reason},
                    )
                    return SessionRunResult(
                        session_id=session_id,
                        session_status=SessionStatus.BLOCKED,
                        tasks_completed=self._accepted_task_count(session_id),
                        checkpoint_sha=self._latest_checkpoint_sha(session_id),
                        failure_reason=plan.blocking_reason,
                    )

                sequence = self.state.next_task_sequence(session_id)
                task = self.state.create_task(
                    session_id,
                    sequence=sequence,
                    title=plan.title or "",
                    objective=plan.objective or "",
                    instructions=plan.instructions or "",
                    acceptance_criteria=plan.acceptance_criteria or "",
                )
                self.state.append_event(
                    session_id,
                    task.task_id,
                    None,
                    "TASK_PLANNED",
                    {"sequence": task.sequence, "title": task.title},
                )
                recovery_outcome = StartupRecoveryOutcome.READY_FOR_EXECUTION

    def _run_task_internal(
        self,
        session_id: str,
        task_id: str | None,
        *,
        finalize_session: bool,
    ) -> TaskRunResult:
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
            self._prepare_target_branch(self.state.get_session(session_id))
            recovery = StartupRecovery(self.state).recover_session(session_id)
            push_recovery_error = self._push_recovered_checkpoint(
                session_id=session_id,
                workspace=workspace,
            )
            if push_recovery_error is not None:
                task = self.state.get_task(resolved_task_id)
                return TaskRunResult(
                    session_id=session_id,
                    task_id=resolved_task_id,
                    task_status=task.status,
                    session_status=self.state.get_session(session_id).status,
                    iterations=task.attempt_count,
                    checkpoint_sha=self._latest_checkpoint_sha(session_id),
                    failure_reason=push_recovery_error,
                )
            if recovery.outcome is StartupRecoveryOutcome.BLOCKED:
                session = self.state.get_session(session_id)
                task = self.state.get_task(resolved_task_id)
                return TaskRunResult(
                    session_id=session_id,
                    task_id=resolved_task_id,
                    task_status=task.status,
                    session_status=session.status,
                    iterations=task.attempt_count,
                    failure_reason=recovery.reason,
                )
            return self._run_task_locked(
                session_id=session_id,
                task_id=resolved_task_id,
                workspace=workspace,
                reviewer_workspace=reviewer_workspace,
                recovery_outcome=recovery.outcome,
                finalize_session=finalize_session,
            )

    def _run_task_locked(
        self,
        *,
        session_id: str,
        task_id: str,
        workspace: GitWorkspace,
        reviewer_workspace: Path,
        recovery_outcome: StartupRecoveryOutcome,
        finalize_session: bool = True,
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
            if finalize_session:
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
                persisted_decision = self.state.latest_supervisor_decision(pending.iteration_id)
                if persisted_decision is not None:
                    recovered_decision = SupervisorDecision.from_dict(persisted_decision["decision"])
                    recovered_result = self._apply_recovered_supervisor_decision(
                        session=session,
                        task=task,
                        iteration=pending,
                        workspace=workspace,
                        decision=recovered_decision,
                        finalize_session=finalize_session,
                    )
                    if recovered_decision.decision is SupervisorDecisionType.REVISE:
                        previous_revision_instructions = list(recovered_decision.instructions)
                        previous_decision = recovered_decision
                        recovery_outcome = StartupRecoveryOutcome.READY_FOR_EXECUTION
                        continue
                    if recovered_decision.decision is SupervisorDecisionType.ACCEPT:
                        push_error = self._push_recovered_checkpoint(
                            session_id=session_id,
                            workspace=workspace,
                        )
                        if push_error is not None:
                            return TaskRunResult(
                                session_id=session_id,
                                task_id=task_id,
                                task_status=self.state.get_task(task_id).status,
                                session_status=self.state.get_session(session_id).status,
                                iterations=self.state.get_task(task_id).attempt_count,
                                checkpoint_sha=self._latest_checkpoint_sha(session_id),
                                failure_reason=push_error,
                            )
                    return recovered_result
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
                remote_sync = None
                if self.config.remote_integration.enabled:
                    remote_sync = GitRemoteIntegration(
                        workspace,
                        self.config.remote_integration,
                    ).prepare_session_branch(session.branch)
                    self.state.append_event(
                        session_id,
                        task_id,
                        iteration.iteration_id,
                        "REMOTE_TARGET_SYNCED",
                        {
                            "remote": self.config.remote_integration.remote,
                            "target_branch": self.config.remote_integration.target_branch,
                            "target_sha": remote_sync.target_sha,
                            "conflicts": list(remote_sync.conflicts),
                            "target_integrated": remote_sync.target_integrated,
                        },
                    )
                    synchronized_commit = workspace.current_commit()
                    if synchronized_commit != base_commit:
                        base_commit = synchronized_commit
                        iteration = self.state.update_pending_iteration_base_commit(
                            iteration.iteration_id,
                            base_commit,
                        )
                        self.state.append_event(
                            session_id,
                            task_id,
                            iteration.iteration_id,
                            "ITERATION_BASE_ADVANCED_AFTER_REMOTE_SYNC",
                            {"base_commit": base_commit},
                        )
                executor_base_commit = workspace.current_commit()
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
                executor_prompt = self._executor_prompt(
                    executor_context.to_json(),
                    integration_conflicts=bool(
                        remote_sync and remote_sync.conflicts
                    ),
                )

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
                    if iteration.executor_conversation_id is not None:
                        execution = executor.run()
                    else:
                        execution = executor.send_and_run(executor_prompt)
                finally:
                    executor.close()

                self._record_executor_output_artifacts(
                    session_id=session_id,
                    task_id=task_id,
                    iteration_id=iteration.iteration_id,
                    conversation_id=execution.conversation_id,
                )

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
                workspace.assert_branch(session.branch)
                if workspace.current_commit() != executor_base_commit:
                    raise WorkspaceError(
                        "Executor must not create commits or rewrite Git history; "
                        "the runtime owns checkpoint commits."
                    )

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
                self._record_workspace_artifacts(
                    session_id=session_id,
                    task_id=task_id,
                    iteration_id=iteration.iteration_id,
                    snapshot=snapshot,
                )
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
                    changed_files=executor_report.changed_files,
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
                try:
                    review = supervisor.review(supervisor_prompt)
                except OpenHandsSupervisorError as exc:
                    if exc.raw_response is not None:
                        raw_path, raw_hash, raw_size = self.artifact_store.write_text(
                            session_id,
                            task_id,
                            iteration.iteration_id,
                            "supervisor-decision-response-raw.txt",
                            exc.raw_response,
                        )
                        self.state.record_artifact(
                            session_id,
                            "supervisor_decision_response_raw",
                            raw_path,
                            raw_hash,
                            raw_size,
                            task_id=task_id,
                            iteration_id=iteration.iteration_id,
                        )
                        self.state.append_event(
                            session_id,
                            task_id,
                            iteration.iteration_id,
                            "SUPERVISOR_DECISION_RAW_RESPONSE_SAVED",
                            {
                                "conversation_id": supervisor.conversation_id,
                                "path": raw_path,
                                "sha256": raw_hash,
                                "size_bytes": raw_size,
                            },
                        )
                        raise RuntimeError(
                            f"{exc}; raw response saved to {raw_path}"
                        ) from exc
                    raise
                finally:
                    supervisor.close()

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
                effective_decision = review.decision
                gate_reasons: list[str] = []
                if effective_decision.decision is SupervisorDecisionType.ACCEPT:
                    if not validation.success:
                        gate_reasons.append(
                            "Required deterministic validation did not pass."
                        )
                    if executor_report.status is not ExecutorStatus.COMPLETED:
                        gate_reasons.append(
                            "Executor did not report a completed execution state."
                        )
                    unresolved_conflicts = workspace.unmerged_paths()
                    if unresolved_conflicts:
                        gate_reasons.append(
                            "Git merge conflicts remain unresolved: "
                            + ", ".join(unresolved_conflicts)
                        )
                if gate_reasons:
                    effective_decision = SupervisorDecision(
                        decision=SupervisorDecisionType.REVISE,
                        task_complete=False,
                        instructions=gate_reasons,
                        blocking_reason=None,
                    )
                    self.state.append_event(
                        session_id,
                        task_id,
                        iteration.iteration_id,
                        "SUPERVISOR_ACCEPT_REJECTED_RUNTIME_GATE",
                        {
                            "original_decision": review.decision.to_dict(),
                            "reasons": gate_reasons,
                        },
                    )

                previous_decision = effective_decision
                if effective_decision.decision is SupervisorDecisionType.REVISE:
                    previous_revision_instructions = list(
                        effective_decision.instructions
                    )

                decision = Decision(effective_decision.decision.value)

                self._heartbeat(session.workspace_path, session_id)

                if decision is Decision.ACCEPT:
                    merge_in_progress = (
                        self.config.remote_integration.enabled
                        and GitRemoteIntegration(
                            workspace,
                            self.config.remote_integration,
                        ).merge_in_progress()
                    )
                    target_integrated_for_push = bool(
                        remote_sync
                        and (remote_sync.target_integrated or merge_in_progress)
                    )
                    checkpoint_sha = (
                        workspace.checkpoint(
                            _checkpoint_message(task.title, iteration.iteration_id)
                        )
                        if workspace.status().strip() or merge_in_progress
                        else workspace.current_commit()
                    )
                    if merge_in_progress:
                        self.state.append_event(
                            session_id,
                            task_id,
                            iteration.iteration_id,
                            "REMOTE_MERGE_CONFLICTS_RESOLVED",
                            {"commit_sha": checkpoint_sha},
                        )
                    if self.config.remote_integration.enabled:
                        integration = GitRemoteIntegration(
                            workspace,
                            self.config.remote_integration,
                        )
                        try:
                            integration.push_session_branch(session.branch)
                            checkpoint_sha = workspace.current_commit()
                            self.state.append_event(
                                session_id,
                                task_id,
                                iteration.iteration_id,
                                "REMOTE_TARGET_PUSHED",
                                {
                                    "remote": self.config.remote_integration.remote,
                                    "target_branch": self.config.remote_integration.target_branch,
                                    "commit_sha": checkpoint_sha,
                                },
                            )
                        except WorkspaceError as push_error:
                            latest_sync = integration.prepare_session_branch(
                                session.branch
                            )
                            if (
                                remote_sync is not None
                                and (
                                    latest_sync.target_sha != remote_sync.target_sha
                                    or not target_integrated_for_push
                                )
                            ):
                                revision_instructions = [
                                    "The remote target advanced after review. Integrate "
                                    "its latest changes into this session branch, resolve "
                                    "any conflicts while preserving valid behavior from "
                                    "both sides, then run validations. The Supervisor "
                                    "must review the merged result before another push."
                                ]
                                previous_revision_instructions = revision_instructions
                                previous_decision = SupervisorDecision(
                                    decision=SupervisorDecisionType.REVISE,
                                    task_complete=False,
                                    instructions=revision_instructions,
                                    blocking_reason=None,
                                )
                                self.state.complete_iteration(
                                    iteration.iteration_id,
                                    Decision.REVISE,
                                    failure_reason="Remote target advanced during push.",
                                )
                                self.state.append_event(
                                    session_id,
                                    task_id,
                                    iteration.iteration_id,
                                    "REMOTE_PUSH_REQUIRES_REVIEW",
                                    {
                                        "target_sha": latest_sync.target_sha,
                                        "conflicts": list(latest_sync.conflicts),
                                        "push_error": str(push_error),
                                    },
                                )
                                continue

                            reason = (
                                "Accepted work is checkpointed locally, but pushing "
                                "to the configured remote target failed: "
                                f"{push_error}"
                            )
                            self.state.complete_iteration(
                                iteration.iteration_id,
                                Decision.BLOCK,
                                failure_reason=reason,
                            )
                            self.state.append_event(
                                session_id,
                                task_id,
                                iteration.iteration_id,
                                "REMOTE_PUSH_BLOCKED",
                                {"reason": reason},
                            )
                            return TaskRunResult(
                                session_id=session_id,
                                task_id=task_id,
                                task_status=TaskStatus.BLOCKED,
                                session_status=self.state.get_session(session_id).status,
                                iterations=self.state.get_task(task_id).attempt_count,
                                checkpoint_sha=checkpoint_sha,
                                failure_reason=reason,
                            )
                    self.state.complete_iteration(
                        iteration.iteration_id,
                        Decision.ACCEPT,
                    )
                    self.state.create_checkpoint(
                        task_id,
                        iteration.iteration_id,
                        commit_sha=checkpoint_sha,
                    )
                    if finalize_session:
                        self.state.set_session_status(session_id, SessionStatus.DONE)
                    return TaskRunResult(
                        session_id=session_id,
                        task_id=task_id,
                        task_status=TaskStatus.ACCEPTED,
                        session_status=self.state.get_session(session_id).status,
                        iterations=self.state.get_task(task_id).attempt_count,
                        checkpoint_sha=checkpoint_sha,
                    )

                self.state.complete_iteration(
                    iteration.iteration_id,
                    decision,
                    failure_reason=effective_decision.blocking_reason,
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

    def _plan_next_task(
        self,
        *,
        session_id: str,
        workspace: GitWorkspace,
        reviewer_workspace: Path,
    ) -> SupervisorPlan:
        session = self.state.get_session(session_id)
        pending_plan = self.state.latest_pending_plan(session_id)
        if pending_plan is not None:
            plan = SupervisorPlan.from_dict(pending_plan["plan"])
            self.state.append_event(
                session_id,
                None,
                None,
                "SUPERVISOR_PLANNING_RESUMED",
                {"plan": plan.to_dict()},
            )
            return plan

        scope = build_scope_context(
            session.workspace_path,
            scope_hash=session.scope_hash,
            limits=self.config.context_limits,
        )
        snapshot = workspace.snapshot()
        tasks = self.state.list_tasks(session_id)
        planner_context = build_planner_context(
            repository=session.repository,
            session_id=session_id,
            scope=scope,
            git=build_git_context(
                snapshot,
                base_commit=self._latest_checkpoint_sha(session_id) or snapshot.commit_sha,
                limits=self.config.context_limits,
            ),
            completed_tasks=tasks,
            next_sequence=self.state.next_task_sequence(session_id),
            tracked_files=workspace.tracked_files(),
            untracked_files=workspace.untracked_files(),
            recent_commits=workspace.recent_commits(),
            limits=self.config.context_limits,
        )
        supervisor = self.supervisor_factory.create(
            reviewer_workspace=reviewer_workspace,
            conversation_id=None,
        )
        self.state.append_event(
            session_id,
            None,
            None,
            "SUPERVISOR_PLANNING_STARTED",
            {"conversation_id": supervisor.conversation_id},
        )
        try:
            result = supervisor.plan(self._planner_prompt(planner_context.to_json()))
        except OpenHandsSupervisorError as exc:
            if exc.raw_response is not None:
                raw_path, raw_hash, raw_size = self.artifact_store.write_text(
                    session_id,
                    "planning",
                    supervisor.conversation_id,
                    "supervisor-plan-response-raw.txt",
                    exc.raw_response,
                )
                self.state.record_artifact(
                    session_id,
                    "supervisor_plan_response_raw",
                    raw_path,
                    raw_hash,
                    raw_size,
                )
                self.state.append_event(
                    session_id,
                    None,
                    None,
                    "SUPERVISOR_PLANNING_RAW_RESPONSE_SAVED",
                    {
                        "conversation_id": supervisor.conversation_id,
                        "path": raw_path,
                        "sha256": raw_hash,
                        "size_bytes": raw_size,
                    },
                )
                raise RuntimeError(
                    f"{exc}; raw response saved to {raw_path}"
                ) from exc
            raise
        finally:
            supervisor.close()

        self.state.append_event(
            session_id,
            None,
            None,
            "SUPERVISOR_PLANNING_DECISION",
            {
                "conversation_id": result.conversation_id,
                "next_sequence": self.state.next_task_sequence(session_id),
                "plan": result.plan.to_dict(),
            },
        )
        return result.plan

    @staticmethod
    def _planner_prompt(context_json: str) -> str:
        return (
            "You are the planning Supervisor for an autonomous software development runtime. "
            "Review only the supplied repository evidence. Do not modify files. "
            "Determine the next atomic implementation task required to satisfy SCOPE.md. "
            "Return only one JSON object matching this exact schema, with every key present: "
            "{\"schema_version\":1,\"message_type\":\"supervisor_plan\",\"action\":\"NEXT_TASK\","
            "\"title\":\"...\",\"objective\":\"...\",\"instructions\":\"...\","
            "\"acceptance_criteria\":\"...\",\"blocking_reason\":null}. "
            "Do not include prose, markdown fences, or a wrapper object. "
            "Use NEXT_TASK when actionable work remains, DONE only when the entire scope is satisfied, "
            "and BLOCK when safe progress cannot continue. "
            "For DONE, set title, objective, instructions, acceptance_criteria, and blocking_reason to null. "
            "For BLOCK, set those four task fields to null and provide blocking_reason as a string. "
            "Before selecting work, perform a milestone preflight against SCOPE.md using the repository "
            "inventory, untracked files, recent commits, current Git diff, and completed session tasks. "
            "Identify the earliest incomplete milestone and plan against its remaining acceptance criteria. "
            "Treat the current Git diff as changes since HEAD, not as a list of all implemented files; "
            "never require an existing tracked file to appear again in the diff. Do not repeat work already "
            "present in the repository or accepted in earlier commits. If implementation exists but evidence "
            "does not establish its acceptance criteria, plan a focused verification or gap-analysis task "
            "instead of recreating it. State the milestone and concrete gap in each NEXT_TASK title, objective, "
            "or instructions. NEXT_TASK must be small, independently reviewable, and include concrete "
            "acceptance criteria.\n\n"
            "Planner context (JSON):\n"
            f"{context_json}"
        )

    def _prepare_target_branch(self, session) -> None:
        workspace = GitWorkspace(session.workspace_path)
        current = workspace.current_branch()
        if current == session.branch:
            return

        had_changes = bool(workspace.status().strip())
        stash_created = False
        if had_changes:
            previous_stashes = workspace.run("stash", "list", "--format=%H")
            workspace.run(
                "stash",
                "push",
                "--include-untracked",
                "--message",
                f"runtime-auto-switch:{session.session_id}",
            )
            current_stashes = workspace.run("stash", "list", "--format=%H")
            stash_created = current_stashes != previous_stashes
            if workspace.status().strip():
                raise OrchestrationError(
                    "Could not safely prepare the workspace for a branch switch; "
                    "changes were stashed where possible."
                )

        try:
            if workspace.branch_exists(session.branch):
                workspace.switch_branch(session.branch)
            else:
                workspace.create_branch(session.branch)
            if stash_created:
                workspace.run("stash", "pop", "--index")
        except WorkspaceError as exc:
            if stash_created:
                raise OrchestrationError(
                    f"Could not switch to runtime branch {session.branch!r} and "
                    "restore the saved workspace changes. The changes remain in "
                    "the Git stash; inspect the stash list before retrying. "
                    f"Git reported: {exc}"
                ) from exc
            raise OrchestrationError(
                f"Could not switch to runtime branch {session.branch!r}: {exc}"
            ) from exc

    def _accepted_task_count(self, session_id: str) -> int:
        return sum(
            1
            for task in self.state.list_tasks(session_id)
            if task.status is TaskStatus.ACCEPTED
        )

    def _push_recovered_checkpoint(
        self,
        *,
        session_id: str,
        workspace: GitWorkspace,
    ) -> str | None:
        """Retry publication after a crash between local acceptance and push."""
        if not self.config.remote_integration.enabled:
            return None
        checkpoint = self.state.latest_checkpoint(session_id)
        if checkpoint is None:
            return None
        session = self.state.get_session(session_id)
        if (
            workspace.current_branch() != session.branch
            or workspace.status().strip()
            or workspace.current_commit() != checkpoint.commit_sha
        ):
            return None

        integration = GitRemoteIntegration(
            workspace,
            self.config.remote_integration,
        )
        try:
            integration.push_session_branch(session.branch)
        except WorkspaceError as exc:
            reason = (
                "Could not publish the recovered accepted checkpoint. The remote "
                "target may have advanced; the session is blocked for safe recovery. "
                f"Git reported: {exc}"
            )
            self.state.append_event(
                session_id,
                None,
                None,
                "RECOVERED_CHECKPOINT_PUSH_BLOCKED",
                {"checkpoint_sha": checkpoint.commit_sha, "reason": reason},
            )
            self.state.set_session_status(session_id, SessionStatus.BLOCKED)
            return reason

        self.state.append_event(
            session_id,
            checkpoint.task_id,
            checkpoint.iteration_id,
            "RECOVERED_CHECKPOINT_PUSHED",
            {
                "remote": self.config.remote_integration.remote,
                "target_branch": self.config.remote_integration.target_branch,
                "commit_sha": checkpoint.commit_sha,
            },
        )
        return None

    def _apply_recovered_supervisor_decision(
        self,
        *,
        session,
        task,
        iteration,
        workspace,
        decision: SupervisorDecision,
        finalize_session: bool,
    ) -> TaskRunResult:
        """Apply a persisted Supervisor decision after a process interruption."""
        if decision.decision is SupervisorDecisionType.REVISE:
            self.state.complete_iteration(
                iteration.iteration_id,
                Decision.REVISE,
            )
            return TaskRunResult(
                session_id=session.session_id,
                task_id=task.task_id,
                task_status=TaskStatus.REVISION_REQUIRED,
                session_status=self.state.get_session(session.session_id).status,
                iterations=task.attempt_count,
            )

        if decision.decision is SupervisorDecisionType.BLOCK:
            self.state.complete_iteration(
                iteration.iteration_id,
                Decision.BLOCK,
                failure_reason=decision.blocking_reason,
            )
            return TaskRunResult(
                session_id=session.session_id,
                task_id=task.task_id,
                task_status=TaskStatus.BLOCKED,
                session_status=self.state.get_session(session.session_id).status,
                iterations=self.state.get_task(task.task_id).attempt_count,
                failure_reason=decision.blocking_reason,
            )

        checkpoint = self.state.checkpoint_for_iteration(iteration.iteration_id)
        if checkpoint is None:
            if workspace.status().strip():
                checkpoint_sha = workspace.checkpoint(
                    _checkpoint_message(task.title, iteration.iteration_id)
                )
            else:
                checkpoint_sha = workspace.current_commit()
            self.state.complete_iteration(
                iteration.iteration_id,
                Decision.ACCEPT,
            )
            self.state.create_checkpoint(
                task.task_id,
                iteration.iteration_id,
                commit_sha=checkpoint_sha,
            )
        else:
            checkpoint_sha = checkpoint.commit_sha
            if checkpoint_sha != workspace.current_commit():
                raise OrchestrationError(
                    "Recovered checkpoint SHA does not match the workspace HEAD."
                )
            self.state.complete_iteration(
                iteration.iteration_id,
                Decision.ACCEPT,
            )

        if finalize_session:
            self.state.set_session_status(session.session_id, SessionStatus.DONE)

        return TaskRunResult(
            session_id=session.session_id,
            task_id=task.task_id,
            task_status=TaskStatus.ACCEPTED,
            session_status=self.state.get_session(session.session_id).status,
            iterations=self.state.get_task(task.task_id).attempt_count,
            checkpoint_sha=checkpoint_sha,
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
    def _executor_prompt(
        context_json: str,
        *,
        integration_conflicts: bool = False,
    ) -> str:
        integration_guidance = (
            "The runtime has merged the configured remote target into this isolated "
            "session branch and Git reports unresolved conflicts. Resolve every "
            "conflict in the working tree, preserve valid behavior from both sides, "
            "stage resolved paths with git add, and do not create a commit. The "
            "runtime will finalize the merge commit, run validations, and ask the "
            "Supervisor to review the result.\n\n"
            if integration_conflicts
            else ""
        )
        return (
            "Execute the assigned coding task in the provided workspace. "
            "You are the only agent allowed to modify source files. "
            "Stay within SCOPE.md and AGENTS.md. Implement the task, run the "
            "available validation commands when practical, and stop when the "
            "task is complete or you are blocked.\n\n"
            f"{integration_guidance}"
            "Executor context (JSON):\n"
            f"{context_json}"
        )

    @staticmethod
    def _supervisor_prompt(context_json: str) -> str:
        return (
            "Review the Executor work as a read-only Supervisor. "
            "Use only supplied evidence and do not modify any repository. "
            "Return exactly one JSON object with every field present and this exact schema: "
            "{\"schema_version\":1,\"message_type\":\"supervisor_decision\","
            "\"decision\":\"ACCEPT|REVISE|BLOCK\",\"task_complete\":false,"
            "\"instructions\":[],\"blocking_reason\":null}. "
            "Use the exact message_type value supervisor_decision. "
            "For ACCEPT, set task_complete=true, instructions=[], and blocking_reason=null. "
            "For REVISE, set task_complete=false, provide non-empty instructions, and set blocking_reason=null. "
            "For BLOCK, set task_complete=false, instructions=[], and provide a non-empty blocking_reason. "
            "Do not omit fields or use a summary, rationale, or required_revisions wrapper. "
            "ACCEPT only when the acceptance criteria are satisfied; REVISE with concrete actions otherwise.\n\n"
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

    def _record_workspace_artifacts(
        self,
        *,
        session_id: str,
        task_id: str,
        iteration_id: str,
        snapshot,
    ) -> None:
        for artifact_type, content in (
            ("git.status", snapshot.status),
            ("git.diff", snapshot.diff),
        ):
            path, digest, size = self.artifact_store.write_text(
                session_id,
                task_id,
                iteration_id,
                artifact_type,
                content,
            )
            self.state.record_artifact(
                session_id,
                artifact_type,
                path,
                digest,
                size,
                task_id=task_id,
                iteration_id=iteration_id,
            )

    def _record_executor_output_artifacts(
        self,
        *,
        session_id: str,
        task_id: str,
        iteration_id: str,
        conversation_id: str,
    ) -> None:
        collect = getattr(self.executor_factory, "full_output_artifacts", None)
        if not callable(collect):
            return
        for filename, content in collect(conversation_id):
            output_id = Path(filename).stem.removeprefix("terminal_output_")
            artifact_type = f"terminal-output-{output_id}"
            path, digest, size = self.artifact_store.write_bytes(
                session_id,
                task_id,
                iteration_id,
                artifact_type,
                content,
            )
            self.state.record_artifact(
                session_id,
                artifact_type,
                path,
                digest,
                size,
                task_id=task_id,
                iteration_id=iteration_id,
            )

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


def _checkpoint_message(task_title: str, iteration_id: str) -> str:
    """Keep accepted task intent visible to planners in future sessions."""
    title = " ".join(task_title.split())[:120] or "accepted task"
    return f"runtime-checkpoint: {title} [{iteration_id}]"


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
