import subprocess
import sys
from pathlib import Path

from runtime.artifacts import ArtifactStore
from runtime.models import SessionStatus, TaskStatus
from runtime.openhands_executor import OpenHandsExecutionResult
from runtime.openhands_supervisor import OpenHandsSupervisorResult
from runtime.orchestrator import Orchestrator, OrchestratorConfig
from runtime.protocol import (
    SupervisorDecision,
    SupervisorDecisionType,
    SupervisorPlan,
    SupervisorPlanType,
)
from runtime.scope import fingerprint_scope
from runtime.state import StateStore


def init_repository(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "runtime@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Runtime Test"],
        cwd=path,
        check=True,
    )
    (path / "README.md").write_text("initial\n", encoding="utf-8")
    (path / "SCOPE.md").write_text(
        "# Test scope\n\nBuild the test project.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "README.md", "SCOPE.md"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def create_runtime(tmp_path: Path, *, max_iterations: int = 3, branch: str = "main"):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)

    store = StateStore(tmp_path / "runtime.db")
    store.create_session(
        repository="owner/repo",
        workspace_path=str(repo),
        branch=branch,
        scope_hash=fingerprint_scope(repo).sha256,
        max_iterations=max_iterations,
        session_id="session-1",
    )
    store.create_task(
        "session-1",
        sequence=1,
        title="Implement feature",
        objective="Implement the requested feature.",
        instructions="Write the requested code.",
        acceptance_criteria="The validation and review evidence support acceptance.",
        task_id="task-1",
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    return repo, store, artifacts


class FakeExecutor:
    def __init__(self, repo: Path, conversation_id: str, contents: list[str]):
        self.repo = repo
        self._conversation_id = conversation_id
        self.contents = contents
        self.closed = False
        self.interrupted = False
        self.send_calls = 0
        self.run_calls = 0

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def _execute(self) -> OpenHandsExecutionResult:
        content = self.contents.pop(0)
        (self.repo / "README.md").write_text(content, encoding="utf-8")
        return OpenHandsExecutionResult(
            conversation_id=self._conversation_id,
            execution_status="finished",
        )

    def send_and_run(self, message: str) -> OpenHandsExecutionResult:
        self.send_calls += 1
        return self._execute()

    def run(self) -> OpenHandsExecutionResult:
        self.run_calls += 1
        return self._execute()

    def interrupt(self) -> None:
        self.interrupted = True

    def close(self) -> None:
        self.closed = True


class FakeExecutorFactory:
    def __init__(self, repo: Path, contents: list[str]):
        self.repo = repo
        self.contents = contents
        self.calls: list[str | None] = []

    def create(self, *, workspace_path, conversation_id=None):
        self.calls.append(conversation_id)
        next_id = conversation_id or f"executor-{len(self.calls)}"
        return FakeExecutor(self.repo, next_id, self.contents)


class FakeSupervisor:
    def __init__(
        self,
        conversation_id: str,
        decision: SupervisorDecision | None,
        plan: SupervisorPlan | None,
    ):
        self._conversation_id = conversation_id
        self.decision = decision
        self.plan_value = plan
        self.closed = False

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def review(self, prompt: str) -> OpenHandsSupervisorResult:
        if self.decision is None:
            raise AssertionError("No fake decision configured for this Supervisor call")
        return OpenHandsSupervisorResult(
            conversation_id=self._conversation_id,
            decision=self.decision,
            raw_response=self.decision.to_json(),
        )

    def plan(self, prompt: str):
        from runtime.openhands_supervisor import OpenHandsSupervisorPlanResult

        if self.plan_value is None:
            raise AssertionError("No fake plan configured for this Supervisor call")
        return OpenHandsSupervisorPlanResult(
            conversation_id=self._conversation_id,
            plan=self.plan_value,
            raw_response=self.plan_value.to_json(),
        )

    def interrupt(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeSupervisorFactory:
    def __init__(
        self,
        decisions: list[SupervisorDecision],
        plans: list[SupervisorPlan] | None = None,
    ):
        self.decisions = decisions
        self.plans = plans or []
        self.calls: list[str | None] = []

    def create(self, *, reviewer_workspace, conversation_id=None):
        self.calls.append(conversation_id)
        decision = self.decisions.pop(0) if self.decisions else None
        plan = self.plans.pop(0) if self.plans else None
        next_id = conversation_id or f"supervisor-{len(self.calls)}"
        return FakeSupervisor(next_id, decision, plan)


def accept_decision() -> SupervisorDecision:
    return SupervisorDecision(
        decision=SupervisorDecisionType.ACCEPT,
        task_complete=True,
        instructions=[],
        blocking_reason=None,
    )


def revise_decision() -> SupervisorDecision:
    return SupervisorDecision(
        decision=SupervisorDecisionType.REVISE,
        task_complete=False,
        instructions=["Fix the implementation detail identified by review."],
        blocking_reason=None,
    )


def block_decision() -> SupervisorDecision:
    return SupervisorDecision(
        decision=SupervisorDecisionType.BLOCK,
        task_complete=False,
        instructions=[],
        blocking_reason="Required repository dependency is unavailable.",
    )


def test_orchestrator_accepts_and_checkpoints(tmp_path):
    repo, store, artifacts = create_runtime(tmp_path)
    executor_factory = FakeExecutorFactory(repo, ["accepted\n"])
    supervisor_factory = FakeSupervisorFactory([accept_decision()])

    result = Orchestrator(
        store,
        artifacts,
        executor_factory,
        supervisor_factory,
        OrchestratorConfig(),
    ).run_task("session-1")

    assert result.task_status is TaskStatus.ACCEPTED
    assert result.session_status is SessionStatus.DONE
    assert result.iterations == 1
    assert result.checkpoint_sha is not None
    assert store.latest_checkpoint("session-1").commit_sha == result.checkpoint_sha
    assert executor_factory.calls == [None]
    assert supervisor_factory.calls == [None]

    events = [event["event_type"] for event in store.list_events("session-1")]
    assert events.count("EXECUTOR_STARTED") == 1
    assert events.count("VALIDATION_COMPLETED") == 1
    assert events.count("SUPERVISOR_DECISION") == 1


def test_orchestrator_revises_until_acceptance(tmp_path):
    repo, store, artifacts = create_runtime(tmp_path, max_iterations=3)
    executor_factory = FakeExecutorFactory(repo, ["first\n", "accepted\n"])
    supervisor_factory = FakeSupervisorFactory([revise_decision(), accept_decision()])

    result = Orchestrator(
        store,
        artifacts,
        executor_factory,
        supervisor_factory,
        OrchestratorConfig(),
    ).run_task("session-1")

    assert result.task_status is TaskStatus.ACCEPTED
    assert result.session_status is SessionStatus.DONE
    assert result.iterations == 2
    assert executor_factory.calls == [None, None]
    assert len(supervisor_factory.calls) == 2
    assert store.get_task("task-1").attempt_count == 2
    assert len([e for e in store.list_events("session-1") if e["event_type"] == "ITERATION_COMPLETED"]) == 2


def test_orchestrator_blocks_on_supervisor_block(tmp_path):
    repo, store, artifacts = create_runtime(tmp_path)
    executor_factory = FakeExecutorFactory(repo, ["blocked\n"])
    supervisor_factory = FakeSupervisorFactory([block_decision()])

    result = Orchestrator(
        store,
        artifacts,
        executor_factory,
        supervisor_factory,
        OrchestratorConfig(),
    ).run_task("session-1")

    assert result.task_status is TaskStatus.BLOCKED
    assert result.session_status is SessionStatus.BLOCKED
    assert "dependency" in (result.failure_reason or "").lower()


def test_orchestrator_resumes_pending_executor_conversation(tmp_path):
    repo, store, artifacts = create_runtime(tmp_path)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    iteration = store.start_iteration(
        "task-1",
        base_commit=base_commit,
        executor_conversation_id="executor-existing",
    )
    (repo / "README.md").write_text("resume\n", encoding="utf-8")

    executor_factory = FakeExecutorFactory(repo, ["resumed\n"])
    supervisor_factory = FakeSupervisorFactory([accept_decision()])

    result = Orchestrator(
        store,
        artifacts,
        executor_factory,
        supervisor_factory,
        OrchestratorConfig(),
    ).run_task("session-1")

    assert result.task_status is TaskStatus.ACCEPTED
    assert executor_factory.calls == ["executor-existing"]
    assert store.get_iteration(iteration.iteration_id).executor_conversation_id == "executor-existing"


def test_validation_command_nul_byte_is_rejected():
    from runtime.validation import ValidationCommand, ValidationError

    try:
        ValidationCommand("bad", (sys.executable, "-c", "print('\x00')"))
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected an actual NUL byte to be rejected")


def next_task_plan(title: str) -> SupervisorPlan:
    return SupervisorPlan(
        action=SupervisorPlanType.NEXT_TASK,
        title=title,
        objective=f"Implement {title}.",
        instructions=f"Implement {title} within the scope.",
        acceptance_criteria=f"{title} is implemented and validated.",
        blocking_reason=None,
    )


def done_plan() -> SupervisorPlan:
    return SupervisorPlan(
        action=SupervisorPlanType.DONE,
        title=None,
        objective=None,
        instructions=None,
        acceptance_criteria=None,
        blocking_reason=None,
    )


def test_orchestrator_runs_full_session_until_supervisor_done(tmp_path):
    repo, store, artifacts = create_runtime(tmp_path)
    executor_factory = FakeExecutorFactory(repo, ["first task\\n", "second task\\n"])
    supervisor_factory = FakeSupervisorFactory(
        decisions=[accept_decision(), accept_decision()],
        plans=[next_task_plan("Second task"), done_plan()],
    )

    result = Orchestrator(
        store,
        artifacts,
        executor_factory,
        supervisor_factory,
        OrchestratorConfig(),
    ).run_session("session-1")

    assert result.session_status is SessionStatus.DONE
    assert result.tasks_completed == 2
    assert store.get_session("session-1").status is SessionStatus.DONE
    assert [task.status for task in store.list_tasks("session-1")] == [
        TaskStatus.ACCEPTED,
        TaskStatus.ACCEPTED,
    ]


def test_orchestrator_creates_missing_runtime_branch(tmp_path):
    repo, store, artifacts = create_runtime(tmp_path, branch="agent/session-1")
    executor_factory = FakeExecutorFactory(repo, ["accepted\n"])
    supervisor_factory = FakeSupervisorFactory([accept_decision()])

    result = Orchestrator(
        store,
        artifacts,
        executor_factory,
        supervisor_factory,
        OrchestratorConfig(),
    ).run_task("session-1")

    assert result.task_status is TaskStatus.ACCEPTED
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert branch == "agent/session-1"
