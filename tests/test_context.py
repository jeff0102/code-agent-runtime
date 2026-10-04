from pathlib import Path

from runtime.context import (
    ContextError,
    ContextLimits,
    build_executor_context,
    build_git_context,
    build_scope_context,
    build_supervisor_context,
)
from runtime.models import Decision, Task, TaskStatus
from runtime.protocol import ExecutorReport, ExecutorStatus, SupervisorDecision, SupervisorDecisionType
from runtime.validation import ValidationCommandResult, ValidationResult
from runtime.workspace import WorkspaceSnapshot


def make_task() -> Task:
    return Task(
        task_id="task-1",
        session_id="session-1",
        sequence=1,
        title="Implement feature",
        objective="Implement the feature safely.",
        instructions="Add the requested module and tests.",
        acceptance_criteria="All relevant tests pass.",
        status=TaskStatus.EXECUTING,
        attempt_count=1,
        created_at="2026-10-04T00:00:00+00:00",
        completed_at=None,
    )


def make_scope_repo(tmp_path: Path) -> None:
    (tmp_path / "SCOPE.md").write_text("scope content\n", encoding="utf-8")


def make_snapshot(diff: str = "diff") -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        branch="agent/session-1",
        commit_sha="abc123",
        status=" M runtime/foo.py\n",
        diff=diff,
    )


def make_validation() -> ValidationResult:
    return ValidationResult(
        success=True,
        commands=(
            ValidationCommandResult(
                name="tests",
                argv=("pytest",),
                required=True,
                exit_code=0,
                timed_out=False,
                duration_seconds=0.1234567,
                stdout_artifact="/state/tests.stdout.log",
                stderr_artifact="/state/tests.stderr.log",
            ),
        ),
    )


def test_executor_context_is_deterministic_and_includes_revision_instructions(tmp_path):
    make_scope_repo(tmp_path)
    scope = build_scope_context(tmp_path, scope_hash="scope-hash")
    git = build_git_context(make_snapshot(), base_commit="base123")

    context = build_executor_context(
        repository="owner/repo",
        session_id="session-1",
        task=make_task(),
        scope=scope,
        git=git,
        previous_revision_instructions=["Fix the failing parser test."],
    )

    payload = context.to_dict()

    assert payload["context_type"] == "executor"
    assert payload["scope"]["scope_hash"] == "scope-hash"
    assert payload["previous_revision_instructions"] == [
        "Fix the failing parser test."
    ]
    assert context.to_json() == context.to_json()


def test_executor_context_does_not_contain_supervisor_review():
    scope = build_scope_context_for_text(
        "scope",
        scope_hash="hash",
    )
    git = build_git_context(make_snapshot(), base_commit="base")
    context = build_executor_context(
        repository="owner/repo",
        session_id="session-1",
        task=make_task(),
        scope=scope,
        git=git,
        previous_revision_instructions=[],
    )

    serialized = context.to_json()

    assert "supervisor_decision" not in serialized
    assert "executor_report" not in serialized


def test_supervisor_context_contains_validation_and_executor_report():
    scope = build_scope_context_for_text("scope", scope_hash="hash")
    git = build_git_context(make_snapshot(), base_commit="base")

    report = ExecutorReport(
        status=ExecutorStatus.COMPLETED,
        summary="Implemented the feature.",
        changed_files=["runtime/foo.py"],
        tests_executed=["pytest"],
        validation_summary="All checks passed.",
        blockers=[],
    )
    decision = SupervisorDecision(
        decision=SupervisorDecisionType.REVISE,
        task_complete=False,
        instructions=["Fix the parser test."],
        blocking_reason=None,
    )

    context = build_supervisor_context(
        repository="owner/repo",
        session_id="session-1",
        task=make_task(),
        scope=scope,
        git=git,
        validation=make_validation(),
        executor_report=report,
        previous_decision=decision,
    )

    payload = context.to_dict()

    assert payload["context_type"] == "supervisor"
    assert payload["validation"]["commands"][0]["name"] == "tests"
    assert payload["executor_report"]["summary"] == "Implemented the feature."
    assert payload["previous_decision"]["decision"] == "REVISE"


def test_large_diff_is_bounded_with_truncation_marker():
    scope = build_scope_context_for_text("scope", scope_hash="hash")
    snapshot = make_snapshot(diff="A" * 100)
    git = build_git_context(
        snapshot,
        base_commit="base",
        limits=ContextLimits(diff_chars=40),
    )

    assert len(git.diff) <= 40
    assert "[TRUNCATED:" in git.diff
    assert git.diff.startswith("A")
    assert git.diff.endswith("A")


def test_missing_optional_agents_file_is_allowed(tmp_path):
    make_scope_repo(tmp_path)

    scope = build_scope_context(tmp_path, scope_hash="hash")

    assert scope.agents_text is None


def test_missing_scope_file_is_rejected(tmp_path):
    try:
        build_scope_context(tmp_path, scope_hash="hash")
    except ContextError as exc:
        assert "scope file not found" in str(exc)
    else:
        raise AssertionError("Expected missing scope to raise ContextError")


def test_scope_and_agents_text_are_bounded(tmp_path):
    (tmp_path / "SCOPE.md").write_text("S" * 50, encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("A" * 50, encoding="utf-8")

    scope = build_scope_context(
        tmp_path,
        scope_hash="hash",
        limits=ContextLimits(scope_chars=20, agents_chars=20),
    )

    assert len(scope.scope_text) == 20
    assert len(scope.agents_text) == 20


def build_scope_context_for_text(text: str, *, scope_hash: str):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "SCOPE.md").write_text(text, encoding="utf-8")
        return build_scope_context(path, scope_hash=scope_hash)
