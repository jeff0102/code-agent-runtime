import subprocess

from runtime.models import Decision, SessionStatus, TaskStatus
from runtime.scope import fingerprint_scope
from runtime.startup_recovery import StartupRecovery, StartupRecoveryOutcome
from runtime.state import StateStore
from runtime.workspace import GitWorkspace


def init_repository(path):
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
    (path / "README.md").write_text("initial\n")
    (path / "SCOPE.md").write_text("# Test scope\n\nBuild the test project.\n")
    subprocess.run(["git", "add", "README.md", "SCOPE.md"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def create_session_and_task(store, repo, session_id="session-1"):
    store.create_session(
        repository="owner/repo",
        workspace_path=str(repo),
        branch="main",
        scope_hash=fingerprint_scope(repo).sha256,
        max_iterations=3,
        session_id=session_id,
    )
    return store.create_task(
        session_id,
        sequence=1,
        title="Implement feature",
        objective="Implement the feature.",
        instructions="Write the requested code.",
        acceptance_criteria="All tests pass.",
        task_id=f"{session_id}-task",
    )


def test_clean_workspace_is_ready_for_execution(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)
    store = StateStore(tmp_path / "runtime.db")
    create_session_and_task(store, repo)

    result = StartupRecovery(store).recover_session("session-1")

    assert result.outcome == StartupRecoveryOutcome.READY_FOR_EXECUTION
    assert result.task_id == "session-1-task"
    assert store.get_session("session-1").status == SessionStatus.RUNNING
    assert store.list_events("session-1")[-1]["event_type"] == "SESSION_RECOVERY_COMPLETED"


def test_active_iteration_with_uncommitted_changes_is_resumable(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)
    store = StateStore(tmp_path / "runtime.db")
    create_session_and_task(store, repo)

    base_commit = GitWorkspace(repo).current_commit()
    iteration = store.start_iteration("session-1-task", base_commit=base_commit)
    (repo / "README.md").write_text("executor changes\n")

    result = StartupRecovery(store).recover_session("session-1")

    assert result.outcome == StartupRecoveryOutcome.RESUME_EXECUTION
    assert result.iteration_id == iteration.iteration_id
    assert (repo / "README.md").read_text() == "executor changes\n"


def test_latest_checkpoint_is_detected_as_already_applied(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)
    store = StateStore(tmp_path / "runtime.db")
    create_session_and_task(store, repo)

    workspace = GitWorkspace(repo)
    base_commit = workspace.current_commit()
    iteration = store.start_iteration("session-1-task", base_commit=base_commit)
    (repo / "README.md").write_text("accepted\n")
    store.set_task_status("session-1-task", TaskStatus.REVIEWING)
    store.complete_iteration(iteration.iteration_id, Decision.ACCEPT)
    checkpoint_sha = workspace.checkpoint("checkpoint: accepted task")
    store.create_checkpoint(
        "session-1-task",
        iteration.iteration_id,
        commit_sha=checkpoint_sha,
    )

    result = StartupRecovery(store).recover_session("session-1")

    assert result.outcome == StartupRecoveryOutcome.ALREADY_CHECKPOINTED
    assert result.reconciliation is not None
    assert result.reconciliation.snapshot.commit_sha == checkpoint_sha


def test_unexpected_workspace_changes_block_session(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)
    store = StateStore(tmp_path / "runtime.db")
    create_session_and_task(store, repo)

    (repo / "README.md").write_text("manual change\n")

    result = StartupRecovery(store).recover_session("session-1")

    assert result.outcome == StartupRecoveryOutcome.BLOCKED
    assert store.get_session("session-1").status == SessionStatus.BLOCKED

    events = store.list_events("session-1")
    assert events[-1]["event_type"] == "SESSION_RECOVERY_BLOCKED"



def test_changed_scope_blocks_session(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)
    store = StateStore(tmp_path / "runtime.db")
    create_session_and_task(store, repo)

    (repo / "SCOPE.md").write_text("# Changed scope\n")

    result = StartupRecovery(store).recover_session("session-1")

    assert result.outcome == StartupRecoveryOutcome.BLOCKED
    assert "SCOPE.md has changed" in result.reason
    assert store.get_session("session-1").status == SessionStatus.BLOCKED
    assert store.list_events("session-1")[-1]["event_type"] == "SCOPE_INTEGRITY_VIOLATION"


def test_missing_scope_blocks_session(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)
    store = StateStore(tmp_path / "runtime.db")
    create_session_and_task(store, repo)

    (repo / "SCOPE.md").unlink()

    result = StartupRecovery(store).recover_session("session-1")

    assert result.outcome == StartupRecoveryOutcome.BLOCKED
    assert "scope file not found" in result.reason
    assert store.get_session("session-1").status == SessionStatus.BLOCKED


def test_recovery_repairs_checkpoint_commit_before_db_persistence(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)
    store = StateStore(tmp_path / "runtime.db")
    create_session_and_task(store, repo)

    workspace = GitWorkspace(repo)
    base_commit = workspace.current_commit()
    iteration = store.start_iteration("session-1-task", base_commit=base_commit)
    (repo / "README.md").write_text("accepted
", encoding="utf-8")
    checkpoint_sha = workspace.checkpoint(
        f"runtime-checkpoint:{iteration.iteration_id}"
    )

    result = StartupRecovery(store).recover_session("session-1")

    assert result.outcome == StartupRecoveryOutcome.ALREADY_CHECKPOINTED
    assert store.get_task("session-1-task").status == TaskStatus.ACCEPTED
    checkpoint = store.checkpoint_for_iteration(iteration.iteration_id)
    assert checkpoint is not None
    assert checkpoint.commit_sha == checkpoint_sha


def test_recovery_repairs_clean_accepted_iteration_before_checkpoint_record(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)
    store = StateStore(tmp_path / "runtime.db")
    create_session_and_task(store, repo)

    workspace = GitWorkspace(repo)
    base_commit = workspace.current_commit()
    iteration = store.start_iteration("session-1-task", base_commit=base_commit)
    store.complete_iteration(iteration.iteration_id, Decision.ACCEPT)

    result = StartupRecovery(store).recover_session("session-1")

    assert result.outcome == StartupRecoveryOutcome.ALREADY_CHECKPOINTED
    checkpoint = store.checkpoint_for_iteration(iteration.iteration_id)
    assert checkpoint is not None
    assert checkpoint.commit_sha == base_commit
