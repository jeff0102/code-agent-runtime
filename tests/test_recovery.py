import subprocess

from runtime.recovery import RecoveryAction, reconcile_workspace
from runtime.workspace import GitWorkspace


def init_repository(path):
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "runtime@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Runtime Test"], cwd=path, check=True)
    (path / "README.md").write_text("initial\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_dirty_workspace_can_be_resumed_for_active_iteration(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)

    workspace = GitWorkspace(repo)
    base = workspace.current_commit()
    (repo / "README.md").write_text("executor work\n")

    result = reconcile_workspace(
        workspace,
        expected_branch="main",
        base_commit=base,
        latest_checkpoint_commit=None,
        active_iteration=True,
    )

    assert result.action == RecoveryAction.RESUME_UNCOMMITTED
    assert result.snapshot.dirty


def test_unexpected_dirty_workspace_is_blocked(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)

    workspace = GitWorkspace(repo)
    base = workspace.current_commit()
    (repo / "README.md").write_text("manual change\n")

    result = reconcile_workspace(
        workspace,
        expected_branch="main",
        base_commit=base,
        latest_checkpoint_commit=None,
        active_iteration=False,
    )

    assert result.action == RecoveryAction.BLOCK_UNEXPECTED


def test_checkpoint_commit_is_detected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repository(repo)

    workspace = GitWorkspace(repo)
    base = workspace.current_commit()
    (repo / "README.md").write_text("accepted\n")
    checkpoint = workspace.checkpoint("checkpoint: accepted task")

    result = reconcile_workspace(
        workspace,
        expected_branch="main",
        base_commit=base,
        latest_checkpoint_commit=checkpoint,
        active_iteration=False,
    )

    assert result.action == RecoveryAction.CHECKPOINT_ALREADY_APPLIED
