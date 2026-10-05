"""Git workspace operations owned by the orchestration runtime."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(RuntimeError):
    """Raised when a Git workspace operation fails."""


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    branch: str
    commit_sha: str
    status: str
    diff: str

    @property
    def dirty(self) -> bool:
        return bool(self.status.strip())


class GitWorkspace:
    """Small, deterministic wrapper around Git for a single workspace."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not (self.path / ".git").exists():
            raise WorkspaceError(f"Not a Git repository: {self.path}")

    def run(self, *args: str) -> str:
        """Run Git and return stdout, raising on non-zero exit."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.path,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise WorkspaceError(
                f"git {' '.join(args)} failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    def current_branch(self) -> str:
        return self.run("branch", "--show-current").strip()

    def current_commit(self) -> str:
        return self.run("rev-parse", "HEAD").strip()

    def branch_exists(self, branch: str) -> bool:
        """Return whether a local branch exists."""
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=self.path,
            check=False,
        )
        return result.returncode == 0

    def switch_branch(self, branch: str) -> None:
        """Switch branches without creating or mutating commits."""
        self.run("switch", branch)

    def commit_message(self, commit_sha: str = "HEAD") -> str:
        """Return the full commit message for a commit."""
        return self.run("log", "-1", "--format=%B", commit_sha).strip()

    def status(self) -> str:
        return self.run("status", "--porcelain=v1")

    def diff(self) -> str:
        """Return the complete working-tree diff, including non-ignored untracked files.

        Git's ordinary ``diff HEAD`` omits untracked files. Stage the worktree into
        a temporary index instead of the repository's real index, then diff that
        index against HEAD. This also leaves the user's staging area untouched.
        """
        with tempfile.TemporaryDirectory(prefix="code-agent-runtime-index-") as tmp:
            index_path = Path(tmp) / "index"
            env = {**os.environ, "GIT_INDEX_FILE": str(index_path)}
            self._run_with_env(env, "read-tree", "HEAD")
            self._run_with_env(env, "add", "--all")
            return self._run_with_env(env, "diff", "--cached", "HEAD", "--binary")

    def _run_with_env(self, env: dict[str, str], *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise WorkspaceError(
                f"git {' '.join(args)} failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    def snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(
            branch=self.current_branch(),
            commit_sha=self.current_commit(),
            status=self.status(),
            diff=self.diff(),
        )

    def assert_branch(self, expected_branch: str) -> None:
        actual = self.current_branch()
        if actual != expected_branch:
            raise WorkspaceError(
                f"Expected branch {expected_branch!r}, found {actual!r}"
            )

    def create_branch(self, branch: str, base_commit: str | None = None) -> None:
        """Create and switch to a new branch."""
        if self.current_branch() != "":
            # Refuse to create from an unexpectedly dirty worktree.
            if self.status().strip():
                raise WorkspaceError("Cannot create a branch with uncommitted changes")
        if base_commit:
            self.run("switch", "-c", branch, base_commit)
        else:
            self.run("switch", "-c", branch)

    def checkpoint(self, message: str) -> str:
        """Commit all current changes and return the resulting SHA.

        The runtime must only call this after validation and Supervisor
        acceptance, and after reconciling the workspace for unexpected changes.
        """
        if not self.status().strip():
            raise WorkspaceError("Cannot create a checkpoint from a clean worktree")
        self.run("add", "--all")
        self.run("commit", "-m", message)
        return self.current_commit()
