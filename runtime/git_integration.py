"""Opt-in fast-forward integration of accepted session branches to a remote."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from runtime.workspace import GitWorkspace, WorkspaceError


@dataclass(frozen=True, slots=True)
class RemoteIntegrationConfig:
    """Remote destination for runtime-owned accepted checkpoints."""

    enabled: bool = False
    remote: str = "origin"
    target_branch: str = "main"

    def __post_init__(self) -> None:
        if not self.remote.strip():
            raise ValueError("remote must not be empty")
        if not self.target_branch.strip():
            raise ValueError("target_branch must not be empty")


@dataclass(frozen=True, slots=True)
class RemoteSyncResult:
    """Fetched target commit and any paths requiring Executor conflict resolution."""

    target_sha: str
    conflicts: tuple[str, ...] = ()
    target_integrated: bool = False


class GitRemoteIntegration:
    """Fetch, merge, and push without force-pushing or switching off the session branch."""

    def __init__(self, workspace: GitWorkspace, config: RemoteIntegrationConfig):
        self.workspace = workspace
        self.config = config

    def prepare_session_branch(self, session_branch: str) -> RemoteSyncResult:
        """Merge the current remote target into the session branch when it advanced."""
        if session_branch == self.config.target_branch:
            raise WorkspaceError(
                "The session branch must be isolated from the remote target branch."
            )
        if self.workspace.current_branch() != session_branch:
            raise WorkspaceError(
                f"Expected session branch {session_branch!r} before remote sync."
            )

        self.workspace.run(
            "fetch", "--no-tags", self.config.remote, self.config.target_branch
        )
        target_sha = self.workspace.run("rev-parse", "FETCH_HEAD").strip()
        conflicts = self.workspace.unmerged_paths()
        merge_in_progress = self.merge_in_progress()
        if conflicts or merge_in_progress:
            return RemoteSyncResult(
                target_sha=target_sha,
                conflicts=tuple(conflicts),
                target_integrated=False,
            )
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "FETCH_HEAD", "HEAD"],
            cwd=self.workspace.path,
            capture_output=True,
            check=False,
        )
        if ancestor.returncode == 0:
            return RemoteSyncResult(target_sha=target_sha, target_integrated=True)
        if ancestor.returncode != 1:
            raise WorkspaceError(
                "git merge-base failed while checking the remote target: "
                f"{ancestor.stderr.decode(errors='replace').strip()}"
            )
        if self.workspace.status().strip():
            # Do not ask Git to merge into a dirty in-progress implementation.
            # The normal push path will detect a stale target and start a fresh,
            # reviewed integration iteration after the current work is accepted.
            return RemoteSyncResult(target_sha=target_sha)

        merge = subprocess.run(
            ["git", "merge", "--no-edit", "FETCH_HEAD"],
            cwd=self.workspace.path,
            text=True,
            capture_output=True,
            check=False,
        )
        conflicts = self.workspace.unmerged_paths()
        if merge.returncode != 0 and not conflicts:
            raise WorkspaceError(
                "git merge of the remote target failed: "
                f"{merge.stderr.strip() or merge.stdout.strip()}"
            )
        return RemoteSyncResult(
            target_sha=target_sha,
            conflicts=tuple(conflicts),
            target_integrated=merge.returncode == 0 and not conflicts,
        )

    def push_session_branch(self, session_branch: str) -> None:
        """Push the session branch to the target using a normal fast-forward push."""
        if self.workspace.current_branch() != session_branch:
            raise WorkspaceError(
                f"Expected session branch {session_branch!r} before push."
            )
        self.workspace.run(
            "push",
            self.config.remote,
            f"HEAD:refs/heads/{self.config.target_branch}",
        )

    def merge_in_progress(self) -> bool:
        return self.workspace.merge_in_progress()
