"""Workspace reconciliation and crash-recovery decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from runtime.workspace import GitWorkspace, WorkspaceSnapshot


class RecoveryAction(StrEnum):
    CLEAN_EXPECTED_BASE = "CLEAN_EXPECTED_BASE"
    RESUME_UNCOMMITTED = "RESUME_UNCOMMITTED"
    CHECKPOINT_ALREADY_APPLIED = "CHECKPOINT_ALREADY_APPLIED"
    BLOCK_UNEXPECTED = "BLOCK_UNEXPECTED"


@dataclass(frozen=True, slots=True)
class Reconciliation:
    action: RecoveryAction
    snapshot: WorkspaceSnapshot
    reason: str


def reconcile_workspace(
    workspace: GitWorkspace,
    *,
    expected_branch: str,
    base_commit: str,
    latest_checkpoint_commit: str | None,
    active_iteration: bool,
) -> Reconciliation:
    """Classify the workspace without modifying it."""
    workspace.assert_branch(expected_branch)
    snapshot = workspace.snapshot()

    if snapshot.commit_sha == base_commit:
        if snapshot.dirty and active_iteration:
            return Reconciliation(
                RecoveryAction.RESUME_UNCOMMITTED,
                snapshot,
                "Workspace has uncommitted changes on the expected iteration base.",
            )
        if snapshot.dirty:
            return Reconciliation(
                RecoveryAction.BLOCK_UNEXPECTED,
                snapshot,
                "Workspace is dirty but there is no active iteration.",
            )
        return Reconciliation(
            RecoveryAction.CLEAN_EXPECTED_BASE,
            snapshot,
            "Workspace is clean at the persisted base commit.",
        )

    if (
        latest_checkpoint_commit is not None
        and snapshot.commit_sha == latest_checkpoint_commit
        and not snapshot.dirty
    ):
        return Reconciliation(
            RecoveryAction.CHECKPOINT_ALREADY_APPLIED,
            snapshot,
            "Workspace already points at the latest accepted checkpoint.",
        )

    return Reconciliation(
        RecoveryAction.BLOCK_UNEXPECTED,
        snapshot,
        "Workspace HEAD does not match the active base or latest checkpoint.",
    )
