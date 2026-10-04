"""Workspace lease helper used by the orchestration runtime."""

from __future__ import annotations

from runtime.state import StateStore


class WorkspaceLease:
    """Context-managed lease preventing concurrent sessions on one workspace."""

    def __init__(
        self,
        store: StateStore,
        workspace_path: str,
        owner_id: str,
        ttl_seconds: float = 60.0,
    ) -> None:
        self.store = store
        self.workspace_path = workspace_path
        self.owner_id = owner_id
        self.ttl_seconds = ttl_seconds
        self._acquired = False

    def acquire(self) -> None:
        self.store.acquire_lease(
            self.workspace_path,
            self.owner_id,
            self.ttl_seconds,
        )
        self._acquired = True

    def heartbeat(self) -> None:
        if not self._acquired:
            raise RuntimeError("Lease has not been acquired")
        self.store.heartbeat_lease(
            self.workspace_path,
            self.owner_id,
            self.ttl_seconds,
        )

    def release(self) -> None:
        if self._acquired:
            self.store.release_lease(self.workspace_path, self.owner_id)
            self._acquired = False

    def __enter__(self) -> "WorkspaceLease":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
