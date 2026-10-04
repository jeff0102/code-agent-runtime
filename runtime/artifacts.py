"""File-backed artifact storage with content hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path


class ArtifactStore:
    """Stores logs, diffs, reports, and other runtime artifacts on disk."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_bytes(
        self,
        session_id: str,
        task_id: str,
        iteration_id: str,
        artifact_type: str,
        content: bytes,
    ) -> tuple[str, str, int]:
        """Write an artifact and return (path, sha256, size_bytes)."""
        safe_type = self._safe_component(artifact_type)
        directory = self.root / session_id / task_id / iteration_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / safe_type
        path.write_bytes(content)

        digest = hashlib.sha256(content).hexdigest()
        return str(path), digest, len(content)

    def write_text(
        self,
        session_id: str,
        task_id: str,
        iteration_id: str,
        artifact_type: str,
        content: str,
    ) -> tuple[str, str, int]:
        """Write a UTF-8 text artifact."""
        return self.write_bytes(
            session_id,
            task_id,
            iteration_id,
            artifact_type,
            content.encode("utf-8"),
        )

    @staticmethod
    def _safe_component(value: str) -> str:
        """Prevent path traversal through artifact type names."""
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Invalid artifact type: {value!r}")
        return value
