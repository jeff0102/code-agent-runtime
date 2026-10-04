"""Deterministic fingerprinting of the target project's authoritative scope."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class ScopeError(RuntimeError):
    """Raised when the target project's scope cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class ScopeSnapshot:
    """Immutable snapshot metadata for SCOPE.md."""

    path: str
    sha256: str


def fingerprint_scope(workspace_path: str | Path) -> ScopeSnapshot:
    """Compute a SHA-256 fingerprint for the repository's SCOPE.md."""
    path = Path(workspace_path) / "SCOPE.md"

    if not path.is_file():
        raise ScopeError(f"Required scope file not found: {path}")

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ScopeError(f"Unable to read scope file: {path}") from exc

    digest = hashlib.sha256(content).hexdigest()
    return ScopeSnapshot(path=str(path), sha256=digest)
