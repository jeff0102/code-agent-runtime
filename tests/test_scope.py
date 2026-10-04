import hashlib

import pytest

from runtime.scope import ScopeError, fingerprint_scope


def test_scope_fingerprint_is_deterministic(tmp_path):
    (tmp_path / "SCOPE.md").write_text("# Project\n\nBuild the thing.\n")

    snapshot = fingerprint_scope(tmp_path)

    expected = hashlib.sha256(
        b"# Project\n\nBuild the thing.\n"
    ).hexdigest()

    assert snapshot.sha256 == expected
    assert snapshot.path.endswith("SCOPE.md")


def test_scope_fingerprint_changes_when_file_changes(tmp_path):
    (tmp_path / "SCOPE.md").write_text("version one\n")
    first = fingerprint_scope(tmp_path)

    (tmp_path / "SCOPE.md").write_text("version two\n")
    second = fingerprint_scope(tmp_path)

    assert first.sha256 != second.sha256


def test_missing_scope_is_rejected(tmp_path):
    with pytest.raises(ScopeError, match="scope file not found"):
        fingerprint_scope(tmp_path)
