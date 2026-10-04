from runtime.artifacts import ArtifactStore


def test_artifact_store_writes_hashed_artifact(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    path, digest, size = store.write_text(
        "session-1",
        "task-1",
        "iteration-1",
        "tests.log",
        "pytest passed\n",
    )

    artifact = (tmp_path / "artifacts" / "session-1" / "task-1" / "iteration-1" / "tests.log")
    assert path == str(artifact)
    assert artifact.read_text() == "pytest passed\n"
    assert len(digest) == 64
    assert size == len("pytest passed\n".encode("utf-8"))


def test_artifact_type_cannot_escape_directory(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    try:
        store.write_text(
            "session-1",
            "task-1",
            "iteration-1",
            "../escape",
            "bad",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected path traversal to be rejected")
