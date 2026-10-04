from runtime.models import Decision, SessionStatus, TaskStatus
from runtime.state import StateError, StateStore


def test_session_task_iteration_checkpoint_lifecycle(tmp_path):
    store = StateStore(tmp_path / "runtime.db")

    session = store.create_session(
        repository="owner/repo",
        workspace_path=str(tmp_path / "repo"),
        branch="agent/session-1",
        scope_hash="scope-123",
        max_iterations=3,
        session_id="session-1",
    )
    assert session.status == SessionStatus.RUNNING

    task = store.create_task(
        "session-1",
        sequence=1,
        title="Create configuration module",
        objective="Add runtime configuration.",
        instructions="Implement the configuration module.",
        acceptance_criteria="Configuration can be loaded and tested.",
        task_id="task-1",
    )
    assert task.status == TaskStatus.PLANNED

    iteration = store.start_iteration("task-1", base_commit="abc123")
    assert iteration.attempt_number == 1
    assert store.get_task("task-1").status == TaskStatus.EXECUTING

    store.set_task_status("task-1", TaskStatus.REVIEWING)
    store.complete_iteration(iteration.iteration_id, Decision.ACCEPT)

    accepted_task = store.get_task("task-1")
    assert accepted_task.status == TaskStatus.ACCEPTED

    checkpoint = store.create_checkpoint(
        "task-1",
        iteration.iteration_id,
        commit_sha="def456",
        checkpoint_id="checkpoint-1",
    )
    assert checkpoint.commit_sha == "def456"
    assert store.latest_checkpoint("session-1") == checkpoint

    events = store.list_events("session-1")
    assert [event["event_type"] for event in events] == [
        "SESSION_CREATED",
        "TASK_CREATED",
        "ITERATION_STARTED",
        "TASK_STATUS_CHANGED",
        "ITERATION_COMPLETED",
        "CHECKPOINT_CREATED",
    ]


def test_invalid_task_transition_is_rejected(tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    store.create_session(
        repository="owner/repo",
        workspace_path=str(tmp_path / "repo"),
        branch="agent/session-1",
        scope_hash="scope",
        max_iterations=3,
        session_id="session-1",
    )
    store.create_task(
        "session-1",
        1,
        "Task",
        "Objective",
        "Instructions",
        "Acceptance",
        task_id="task-1",
    )

    store.set_task_status("task-1", TaskStatus.EXECUTING)

    try:
        store.set_task_status("task-1", TaskStatus.ACCEPTED)
    except StateError:
        pass
    else:
        raise AssertionError("Expected invalid task transition to raise StateError")


def test_lease_prevents_two_active_owners(tmp_path):
    store = StateStore(tmp_path / "runtime.db")
    workspace = str(tmp_path / "repo")

    store.acquire_lease(workspace, "owner-a", ttl_seconds=60)

    try:
        store.acquire_lease(workspace, "owner-b", ttl_seconds=60)
    except StateError:
        pass
    else:
        raise AssertionError("Expected the second owner to be rejected")

    store.heartbeat_lease(workspace, "owner-a", ttl_seconds=60)
    store.release_lease(workspace, "owner-a")
    store.acquire_lease(workspace, "owner-b", ttl_seconds=60)
