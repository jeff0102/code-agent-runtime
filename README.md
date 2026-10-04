# code-agent-runtime

Persistent local orchestration infrastructure for autonomous Supervisor/Executor coding agents.

## Current Architecture

The runtime is intentionally split into independent layers:

- **StateStore** — SQLite persistence for sessions, tasks, iterations, checkpoints, events, artifacts, and workspace leases.
- **ArtifactStore** — file-backed storage for diffs, logs, validation output, and agent reports.
- **GitWorkspace** — deterministic Git operations owned by the runtime.
- **Recovery** — read-only reconciliation of a workspace after interruption.
- **WorkspaceLease** — prevents concurrent sessions from modifying the same workspace.

The Supervisor/Executor agent integration is intentionally built on top of this foundation.

## State Layout

Runtime state must live outside the target repository:

```text
/state/
├── runtime.db
└── artifacts/
    └── <session>/<task>/<iteration>/
        ├── git.diff
        ├── git.status
        ├── validation.log
        ├── executor-report.json
        └── supervisor-review.json
```

SQLite stores metadata and hashes. Large artifacts remain on disk.

## Recovery Model

The runtime never deletes unreviewed workspace changes.

After a restart it compares:

- expected branch;
- persisted base commit;
- latest accepted checkpoint;
- current Git status.

A clean expected base can start normally. Dirty changes during an active iteration can be resumed. Unexpected workspace changes are blocked for human inspection.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
pytest
```

The current branch implements the persistent runtime foundation. OpenHands integration and the autonomous Supervisor/Executor loop are separate layers built on top of it.
