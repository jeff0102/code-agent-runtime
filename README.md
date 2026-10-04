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


## Runtime MVP

The runtime can execute a complete session without product-specific code:

- The Supervisor plans the next atomic task.
- The Executor implements it in the target workspace.
- Deterministic validation runs after execution.
- The Supervisor reviews the evidence and returns ACCEPT, REVISE, or BLOCK.
- Accepted work is checkpointed by the runtime.
- REVISE creates another iteration for the same task.
- After each accepted task, the Supervisor plans the next task or declares DONE.
- Session state, events, artifacts, and OpenHands conversation identifiers are persisted outside the target repository.
- Startup recovery can resume an interrupted Executor iteration and repair accepted checkpoint boundaries.

### Local usage

Install the package and OpenHands adapters:

```bash
pip install -e ".[agents]"
```

Run a complete autonomous session:

```bash
code-agent-runtime \\
  --workspace /path/to/repository \\
  --executor-model openai/gemini-primary \\
  --supervisor-model openai/gemini-primary \\
  --executor-base-url http://localhost:4000/v1 \\
  --supervisor-base-url http://localhost:4000/v1 \\
  --validation tests::python -m pytest
```

The first run creates a persistent session and lets the Supervisor plan the first task. Re-running with `--session-id <id>` resumes the same session.

Runtime state defaults to a directory under `~/.code-agent-runtime/`, keeping SQLite, artifacts, and OpenHands persistence outside the target repository.

### Docker

Build:

```bash
docker build -t code-agent-runtime .
```

The target repository should be mounted at `/workspace`. Runtime state can be mounted at `/runtime` and passed with `--state-path /runtime`.

The runtime is intentionally project-agnostic. A target project only needs its own `SCOPE.md`, optional `AGENTS.md`, and normal Git/test tooling.

### Ready criteria

The runtime is considered ready for external projects when:

- `pytest` is green;
- the Supervisor and Executor adapters are optional and independently testable;
- a complete multi-task session can be driven through the public orchestrator;
- a session can be restarted without losing persisted work or bypassing Git safety checks;
- workspace ownership is enforced by a lease;
- accepted work always has a durable checkpoint;
- total task creation is bounded per session;
- unexpected workspace or scope changes block execution rather than being overwritten.
