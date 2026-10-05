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
- Remote publication is opt-in with `--enable-push`; the runtime integrates the configured remote target into the isolated session branch and performs only ordinary fast-forward pushes after acceptance.
- REVISE creates another iteration for the same task.
- After each accepted task, the Supervisor plans the next task or declares DONE.
- Session state, events, artifacts, and OpenHands conversation identifiers are persisted outside the target repository.
- Startup recovery can resume an interrupted Executor iteration and repair accepted checkpoint boundaries.

### Supervisor planner protocol

The runtime owns the Supervisor wire format. `SCOPE.md` and `AGENTS.md` define product requirements and repository guidance; they do not redefine this protocol. The planner returns one JSON object. For an actionable task, the canonical format is:

Before requesting a task, the planner context includes a bounded snapshot of tracked and untracked paths, recent commit subjects, the current Git status and diff, and tasks completed in this session. The planner must compare that evidence with the milestones in `SCOPE.md`, skip work already present in the repository, and use a verification task when acceptance evidence is unclear.

Runtime checkpoint subjects include a bounded version of the accepted task title so later sessions can use Git history to understand previously completed work.

```json
{
  "schema_version": 1,
  "message_type": "supervisor_plan",
  "action": "NEXT_TASK",
  "title": "Add a health endpoint",
  "objective": "Expose a deterministic health check.",
  "instructions": "Implement GET /health and its focused test.",
  "acceptance_criteria": "GET /health returns HTTP 200 with a healthy JSON response.",
  "blocking_reason": null
}
```

`instructions` and `acceptance_criteria` are strings in the canonical protocol. The adapter also converts observed `SupervisorTaskPlan` forms (`status: TASK` with task fields at the top level, and `status: READY` with a nested `task` object) into this format. In those forms, `scope` may provide task instructions when `instructions` is absent; list values become readable bullet lists. The adapter preserves `validation` and `constraints` as additional instructions. Unknown fields and unsupported statuses remain errors. SupervisorDecision responses also canonicalize the observed message_type value SupervisorDecision and accept the type/status envelope and can omit unambiguous protocol metadata: the adapter fills schema version and message type, infers task_complete from ACCEPT/REVISE/BLOCK, and defaults blocking_reason to null for ACCEPT and REVISE. BLOCK still requires a reason.

For `DONE`, set `title`, `objective`, `instructions`, `acceptance_criteria`, and `blocking_reason` to `null`. For `BLOCK`, set the task fields to `null` and provide a non-empty `blocking_reason`.

### Local usage

Install the package and OpenHands adapters:

```bash
pip install -e ".[agents]"
```

Run a complete autonomous session with direct provider access through the OpenHands SDK/LiteLLM integration:

```bash
code-agent-runtime \
  --workspace /path/to/repository \
  --executor-model gemini/gemini-3.8-flash \
  --supervisor-model gemini/gemini-3.8-flash \
  --validation tests::python -m pytest
```

To enable automatic publication of accepted checkpoints, add `--enable-push`. The defaults are remote `origin` and target branch `main`; override them with `--git-remote` and `--target-branch`. The runtime fetches and merges the target branch into the session branch before implementation. It asks the Executor to resolve merge conflicts when needed, then requires validation and Supervisor acceptance before pushing. Pushes are normal fast-forward pushes; the runtime never force-pushes or asks the Executor to switch to the target branch.

Fallbacks are configured independently for Executor and Supervisor through numbered environment variables. Any number of fallback slots may be supplied:

```env
OPENHANDS_EXECUTOR_FALLBACK_1_MODEL=xai/grok-4.7
OPENHANDS_EXECUTOR_FALLBACK_1_API_KEY=
OPENHANDS_EXECUTOR_FALLBACK_1_BASE_URL=https://api.x.ai/v1

OPENHANDS_EXECUTOR_FALLBACK_2_MODEL=gemini/gemini-3.7-flash
OPENHANDS_EXECUTOR_FALLBACK_2_API_KEY=
OPENHANDS_EXECUTOR_FALLBACK_2_BASE_URL=
```

The equivalent Supervisor variables use the `OPENHANDS_SUPERVISOR_FALLBACK_<N>_*` prefix.

Fallbacks are tried in order only after the primary model fails with a transient error. Each new model call starts from the primary model again. The runtime uses a deliberately short retry policy: **3 total attempts per provider with 60 seconds between attempts**. After the primary provider exhausts its three attempts, the fallback strategy moves to the next configured provider. The same retry policy applies to each fallback provider. Fallback credentials are stored only in a temporary OpenHands profile directory and are not written into the runtime state database or artifacts.

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
