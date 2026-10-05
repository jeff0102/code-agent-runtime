"""Command-line entry point for code-agent-runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from uuid import uuid4

from runtime.artifacts import ArtifactStore
from runtime.git_integration import RemoteIntegrationConfig
from runtime.openhands_executor import OpenHandsExecutorConfig, OpenHandsExecutorFactory
from runtime.openhands_fallback import OpenHandsLLMFallbackConfig
from runtime.openhands_supervisor import OpenHandsSupervisorConfig, OpenHandsSupervisorFactory
from runtime.orchestrator import Orchestrator, OrchestratorConfig
from runtime.scope import fingerprint_scope
from runtime.state import StateStore
from runtime.validation import ValidationCommand


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-agent-runtime",
        description="Run a durable autonomous Supervisor/Executor coding session.",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--repository")
    parser.add_argument("--branch")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument(
        "--validation",
        action="append",
        default=[],
        metavar="NAME::COMMAND",
        help="Repeatable validation command, e.g. tests::python -m pytest",
    )
    parser.add_argument("--executor-model", default=os.getenv("OPENHANDS_EXECUTOR_MODEL"))
    parser.add_argument("--supervisor-model", default=os.getenv("OPENHANDS_SUPERVISOR_MODEL"))
    parser.add_argument("--executor-api-key", default=os.getenv("OPENHANDS_EXECUTOR_API_KEY"))
    parser.add_argument("--supervisor-api-key", default=os.getenv("OPENHANDS_SUPERVISOR_API_KEY"))
    parser.add_argument("--executor-base-url", default=os.getenv("OPENHANDS_EXECUTOR_BASE_URL"))
    parser.add_argument("--supervisor-base-url", default=os.getenv("OPENHANDS_SUPERVISOR_BASE_URL"))
    parser.add_argument("--reviewer-workspace", type=Path)
    parser.add_argument(
        "--enable-push",
        action="store_true",
        help="After acceptance, fast-forward the configured remote target branch.",
    )
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--target-branch", default="main")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()

    if not workspace.is_dir():
        raise SystemExit(f"Workspace does not exist: {workspace}")
    if not (workspace / ".git").exists():
        raise SystemExit(f"Workspace is not a Git repository: {workspace}")
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be greater than zero")
    if args.max_tasks < 1:
        raise SystemExit("--max-tasks must be greater than zero")
    if not args.executor_model or not args.supervisor_model:
        raise SystemExit(
            "Both --executor-model and --supervisor-model are required "
            "(or OPENHANDS_*_MODEL environment variables)."
        )

    state_root = (
        args.state_path.expanduser().resolve()
        if args.state_path
        else Path.home() / ".code-agent-runtime" / _workspace_key(workspace)
    )
    state_root.mkdir(parents=True, exist_ok=True)
    database_path = state_root / "runtime.db"
    artifact_root = state_root / "artifacts"

    state = StateStore(database_path)

    if args.session_id:
        session = state.get_session(args.session_id)
        if Path(session.workspace_path).resolve() != workspace:
            raise SystemExit(
                f"Session workspace mismatch: expected {workspace}, "
                f"found {session.workspace_path}"
            )
    else:
        scope = fingerprint_scope(workspace)
        session_id = str(uuid4())
        branch = args.branch or f"agent/{session_id}"
        repository = args.repository or workspace.name
        session = state.create_session(
            repository=repository,
            workspace_path=str(workspace),
            branch=branch,
            scope_hash=scope.sha256,
            max_iterations=args.max_iterations,
            session_id=session_id,
        )

    validations = tuple(_parse_validation(item) for item in args.validation)
    executor_fallbacks = _read_fallbacks_from_env("OPENHANDS_EXECUTOR")
    supervisor_fallbacks = _read_fallbacks_from_env("OPENHANDS_SUPERVISOR")

    orchestrator = Orchestrator(
        state=state,
        artifact_store=ArtifactStore(artifact_root),
        executor_factory=OpenHandsExecutorFactory(
            OpenHandsExecutorConfig(
                model=args.executor_model,
                api_key=args.executor_api_key,
                base_url=args.executor_base_url,
                fallbacks=executor_fallbacks,
                persistence_dir=state_root / "openhands" / "executor",
            )
        ),
        supervisor_factory=OpenHandsSupervisorFactory(
            OpenHandsSupervisorConfig(
                model=args.supervisor_model,
                api_key=args.supervisor_api_key,
                base_url=args.supervisor_base_url,
                fallbacks=supervisor_fallbacks,
                persistence_dir=state_root / "openhands" / "supervisor",
            )
        ),
        config=OrchestratorConfig(
            validation_commands=validations,
            reviewer_workspace=args.reviewer_workspace,
            max_tasks_per_session=args.max_tasks,
            remote_integration=RemoteIntegrationConfig(
                enabled=args.enable_push,
                remote=args.git_remote,
                target_branch=args.target_branch,
            ),
        ),
    )

    print(f"session_id={session.session_id}", file=sys.stderr)
    result = orchestrator.run_session(session.session_id)
    print(json.dumps(_result_dict(result), sort_keys=True, indent=2))
    return 0 if result.session_status.value == "DONE" else 1


def _read_fallbacks_from_env(
    prefix: str,
) -> tuple[OpenHandsLLMFallbackConfig, ...]:
    """Read contiguous or sparse numbered fallback configurations from environment."""
    marker = f"{prefix}_FALLBACK_"
    suffix = "_MODEL"
    indices: set[int] = set()

    for key in os.environ:
        if key.startswith(marker) and key.endswith(suffix):
            index_text = key[len(marker) : -len(suffix)]
            if index_text.isdigit():
                indices.add(int(index_text))

    fallbacks: list[OpenHandsLLMFallbackConfig] = []
    for index in sorted(indices):
        model = os.getenv(f"{marker}{index}{suffix}")
        if model is None:
            continue

        try:
            fallbacks.append(
                OpenHandsLLMFallbackConfig(
                    model=model,
                    api_key=os.getenv(f"{marker}{index}_API_KEY"),
                    base_url=os.getenv(f"{marker}{index}_BASE_URL"),
                )
            )
        except ValueError as exc:
            raise SystemExit(
                f"Invalid {marker}{index}{suffix} configuration: {exc}"
            ) from exc

    return tuple(fallbacks)


def _parse_validation(value: str) -> ValidationCommand:
    if "::" not in value:
        raise SystemExit(
            f"Invalid --validation value {value!r}; expected NAME::COMMAND"
        )
    name, command = value.split("::", 1)
    try:
        argv = tuple(shlex.split(command))
    except ValueError as exc:
        raise SystemExit(f"Invalid validation command {name!r}: {exc}") from exc
    try:
        return ValidationCommand(name=name, argv=argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _workspace_key(workspace: Path) -> str:
    digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:16]
    return f"{workspace.name}-{digest}"


def _result_dict(result) -> dict[str, object]:
    return {
        "session_id": result.session_id,
        "session_status": result.session_status.value,
        "tasks_completed": result.tasks_completed,
        "failure_reason": result.failure_reason,
    }


if __name__ == "__main__":
    raise SystemExit(main())
