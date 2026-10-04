"""Deterministic command validation for Executor workspaces."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from runtime.artifacts import ArtifactStore


class ValidationError(ValueError):
    """Raised when a validation command is invalid."""


@dataclass(frozen=True, slots=True)
class ValidationCommand:
    """One command executed by the validation runner."""

    name: str
    argv: tuple[str, ...]
    required: bool = True
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValidationError("Validation command name must not be empty")
        if self.name in {".", ".."} or "/" in self.name or "\\" in self.name:
            raise ValidationError("Validation command name must be a safe path component")
        if not self.argv:
            raise ValidationError(f"Validation command {self.name!r} has no argv")
        if any(not isinstance(item, str) or not item for item in self.argv):
            raise ValidationError(f"Validation command {self.name!r} contains an invalid argv item")
        if any("\x00" in item for item in self.argv):
            raise ValidationError(f"Validation command {self.name!r} contains a NUL byte")
        if self.timeout_seconds <= 0:
            raise ValidationError(
                f"Validation command {self.name!r} timeout must be greater than zero"
            )


@dataclass(frozen=True, slots=True)
class ValidationCommandResult:
    """Result of one validation command."""

    name: str
    argv: tuple[str, ...]
    required: bool
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    stdout_artifact: str | None
    stderr_artifact: str | None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Aggregate validation result for one Executor iteration."""

    success: bool
    commands: tuple[ValidationCommandResult, ...]


class ValidationRunner:
    """Run validation commands sequentially and capture their output."""

    def __init__(
        self,
        workspace_path: str | Path,
        artifact_store: ArtifactStore,
    ) -> None:
        self.workspace_path = Path(workspace_path)
        self.artifact_store = artifact_store

        if not self.workspace_path.is_dir():
            raise ValidationError(f"Validation workspace does not exist: {self.workspace_path}")

    def run(
        self,
        commands: Sequence[ValidationCommand],
        *,
        session_id: str,
        task_id: str,
        iteration_id: str,
    ) -> ValidationResult:
        """Execute commands until a required command fails."""
        results: list[ValidationCommandResult] = []
        names: set[str] = set()

        for command in commands:
            if command.name in names:
                raise ValidationError(
                    f"Validation command name must be unique within a run: {command.name!r}"
                )
            names.add(command.name)
            result = self._run_command(
                command,
                session_id=session_id,
                task_id=task_id,
                iteration_id=iteration_id,
            )
            results.append(result)

            if command.required and not result.success:
                break

        return ValidationResult(
            success=all(
                result.success or not result.required
                for result in results
            ) and len(results) == len(commands),
            commands=tuple(results),
        )

    def _run_command(
        self,
        command: ValidationCommand,
        *,
        session_id: str,
        task_id: str,
        iteration_id: str,
    ) -> ValidationCommandResult:
        started = time.monotonic()

        try:
            completed = subprocess.run(
                list(command.argv),
                cwd=self.workspace_path,
                text=True,
                capture_output=True,
                check=False,
                shell=False,
                timeout=command.timeout_seconds,
            )
            exit_code = completed.returncode
            timed_out = False
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            exit_code = None
            timed_out = True
            stdout = self._coerce_timeout_output(exc.stdout)
            stderr = self._coerce_timeout_output(exc.stderr)
            timeout_message = (
                f"Validation command timed out after {command.timeout_seconds:.1f} seconds."
            )
            stderr = f"{stderr}{timeout_message}\n"

        duration = time.monotonic() - started

        stdout_path, _, _ = self.artifact_store.write_text(
            session_id,
            task_id,
            iteration_id,
            f"{command.name}.stdout.log",
            stdout,
        )
        stderr_path, _, _ = self.artifact_store.write_text(
            session_id,
            task_id,
            iteration_id,
            f"{command.name}.stderr.log",
            stderr,
        )

        return ValidationCommandResult(
            name=command.name,
            argv=command.argv,
            required=command.required,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=duration,
            stdout_artifact=stdout_path,
            stderr_artifact=stderr_path,
        )

    @staticmethod
    def _coerce_timeout_output(value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
