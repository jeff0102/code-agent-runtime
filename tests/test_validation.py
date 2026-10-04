import sys
from pathlib import Path

import pytest

from runtime.artifacts import ArtifactStore
from runtime.validation import ValidationCommand, ValidationError, ValidationRunner


def runner(tmp_path: Path) -> ValidationRunner:
    return ValidationRunner(
        tmp_path,
        ArtifactStore(tmp_path / "artifacts"),
    )


def test_successful_required_command_captures_output(tmp_path):
    result = runner(tmp_path).run(
        [
            ValidationCommand(
                name="unit-tests",
                argv=(sys.executable, "-c", "print('all good')"),
            )
        ],
        session_id="session",
        task_id="task",
        iteration_id="iteration",
    )

    command = result.commands[0]

    assert result.success
    assert command.success
    assert command.exit_code == 0
    assert Path(command.stdout_artifact).read_text() == "all good\n"
    assert Path(command.stderr_artifact).read_text() == ""


def test_optional_failure_does_not_fail_overall_validation(tmp_path):
    result = runner(tmp_path).run(
        [
            ValidationCommand(
                name="optional-check",
                argv=(sys.executable, "-c", "raise SystemExit(4)"),
                required=False,
            ),
            ValidationCommand(
                name="required-check",
                argv=(sys.executable, "-c", "print('required passed')"),
            ),
        ],
        session_id="session",
        task_id="task",
        iteration_id="iteration",
    )

    assert result.success
    assert len(result.commands) == 2
    assert result.commands[0].exit_code == 4
    assert result.commands[1].success


def test_required_failure_stops_following_commands(tmp_path):
    marker = tmp_path / "marker.txt"

    result = runner(tmp_path).run(
        [
            ValidationCommand(
                name="failing-check",
                argv=(sys.executable, "-c", "raise SystemExit(2)"),
            ),
            ValidationCommand(
                name="should-not-run",
                argv=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                ),
            ),
        ],
        session_id="session",
        task_id="task",
        iteration_id="iteration",
    )

    assert not result.success
    assert len(result.commands) == 1
    assert result.commands[0].exit_code == 2
    assert not marker.exists()


def test_timeout_is_reported_as_failure_and_is_captured(tmp_path):
    result = runner(tmp_path).run(
        [
            ValidationCommand(
                name="slow-check",
                argv=(
                    sys.executable,
                    "-c",
                    "import time; print('before timeout', flush=True); time.sleep(0.2)",
                ),
                timeout_seconds=0.05,
            )
        ],
        session_id="session",
        task_id="task",
        iteration_id="iteration",
    )

    command = result.commands[0]

    assert not result.success
    assert command.timed_out
    assert command.exit_code is None
    assert "timed out" in Path(command.stderr_artifact).read_text()


def test_invalid_validation_command_name_is_rejected():
    with pytest.raises(ValidationError):
        ValidationCommand(
            name="../unsafe",
            argv=("python", "-c", "print('no')"),
        )


def test_empty_command_is_rejected():
    with pytest.raises(ValidationError):
        ValidationCommand(name="empty", argv=())
