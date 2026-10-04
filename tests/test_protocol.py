import json

import pytest

from runtime.protocol import (
    ExecutorReport,
    ExecutorStatus,
    ProtocolError,
    SupervisorDecision,
    SupervisorDecisionType,
    SupervisorPlan,
    SupervisorPlanType,
)


def test_supervisor_accept_round_trips_through_json():
    message = SupervisorDecision(
        decision=SupervisorDecisionType.ACCEPT,
        task_complete=True,
        instructions=[],
        blocking_reason=None,
    )

    restored = SupervisorDecision.from_json(message.to_json())

    assert restored == message
    assert restored.to_dict()["schema_version"] == 1
    assert restored.to_dict()["message_type"] == "supervisor_decision"


def test_supervisor_revision_requires_instructions():
    with pytest.raises(ProtocolError, match="REVISE requires"):
        SupervisorDecision(
            decision=SupervisorDecisionType.REVISE,
            task_complete=False,
            instructions=[],
            blocking_reason=None,
        )


def test_supervisor_block_requires_reason():
    with pytest.raises(ProtocolError, match="BLOCK requires"):
        SupervisorDecision(
            decision=SupervisorDecisionType.BLOCK,
            task_complete=False,
            instructions=[],
            blocking_reason=None,
        )


def test_supervisor_rejects_unknown_fields():
    payload = SupervisorDecision(
        decision=SupervisorDecisionType.ACCEPT,
        task_complete=True,
        instructions=[],
        blocking_reason=None,
    ).to_dict()
    payload["extra"] = "not allowed"

    with pytest.raises(ProtocolError, match="unknown fields"):
        SupervisorDecision.from_dict(payload)


def test_supervisor_rejects_wrong_schema_version():
    payload = SupervisorDecision(
        decision=SupervisorDecisionType.ACCEPT,
        task_complete=True,
        instructions=[],
        blocking_reason=None,
    ).to_dict()
    payload["schema_version"] = 999

    with pytest.raises(ProtocolError, match="Unsupported"):
        SupervisorDecision.from_dict(payload)


def test_executor_completed_round_trips_through_dict():
    report = ExecutorReport(
        status=ExecutorStatus.COMPLETED,
        summary="Implemented the requested configuration module.",
        changed_files=["runtime/config.py", "tests/test_config.py"],
        tests_executed=["pytest tests/test_config.py"],
        validation_summary="2 tests passed.",
        blockers=[],
    )

    restored = ExecutorReport.from_dict(report.to_dict())

    assert restored == report


def test_executor_blocked_requires_blockers():
    with pytest.raises(ProtocolError, match="BLOCKED requires"):
        ExecutorReport(
            status=ExecutorStatus.BLOCKED,
            summary="Could not continue.",
            changed_files=[],
            tests_executed=[],
            validation_summary="Validation did not start.",
            blockers=[],
        )


def test_executor_completed_rejects_blockers():
    with pytest.raises(ProtocolError, match="COMPLETED must not"):
        ExecutorReport(
            status=ExecutorStatus.COMPLETED,
            summary="Finished.",
            changed_files=[],
            tests_executed=[],
            validation_summary="All checks passed.",
            blockers=["unexpected blocker"],
        )


def test_executor_rejects_non_string_list_items():
    payload = {
        "schema_version": 1,
        "message_type": "executor_report",
        "status": "COMPLETED",
        "summary": "Finished.",
        "changed_files": ["runtime/main.py", 123],
        "tests_executed": [],
        "validation_summary": "All checks passed.",
        "blockers": [],
    }

    with pytest.raises(ProtocolError, match=r"changed_files\[1\]"):
        ExecutorReport.from_dict(payload)


def test_invalid_json_is_rejected():
    with pytest.raises(ProtocolError, match="Invalid SupervisorDecision JSON"):
        SupervisorDecision.from_json("{not-json")


def test_serialization_is_valid_json():
    message = ExecutorReport(
        status=ExecutorStatus.COMPLETED,
        summary="Finished.",
        changed_files=[],
        tests_executed=[],
        validation_summary="All checks passed.",
        blockers=[],
    )

    decoded = json.loads(message.to_json())

    assert decoded["message_type"] == "executor_report"
    assert decoded["status"] == "COMPLETED"


def test_supervisor_next_task_plan_round_trips():
    plan = SupervisorPlan(
        action=SupervisorPlanType.NEXT_TASK,
        title="Add configuration loading",
        objective="Load runtime configuration from environment variables.",
        instructions="Implement the loader and its validation.",
        acceptance_criteria="Configuration is validated and covered by tests.",
        blocking_reason=None,
    )

    restored = SupervisorPlan.from_json(plan.to_json())

    assert restored == plan
    assert restored.to_dict()["message_type"] == "supervisor_plan"


def test_supervisor_done_plan_rejects_task_fields():
    with pytest.raises(ProtocolError, match="DONE must not"):
        SupervisorPlan(
            action=SupervisorPlanType.DONE,
            title="Should not exist",
            objective=None,
            instructions=None,
            acceptance_criteria=None,
            blocking_reason=None,
        )


def test_supervisor_block_plan_requires_reason():
    with pytest.raises(ProtocolError, match="BLOCK requires"):
        SupervisorPlan(
            action=SupervisorPlanType.BLOCK,
            title=None,
            objective=None,
            instructions=None,
            acceptance_criteria=None,
            blocking_reason=None,
        )
