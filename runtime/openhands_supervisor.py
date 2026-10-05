"""Optional OpenHands adapter for read-only Supervisor reviews."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from runtime.openhands_fallback import (
    FallbackProfileManager,
    OpenHandsLLMFallbackConfig,
    build_llm_kwargs,
    SUPERVISOR_TIMEOUT_SECONDS,
)
from runtime.protocol import (
    SCHEMA_VERSION,
    ProtocolError,
    SupervisorDecision,
    SupervisorPlan,
)


class OpenHandsSupervisorError(RuntimeError):
    """Raised when a Supervisor review cannot be completed."""

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class SupervisorConversationLike(Protocol):
    id: UUID

    def ask_agent(self, question: str) -> str: ...
    def interrupt(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class OpenHandsSupervisorConfig:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    fallbacks: tuple[OpenHandsLLMFallbackConfig, ...] = ()
    timeout: int = SUPERVISOR_TIMEOUT_SECONDS
    persistence_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OpenHands Supervisor model must not be empty")
        if self.timeout < 1:
            raise ValueError("timeout must be greater than zero")


@dataclass(frozen=True, slots=True)
class OpenHandsSupervisorResult:
    conversation_id: str
    decision: SupervisorDecision
    raw_response: str


@dataclass(frozen=True, slots=True)
class OpenHandsSupervisorPlanResult:
    conversation_id: str
    plan: SupervisorPlan
    raw_response: str


class OpenHandsSupervisorAdapter:
    def __init__(self, conversation: SupervisorConversationLike) -> None:
        self._conversation = conversation

    @property
    def conversation_id(self) -> str:
        return str(self._conversation.id)

    def review(self, prompt: str) -> OpenHandsSupervisorResult:
        if not prompt.strip():
            raise ValueError("Supervisor review prompt must not be empty")

        raw_response: str | None = None
        try:
            raw_response = self._conversation.ask_agent(prompt)
            decision = parse_supervisor_decision(raw_response)
        except (ProtocolError, ValueError) as exc:
            raise OpenHandsSupervisorError(
                f"Supervisor returned an invalid decision: {exc}",
                raw_response=raw_response,
            ) from exc
        except Exception as exc:
            raise OpenHandsSupervisorError(
                f"OpenHands Supervisor review failed: {exc}",
                raw_response=raw_response,
            ) from exc

        return OpenHandsSupervisorResult(
            conversation_id=self.conversation_id,
            decision=decision,
            raw_response=raw_response,
        )

    def plan(self, prompt: str) -> OpenHandsSupervisorPlanResult:
        if not prompt.strip():
            raise ValueError("Supervisor plan prompt must not be empty")

        raw_response: str | None = None
        try:
            raw_response = self._conversation.ask_agent(prompt)
            plan = parse_supervisor_plan(raw_response)
        except (ProtocolError, ValueError) as exc:
            raise OpenHandsSupervisorError(
                f"Supervisor returned an invalid plan: {exc}",
                raw_response=raw_response,
            ) from exc
        except Exception as exc:
            raise OpenHandsSupervisorError(
                f"OpenHands Supervisor planning failed: {exc}",
                raw_response=raw_response,
            ) from exc

        return OpenHandsSupervisorPlanResult(
            conversation_id=self.conversation_id,
            plan=plan,
            raw_response=raw_response,
        )

    def interrupt(self) -> None:
        self._conversation.interrupt()

    def close(self) -> None:
        self._conversation.close()


class OpenHandsSupervisorFactory:
    def __init__(self, config: OpenHandsSupervisorConfig) -> None:
        self.config = config
        self._fallback_profiles: FallbackProfileManager | None = None

    def create(
        self,
        *,
        reviewer_workspace: str | Path,
        conversation_id: str | UUID | None = None,
    ) -> OpenHandsSupervisorAdapter:
        sdk = self._load_sdk()
        workspace = Path(reviewer_workspace)
        if not workspace.is_dir():
            raise OpenHandsSupervisorError(
                f"Supervisor workspace does not exist: {workspace}"
            )

        if self._fallback_profiles is None and self.config.fallbacks:
            self._fallback_profiles = FallbackProfileManager(
                sdk=sdk,
                fallbacks=self.config.fallbacks,
                usage_prefix="supervisor",
                timeout=self.config.timeout,
            )

        llm_kwargs = build_llm_kwargs(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        if self._fallback_profiles is not None:
            strategy = self._fallback_profiles.strategy()
            if strategy is not None:
                llm_kwargs["fallback_strategy"] = strategy

        llm = sdk["LLM"](**llm_kwargs)
        agent = sdk["Agent"](
            llm=llm,
            tools=[],
            persona=(
                "You are a read-only Supervisor for an autonomous software "
                "development runtime. Review only supplied evidence. Do not "
                "modify files. Follow the prompt's requested protocol exactly: "
                "return either SupervisorDecision for implementation review "
                "or SupervisorPlan for task planning."
            ),
        )

        kwargs: dict[str, Any] = {
            "agent": agent,
            "workspace": workspace,
            "delete_on_close": False,
        }
        if self.config.persistence_dir is not None:
            kwargs["persistence_dir"] = str(self.config.persistence_dir)
        if conversation_id is not None:
            kwargs["conversation_id"] = _coerce_uuid(conversation_id)

        try:
            conversation = sdk["Conversation"](**kwargs)
        except Exception as exc:
            raise OpenHandsSupervisorError(
                f"Failed to create OpenHands Supervisor conversation: {exc}"
            ) from exc

        return OpenHandsSupervisorAdapter(conversation)

    @staticmethod
    def _load_sdk() -> dict[str, Any]:
        try:
            from openhands.sdk import (
                Agent,
                Conversation,
                FallbackStrategy,
                LLM,
                LLMProfileStore,
            )
        except ImportError as exc:
            raise OpenHandsSupervisorError(
                "OpenHands SDK dependencies are not installed. "
                'Install with: pip install -e ".[agents]"'
            ) from exc

        return {
            "Agent": Agent,
            "Conversation": Conversation,
            "FallbackStrategy": FallbackStrategy,
            "LLM": LLM,
            "LLMProfileStore": LLMProfileStore,
        }


def _coerce_uuid(value: str | UUID) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(value)
    except ValueError as exc:
        raise OpenHandsSupervisorError(
            f"Invalid OpenHands Supervisor conversation ID: {value!r}"
        ) from exc


def parse_supervisor_decision(response: str) -> SupervisorDecision:
    candidate = response.strip()
    fence = chr(96) * 3
    if candidate.startswith(fence) and candidate.endswith(fence):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            f"Supervisor response was not valid JSON: {exc}"
        ) from exc

    if isinstance(payload, dict):
        payload = _normalize_supervisor_decision_payload(payload)
    return SupervisorDecision.from_dict(payload)


def _normalize_supervisor_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize known SupervisorDecision envelopes into the runtime protocol."""
    normalized = dict(payload)

    if normalized.get("type") == "SupervisorDecision":
        nested = normalized.get("decision")
        if isinstance(nested, dict):
            normalized = dict(nested)
        else:
            status = normalized.get("status")
            if "decision" not in normalized and status in {
                "ACCEPT",
                "REVISE",
                "BLOCK",
            }:
                normalized["decision"] = status
            normalized.pop("type", None)
            normalized.pop("status", None)

    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("message_type", "supervisor_decision")

    decision = normalized.get("decision")
    if decision == "ACCEPT":
        normalized.setdefault("task_complete", True)
        normalized.setdefault("blocking_reason", None)
    elif decision == "REVISE":
        normalized.setdefault("task_complete", False)
        normalized.setdefault("blocking_reason", None)
        summary = normalized.pop("summary", None)
        rationale = normalized.pop("rationale", None)
        required_revisions = normalized.pop("required_revisions", None)

        instructions = normalized.get("instructions")
        if instructions is None:
            instructions = []
        if not isinstance(instructions, list):
            return normalized

        context: list[str] = []
        if summary is not None:
            if not isinstance(summary, str) or not summary.strip():
                raise ProtocolError("SupervisorDecision summary must be a non-empty string")
            context.append("Review summary: " + summary.strip())
        if rationale is not None:
            if (
                not isinstance(rationale, list)
                or any(not isinstance(item, str) or not item.strip() for item in rationale)
            ):
                raise ProtocolError("SupervisorDecision rationale must be a list of non-empty strings")
            context.append(
                "Rationale:\n" + "\n".join(f"- {item.strip()}" for item in rationale)
            )
        if required_revisions is not None:
            if (
                not isinstance(required_revisions, list)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in required_revisions
                )
            ):
                raise ProtocolError(
                    "SupervisorDecision required_revisions must be a list of non-empty strings"
                )
            if not instructions:
                instructions = []
            if not isinstance(instructions, list):
                return normalized
            instructions.extend(item.strip() for item in required_revisions)

        if context and instructions:
            instructions = context + instructions
        normalized["instructions"] = instructions
    elif decision == "BLOCK":
        normalized.setdefault("task_complete", False)

    normalized.setdefault("instructions", [])
    return normalized


def parse_supervisor_plan(response: str) -> SupervisorPlan:
    candidate = response.strip()
    fence = chr(96) * 3
    if candidate.startswith(fence) and candidate.endswith(fence):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            f"Supervisor plan response was not valid JSON: {exc}"
        ) from exc

    if isinstance(payload, dict):
        if payload.get("type") == "SupervisorTaskPlan":
            payload = _convert_supervisor_task_plan(payload)
        else:
            payload = _normalize_supervisor_plan_payload(payload)
    return SupervisorPlan.from_dict(payload)


def _convert_supervisor_task_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert the observed alternate task-plan shapes to the runtime contract."""
    allowed_top_keys = {
        "type",
        "status",
        "task_id",
        "title",
        "objective",
        "instructions",
        "scope",
        "acceptance_criteria",
        "validation",
        "constraints",
        "task",
    }
    unknown = set(payload) - allowed_top_keys
    if unknown:
        raise ProtocolError(
            f"SupervisorTaskPlan contains unknown fields: {sorted(unknown)}"
        )

    status = payload.get("status")
    task = payload.get("task")
    if task is not None:
        if status != "READY":
            raise ProtocolError(
                f"Nested SupervisorTaskPlan requires READY status; got {status!r}"
            )
        if not isinstance(task, dict):
            raise ProtocolError("SupervisorTaskPlan task must be a JSON object")
        task_keys = {
            "task_id",
            "title",
            "objective",
            "instructions",
            "scope",
            "acceptance_criteria",
            "validation",
            "constraints",
        }
        task_unknown = set(task) - task_keys
        if task_unknown:
            raise ProtocolError(
                f"SupervisorTaskPlan task contains unknown fields: {sorted(task_unknown)}"
            )
        task_data = task
    elif status == "TASK":
        task_data = payload
    else:
        raise ProtocolError(f"Unsupported SupervisorTaskPlan status: {status!r}")

    required = {"title", "objective", "acceptance_criteria"}
    missing = required - set(task_data)
    if missing:
        raise ProtocolError(
            f"SupervisorTaskPlan task is missing fields: {sorted(missing)}"
        )

    title = _require_plan_text(task_data["title"], "title")
    objective = _require_plan_text(task_data["objective"], "objective")
    instruction_values: list[str] = []
    if "instructions" in task_data:
        instruction_values.append(
            _require_plan_text(task_data["instructions"], "instructions")
        )
    if "scope" in task_data:
        instruction_values.append(_require_plan_text(task_data["scope"], "scope"))
    if not instruction_values:
        raise ProtocolError(
            "SupervisorTaskPlan task is missing fields: ['instructions' or 'scope']"
        )
    instructions = "\n\n".join(instruction_values)
    acceptance_criteria = _require_plan_text(
        task_data["acceptance_criteria"], "acceptance_criteria"
    )

    extra_instructions: list[str] = []
    task_id = task_data.get("task_id", payload.get("task_id"))
    if task_id is not None:
        task_id_text = _require_plan_text(task_id, "task_id")
        extra_instructions.append("Planner task ID: " + task_id_text)
    for field in ("validation", "constraints"):
        detail_value = task_data.get(field, payload.get(field))
        if detail_value is not None:
            detail = _require_plan_text(detail_value, field, allow_empty_list=True)
            if detail:
                extra_instructions.append(f"{field.title()}:\n{detail}")
    if extra_instructions:
        instructions = instructions + "\n\n" + "\n\n".join(extra_instructions)

    return {
        "schema_version": SCHEMA_VERSION,
        "message_type": "supervisor_plan",
        "action": "NEXT_TASK",
        "title": title,
        "objective": objective,
        "instructions": instructions,
        "acceptance_criteria": acceptance_criteria,
        "blocking_reason": None,
    }


def _require_plan_text(value: Any, field: str, *, allow_empty_list: bool = False) -> str:
    if isinstance(value, str):
        if value.strip():
            return value.strip()
        if allow_empty_list:
            return ""
    elif isinstance(value, list):
        if not value and allow_empty_list:
            return ""
        if value and all(isinstance(item, str) and item.strip() for item in value):
            return "\n".join(f"- {item.strip()}" for item in value)
    raise ProtocolError(
        f"{field} must be a non-empty string or list of non-empty strings"
    )


def _normalize_supervisor_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill unambiguous protocol metadata omitted by some model responses."""
    normalized = dict(payload)
    expected_keys = {
        "schema_version",
        "message_type",
        "action",
        "title",
        "objective",
        "instructions",
        "acceptance_criteria",
        "blocking_reason",
    }
    if set(normalized) - expected_keys:
        return normalized

    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("message_type", "supervisor_plan")

    task_fields = (
        "title",
        "objective",
        "instructions",
        "acceptance_criteria",
    )
    if "action" not in normalized and all(
        isinstance(normalized.get(field), str) and normalized[field].strip()
        for field in task_fields
    ):
        normalized["action"] = "NEXT_TASK"

    if normalized.get("action") == "NEXT_TASK":
        normalized.setdefault("blocking_reason", None)
    return normalized
