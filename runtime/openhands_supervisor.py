"""Optional OpenHands adapter for read-only Supervisor reviews."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from runtime.protocol import ProtocolError, SupervisorDecision, SupervisorPlan


class OpenHandsSupervisorError(RuntimeError):
    """Raised when a Supervisor review cannot be completed."""


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
    persistence_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OpenHands Supervisor model must not be empty")


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

        try:
            raw_response = self._conversation.ask_agent(prompt)
            decision = parse_supervisor_decision(raw_response)
        except (ProtocolError, ValueError) as exc:
            raise OpenHandsSupervisorError(
                f"Supervisor returned an invalid decision: {exc}"
            ) from exc
        except Exception as exc:
            raise OpenHandsSupervisorError(
                f"OpenHands Supervisor review failed: {exc}"
            ) from exc

        return OpenHandsSupervisorResult(
            conversation_id=self.conversation_id,
            decision=decision,
            raw_response=raw_response,
        )

    def plan(self, prompt: str) -> OpenHandsSupervisorPlanResult:
        if not prompt.strip():
            raise ValueError("Supervisor planning prompt must not be empty")

        try:
            raw_response = self._conversation.ask_agent(prompt)
            plan = parse_supervisor_plan(raw_response)
        except (ProtocolError, ValueError) as exc:
            raise OpenHandsSupervisorError(
                f"Supervisor returned an invalid plan: {exc}"
            ) from exc
        except Exception as exc:
            raise OpenHandsSupervisorError(
                f"OpenHands Supervisor planning failed: {exc}"
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

        llm_kwargs: dict[str, Any] = {"model": self.config.model}
        if self.config.api_key is not None:
            llm_kwargs["api_key"] = self.config.api_key
        if self.config.base_url is not None:
            llm_kwargs["base_url"] = self.config.base_url

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
            from openhands.sdk import Agent, Conversation, LLM
        except ImportError as exc:
            raise OpenHandsSupervisorError(
                "OpenHands SDK dependencies are not installed. "
                'Install with: pip install -e ".[agents]"'
            ) from exc

        return {"Agent": Agent, "Conversation": Conversation, "LLM": LLM}


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

    return SupervisorDecision.from_dict(payload)



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

    return SupervisorPlan.from_dict(payload)
