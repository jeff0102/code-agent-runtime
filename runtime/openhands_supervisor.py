"""Optional OpenHands adapter for read-only Supervisor reviews."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Protocol
from uuid import UUID

from runtime.openhands_fallback import (
    FallbackProfileManager,
    OpenHandsLLMFallbackConfig,
    build_llm_kwargs,
    SUPERVISOR_TIMEOUT_SECONDS,
)
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
    def __init__(
        self,
        conversation: SupervisorConversationLike,
        *,
        timeout: int = SUPERVISOR_TIMEOUT_SECONDS,
    ) -> None:
        if timeout < 1:
            raise ValueError("timeout must be greater than zero")
        self._conversation = conversation
        self._timeout = timeout

    @property
    def conversation_id(self) -> str:
        return str(self._conversation.id)

    def review(self, prompt: str) -> OpenHandsSupervisorResult:
        if not prompt.strip():
            raise ValueError("Supervisor review prompt must not be empty")

        try:
            raw_response = self._ask_agent_with_timeout(prompt)
            decision = parse_supervisor_decision(raw_response)
        except (ProtocolError, ValueError) as exc:
            raise OpenHandsSupervisorError(
                f"Supervisor returned an invalid decision: {exc}"
            ) from exc
        except OpenHandsSupervisorError:
            raise
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
            raise ValueError("Supervisor plan prompt must not be empty")

        try:
            raw_response = self._ask_agent_with_timeout(prompt)
            plan = parse_supervisor_plan(raw_response)
        except (ProtocolError, ValueError) as exc:
            raise OpenHandsSupervisorError(
                f"Supervisor returned an invalid plan: {exc}"
            ) from exc
        except OpenHandsSupervisorError:
            raise
        except Exception as exc:
            raise OpenHandsSupervisorError(
                f"OpenHands Supervisor planning failed: {exc}"
            ) from exc

        return OpenHandsSupervisorPlanResult(
            conversation_id=self.conversation_id,
            plan=plan,
            raw_response=raw_response,
        )

    def _ask_agent_with_timeout(self, prompt: str) -> str:
        """Bound the synchronous SDK call and interrupt the conversation on timeout."""
        result_queue: Queue[tuple[str, str | BaseException]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                result_queue.put(("result", self._conversation.ask_agent(prompt)))
            except BaseException as exc:  # noqa: BLE001
                result_queue.put(("error", exc))

        thread = threading.Thread(
            target=worker,
            name="openhands-supervisor-call",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            try:
                self._conversation.interrupt()
            except Exception as exc:  # noqa: BLE001
                raise OpenHandsSupervisorError(
                    f"Supervisor timed out after {self._timeout}s and interrupt failed: {exc}"
                ) from exc
            raise OpenHandsSupervisorError(
                f"Supervisor timed out after {self._timeout}s"
            )

        kind, value = result_queue.get()
        if kind == "error":
            raise value
        return value

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

        return OpenHandsSupervisorAdapter(
            conversation,
            timeout=self.config.timeout,
        )

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
