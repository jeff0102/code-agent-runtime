"""Optional OpenHands adapter for read-only Supervisor reviews."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from pathlib import Path
from queue import Queue
from typing import Any, Protocol
from uuid import UUID

from runtime.openhands_fallback import (
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
        fallback_conversations: tuple[SupervisorConversationLike, ...] = (),
        fallback_storage: TemporaryDirectory[str] | None = None,
        timeout: int = SUPERVISOR_TIMEOUT_SECONDS,
    ) -> None:
        if timeout < 1:
            raise ValueError("timeout must be greater than zero")
        self._conversations = (conversation, *fallback_conversations)
        self._active_provider_index = 0
        self._fallback_storage = fallback_storage
        self._timeout = timeout

    @property
    def conversation_id(self) -> str:
        return str(self._conversations[self._active_provider_index].id)

    def review(self, prompt: str) -> OpenHandsSupervisorResult:
        if not prompt.strip():
            raise ValueError("Supervisor review prompt must not be empty")

        try:
            raw_response = self._ask_agent_with_provider_failover(prompt)
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
            raw_response = self._ask_agent_with_provider_failover(prompt)
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

    def _ask_agent_with_provider_failover(self, prompt: str) -> str:
        """Try each configured Supervisor provider, including timeout failures."""
        last_error: Exception | None = None

        for index, conversation in enumerate(self._conversations):
            self._active_provider_index = index
            try:
                return self._ask_agent_with_timeout(conversation, prompt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                try:
                    conversation.interrupt()
                except Exception:
                    pass

        if last_error is not None:
            raise last_error
        raise OpenHandsSupervisorError("No Supervisor conversations are configured.")

    def _ask_agent_with_timeout(
        self,
        conversation: SupervisorConversationLike,
        prompt: str,
    ) -> str:
        """Bound the synchronous SDK call and interrupt the conversation on timeout."""
        result_queue: Queue[tuple[str, str | BaseException]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                result_queue.put(("result", conversation.ask_agent(prompt)))
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
                conversation.interrupt()
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
        self._conversations[self._active_provider_index].interrupt()

    def close(self) -> None:
        try:
            for conversation in self._conversations:
                conversation.close()
        finally:
            if self._fallback_storage is not None:
                self._fallback_storage.cleanup()
                self._fallback_storage = None


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

        provider_configs = (
            {
                "model": self.config.model,
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                "conversation_id": conversation_id,
            },
            *(
                {
                    "model": fallback.model,
                    "api_key": fallback.api_key,
                    "base_url": fallback.base_url,
                    "conversation_id": None,
                }
                for fallback in self.config.fallbacks
            ),
        )

        conversations: list[SupervisorConversationLike] = []
        fallback_storage = (
            TemporaryDirectory(prefix="code-agent-runtime-supervisor-fallbacks-")
            if len(provider_configs) > 1
            else None
        )
        try:
            for provider_index, provider in enumerate(provider_configs):
                llm = sdk["LLM"](
                    **build_llm_kwargs(
                        model=provider["model"],
                        api_key=provider["api_key"],
                        base_url=provider["base_url"],
                        timeout=self.config.timeout,
                    )
                )
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
                if provider_index == 0:
                    if self.config.persistence_dir is not None:
                        kwargs["persistence_dir"] = str(self.config.persistence_dir)
                elif fallback_storage is not None:
                    fallback_path = Path(fallback_storage.name) / f"provider-{provider_index + 1}"
                    fallback_path.mkdir(parents=True, exist_ok=True)
                    kwargs["persistence_dir"] = str(fallback_path)
                if provider["conversation_id"] is not None:
                    kwargs["conversation_id"] = _coerce_uuid(provider["conversation_id"])

                try:
                    conversations.append(sdk["Conversation"](**kwargs))
                except Exception as exc:
                    raise OpenHandsSupervisorError(
                        f"Failed to create OpenHands Supervisor conversation "
                        f"for provider {provider_index + 1}: {exc}"
                    ) from exc
        except Exception:
            for conversation in conversations:
                try:
                    conversation.close()
                except Exception:
                    pass
            if fallback_storage is not None:
                fallback_storage.cleanup()
            raise

        return OpenHandsSupervisorAdapter(
            conversations[0],
            fallback_conversations=tuple(conversations[1:]),
            fallback_storage=fallback_storage,
            timeout=self.config.timeout,
        )

    @staticmethod
    def _load_sdk() -> dict[str, Any]:
        try:
            from openhands.sdk import (
                Agent,
                Conversation,
                LLM,
            )
        except ImportError as exc:
            raise OpenHandsSupervisorError(
                "OpenHands SDK dependencies are not installed. "
                'Install with: pip install -e ".[agents]"'
            ) from exc

        return {
            "Agent": Agent,
            "Conversation": Conversation,
            "LLM": LLM,
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
