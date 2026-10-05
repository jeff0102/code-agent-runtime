"""Optional OpenHands SDK adapter for Executor conversations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from runtime.openhands_fallback import (
    FallbackProfileManager,
    OpenHandsLLMFallbackConfig,
    build_llm_kwargs,
    EXECUTOR_TIMEOUT_SECONDS,
)


class OpenHandsAdapterError(RuntimeError):
    """Raised when the OpenHands adapter cannot create or control a conversation."""


class ConversationLike(Protocol):
    """Minimal conversation interface required by the adapter."""

    id: UUID

    def send_message(self, message: str, sender: str | None = None) -> None: ...

    def run(self) -> None: ...

    def interrupt(self) -> None: ...

    def close(self) -> None: ...

    @property
    def state(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class OpenHandsExecutorConfig:
    """Configuration required to create an OpenHands Executor agent."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    fallbacks: tuple[OpenHandsLLMFallbackConfig, ...] = ()
    max_iteration_per_run: int = 500
    timeout: int = EXECUTOR_TIMEOUT_SECONDS
    persistence_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OpenHands model must not be empty")
        if self.max_iteration_per_run < 1:
            raise ValueError("max_iteration_per_run must be greater than zero")
        if self.timeout < 1:
            raise ValueError("timeout must be greater than zero")


@dataclass(frozen=True, slots=True)
class OpenHandsExecutionResult:
    """Project-agnostic result of an Executor conversation run."""

    conversation_id: str
    execution_status: str
    error: str | None = None


class OpenHandsConversationAdapter:
    """Thin wrapper around an OpenHands SDK Conversation."""

    def __init__(self, conversation: ConversationLike) -> None:
        self._conversation = conversation

    @property
    def conversation_id(self) -> str:
        return str(self._conversation.id)

    @property
    def execution_status(self) -> str:
        status = getattr(self._conversation.state, "execution_status", None)
        return getattr(status, "value", str(status))

    def send_message(self, message: str) -> None:
        if not message.strip():
            raise ValueError("Executor message must not be empty")
        self._conversation.send_message(message)

    def run(self) -> OpenHandsExecutionResult:
        try:
            self._conversation.run()
        except Exception as exc:  # noqa: BLE001
            return OpenHandsExecutionResult(
                conversation_id=self.conversation_id,
                execution_status=self.execution_status,
                error=str(exc),
            )

        return OpenHandsExecutionResult(
            conversation_id=self.conversation_id,
            execution_status=self.execution_status,
        )

    def send_and_run(self, message: str) -> OpenHandsExecutionResult:
        self.send_message(message)
        return self.run()

    def interrupt(self) -> None:
        self._conversation.interrupt()

    def close(self) -> None:
        self._conversation.close()


class OpenHandsExecutorFactory:
    """Create OpenHands Executor conversations on demand."""

    def __init__(self, config: OpenHandsExecutorConfig) -> None:
        self.config = config
        self._fallback_profiles: FallbackProfileManager | None = None

    def create(
        self,
        *,
        workspace_path: str | Path,
        conversation_id: str | UUID | None = None,
    ) -> OpenHandsConversationAdapter:
        """Create a new conversation or resume an existing one."""
        sdk = self._load_sdk()
        workspace = Path(workspace_path)
        if not workspace.is_dir():
            raise OpenHandsAdapterError(
                f"Executor workspace does not exist: {workspace}"
            )

        if self._fallback_profiles is None and self.config.fallbacks:
            self._fallback_profiles = FallbackProfileManager(
                sdk=sdk,
                fallbacks=self.config.fallbacks,
                usage_prefix="executor",
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
            tools=[
                sdk["Tool"](name=sdk["TerminalTool"].name),
                sdk["Tool"](name=sdk["FileEditorTool"].name),
            ],
        )

        kwargs: dict[str, Any] = {
            "agent": agent,
            "workspace": workspace,
            "max_iteration_per_run": self.config.max_iteration_per_run,
            "delete_on_close": False,
        }

        if self.config.persistence_dir is not None:
            kwargs["persistence_dir"] = str(self.config.persistence_dir)

        if conversation_id is not None:
            kwargs["conversation_id"] = self._coerce_uuid(conversation_id)

        try:
            conversation = sdk["Conversation"](**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise OpenHandsAdapterError(
                f"Failed to create OpenHands Executor conversation: {exc}"
            ) from exc

        return OpenHandsConversationAdapter(conversation)

    @staticmethod
    def _coerce_uuid(value: str | UUID) -> UUID:
        try:
            return value if isinstance(value, UUID) else UUID(value)
        except ValueError as exc:
            raise OpenHandsAdapterError(
                f"Invalid OpenHands conversation ID: {value!r}"
            ) from exc

    @staticmethod
    def _load_sdk() -> dict[str, Any]:
        try:
            from openhands.sdk import (
                Agent,
                Conversation,
                FallbackStrategy,
                LLM,
                LLMProfileStore,
                Tool,
            )
            from openhands.tools.file_editor import FileEditorTool
            from openhands.tools.terminal import TerminalTool
        except ImportError as exc:
            raise OpenHandsAdapterError(
                "OpenHands SDK dependencies are not installed. "
                'Install with: pip install -e ".[agents]"'
            ) from exc

        return {
            "Agent": Agent,
            "Conversation": Conversation,
            "FallbackStrategy": FallbackStrategy,
            "LLM": LLM,
            "LLMProfileStore": LLMProfileStore,
            "Tool": Tool,
            "FileEditorTool": FileEditorTool,
            "TerminalTool": TerminalTool,
        }
