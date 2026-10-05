"""Shared helpers for configuring OpenHands LLM fallbacks without persisting secrets."""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any

# Runtime policy: give a provider a few chances, but fail over quickly enough
# that transient provider outages do not stall an autonomous development session.
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_WAIT_SECONDS = 60
LLM_RETRY_MULTIPLIER = 1.0
EXECUTOR_TIMEOUT_SECONDS = 120
SUPERVISOR_TIMEOUT_SECONDS = 90


@dataclass(frozen=True, slots=True)
class OpenHandsLLMFallbackConfig:
    """Configuration for one OpenHands fallback LLM."""

    model: str
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OpenHands fallback model must not be empty")


def build_llm_kwargs(
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    timeout: int,
) -> dict[str, Any]:
    """Build keyword arguments for the OpenHands SDK LLM constructor."""
    if not model.strip():
        raise ValueError("OpenHands model must not be empty")
    if timeout < 1:
        raise ValueError("OpenHands timeout must be greater than zero")

    kwargs: dict[str, Any] = {
        "model": model,
        "num_retries": LLM_RETRY_ATTEMPTS,
        "retry_min_wait": LLM_RETRY_WAIT_SECONDS,
        "retry_max_wait": LLM_RETRY_WAIT_SECONDS,
        "retry_multiplier": LLM_RETRY_MULTIPLIER,
        "timeout": timeout,
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    return kwargs


class FallbackProfileManager:
    """Create temporary SDK profiles so fallback credentials are never persisted in runtime state."""

    def __init__(
        self,
        *,
        sdk: dict[str, Any],
        fallbacks: tuple[OpenHandsLLMFallbackConfig, ...],
        usage_prefix: str,
        timeout: int,
    ) -> None:
        self._sdk = sdk
        self._fallbacks = fallbacks
        self._usage_prefix = usage_prefix
        if timeout < 1:
            raise ValueError("OpenHands timeout must be greater than zero")
        self._timeout = timeout
        self._profile_store_dir: TemporaryDirectory[str] | None = None
        self._profile_names: list[str] = []

        if not fallbacks:
            return

        self._profile_store_dir = TemporaryDirectory(
            prefix=f"code-agent-runtime-{usage_prefix}-fallbacks-"
        )
        store = sdk["LLMProfileStore"](base_dir=self._profile_store_dir.name)

        for index, fallback in enumerate(fallbacks, start=1):
            profile_name = f"{usage_prefix}-fallback-{index}"
            llm = sdk["LLM"](
                usage_id=profile_name,
                **build_llm_kwargs(
                    model=fallback.model,
                    api_key=fallback.api_key,
                    base_url=fallback.base_url,
                    timeout=self._timeout,
                ),
            )
            store.save(profile_name, llm, include_secrets=True)
            self._profile_names.append(profile_name)

    def strategy(self) -> Any | None:
        """Return an OpenHands FallbackStrategy, or None when no fallbacks are configured."""
        if self._profile_store_dir is None:
            return None

        return self._sdk["FallbackStrategy"](
            fallback_llms=list(self._profile_names),
            profile_store_dir=self._profile_store_dir.name,
        )
