from pathlib import Path

import pytest

from runtime.openhands_fallback import (
    FallbackProfileManager,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_MULTIPLIER,
    LLM_RETRY_WAIT_SECONDS,
    EXECUTOR_TIMEOUT_SECONDS,
    SUPERVISOR_TIMEOUT_SECONDS,
    OpenHandsLLMFallbackConfig,
    build_llm_kwargs,
)


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeProfileStore:
    created = []

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.saved = []
        self.created.append(self)

    def save(self, name, llm, include_secrets=False):
        self.saved.append((name, llm, include_secrets))


class FakeFallbackStrategy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_sdk():
    return {
        "LLM": FakeLLM,
        "LLMProfileStore": FakeProfileStore,
        "FallbackStrategy": FakeFallbackStrategy,
    }


def test_build_llm_kwargs_sets_fast_retry_policy():
    assert build_llm_kwargs(
        model="gemini/gemini-3.8-flash",
        api_key=None,
        base_url=None,
        timeout=SUPERVISOR_TIMEOUT_SECONDS,
    ) == {
        "model": "gemini/gemini-3.8-flash",
        "num_retries": LLM_RETRY_ATTEMPTS,
        "retry_min_wait": LLM_RETRY_WAIT_SECONDS,
        "retry_max_wait": LLM_RETRY_WAIT_SECONDS,
        "retry_multiplier": LLM_RETRY_MULTIPLIER,
        "timeout": SUPERVISOR_TIMEOUT_SECONDS,
    }


def test_build_llm_kwargs_preserves_configured_values():
    kwargs = build_llm_kwargs(
        model="xai/grok-4.7",
        api_key="sk-dummy",
        base_url="https://api.x.ai/v1",
        timeout=EXECUTOR_TIMEOUT_SECONDS,
    )

    assert kwargs["model"] == "xai/grok-4.7"
    assert kwargs["api_key"] == "sk-dummy"
    assert kwargs["base_url"] == "https://api.x.ai/v1"
    assert kwargs["num_retries"] == 3
    assert kwargs["retry_min_wait"] == 60
    assert kwargs["retry_max_wait"] == 60
    assert kwargs["timeout"] == EXECUTOR_TIMEOUT_SECONDS


def test_fallback_manager_builds_ordered_profiles_without_persisting_to_state(
    tmp_path,
):
    FakeProfileStore.created.clear()

    manager = FallbackProfileManager(
        sdk=fake_sdk(),
        fallbacks=(
            OpenHandsLLMFallbackConfig(
                model="xai/grok-4.7",
                api_key="sk-grok",
                base_url="https://api.x.ai/v1",
            ),
            OpenHandsLLMFallbackConfig(
                model="gemini/gemini-3.7-flash",
                api_key="sk-gemini",
            ),
        ),
        usage_prefix="executor",
        timeout=EXECUTOR_TIMEOUT_SECONDS,
    )

    store = FakeProfileStore.created[-1]
    assert store.base_dir != tmp_path
    assert [item[0] for item in store.saved] == [
        "executor-fallback-1",
        "executor-fallback-2",
    ]
    assert all(item[2] is True for item in store.saved)
    assert all(
        item[1].kwargs["num_retries"] == LLM_RETRY_ATTEMPTS
        and item[1].kwargs["retry_min_wait"] == LLM_RETRY_WAIT_SECONDS
        and item[1].kwargs["retry_max_wait"] == LLM_RETRY_WAIT_SECONDS
        and item[1].kwargs["timeout"] == EXECUTOR_TIMEOUT_SECONDS
        for item in store.saved
    )

    strategy = manager.strategy()
    assert strategy is not None
    assert strategy.kwargs["fallback_llms"] == [
        "executor-fallback-1",
        "executor-fallback-2",
    ]
    assert strategy.kwargs["profile_store_dir"] == str(store.base_dir)


def test_fallback_config_rejects_empty_model():
    with pytest.raises(ValueError, match="fallback model"):
        OpenHandsLLMFallbackConfig(model="  ")
