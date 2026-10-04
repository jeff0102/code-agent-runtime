from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from runtime.openhands_fallback import OpenHandsLLMFallbackConfig
from runtime.openhands_executor import (
    OpenHandsAdapterError,
    OpenHandsConversationAdapter,
    OpenHandsExecutionResult,
    OpenHandsExecutorConfig,
    OpenHandsExecutorFactory,
)


class FakeConversation:
    def __init__(self):
        self.id = UUID("11111111-1111-1111-1111-111111111111")
        self.state = SimpleNamespace(
            execution_status=SimpleNamespace(value="finished"),
        )
        self.messages = []
        self.run_count = 0
        self.interrupted = False
        self.closed = False

    def send_message(self, message, sender=None):
        self.messages.append((message, sender))

    def run(self):
        self.run_count += 1

    def interrupt(self):
        self.interrupted = True

    def close(self):
        self.closed = True


class FakeTool:
    name = "fake-tool"

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeConversationFactory:
    created = []

    def __new__(cls, **kwargs):
        conversation = FakeConversation()
        conversation.kwargs = kwargs
        cls.created.append(conversation)
        return conversation


def fake_sdk():
    return {
        "Agent": FakeAgent,
        "Conversation": FakeConversationFactory,
        "LLM": FakeLLM,
        "Tool": lambda **kwargs: FakeTool(**kwargs),
        "FileEditorTool": FakeTool,
        "TerminalTool": FakeTool,
    }


def test_config_requires_model():
    with pytest.raises(ValueError, match="model"):
        OpenHandsExecutorConfig(model="")


def test_config_rejects_invalid_iteration_limit():
    with pytest.raises(ValueError, match="greater than zero"):
        OpenHandsExecutorConfig(model="test", max_iteration_per_run=0)


def test_adapter_wraps_conversation_lifecycle():
    conversation = FakeConversation()
    adapter = OpenHandsConversationAdapter(conversation)

    assert adapter.conversation_id == str(conversation.id)
    assert adapter.execution_status == "finished"

    adapter.send_message("Implement the task.")
    result = adapter.run()

    assert result == OpenHandsExecutionResult(
        conversation_id=str(conversation.id),
        execution_status="finished",
    )
    assert conversation.messages == [("Implement the task.", None)]
    assert conversation.run_count == 1

    adapter.interrupt()
    adapter.close()

    assert conversation.interrupted
    assert conversation.closed


def test_factory_builds_executor_with_only_write_tools(monkeypatch, tmp_path):
    FakeConversationFactory.created.clear()
    monkeypatch.setattr(
        OpenHandsExecutorFactory,
        "_load_sdk",
        staticmethod(fake_sdk),
    )

    factory = OpenHandsExecutorFactory(
        OpenHandsExecutorConfig(
            model="openai/gemini-primary",
            api_key="sk-dummy",
            base_url="http://litellm:4000/v1",
            max_iteration_per_run=42,
            persistence_dir=tmp_path / "sessions",
        )
    )

    adapter = factory.create(
        workspace_path=tmp_path,
        conversation_id="22222222-2222-2222-2222-222222222222",
    )

    created = FakeConversationFactory.created[-1]

    assert adapter.conversation_id == "11111111-1111-1111-1111-111111111111"
    assert created.kwargs["workspace"] == Path(tmp_path)
    assert created.kwargs["persistence_dir"] == str(tmp_path / "sessions")
    assert created.kwargs["delete_on_close"] is False
    assert created.kwargs["conversation_id"] == UUID(
        "22222222-2222-2222-2222-222222222222"
    )

    llm = created.kwargs["agent"].kwargs["llm"]
    assert llm.kwargs == {
        "model": "openai/gemini-primary",
        "api_key": "sk-dummy",
        "base_url": "http://litellm:4000/v1",
    }

    tool_names = [
        tool.kwargs["name"]
        for tool in created.kwargs["agent"].kwargs["tools"]
    ]
    assert tool_names == ["fake-tool", "fake-tool"]


def test_factory_rejects_missing_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(
        OpenHandsExecutorFactory,
        "_load_sdk",
        staticmethod(fake_sdk),
    )

    factory = OpenHandsExecutorFactory(OpenHandsExecutorConfig(model="test"))

    with pytest.raises(OpenHandsAdapterError, match="workspace"):
        factory.create(workspace_path=tmp_path / "missing")


def test_factory_rejects_invalid_conversation_id(monkeypatch, tmp_path):
    monkeypatch.setattr(
        OpenHandsExecutorFactory,
        "_load_sdk",
        staticmethod(fake_sdk),
    )

    factory = OpenHandsExecutorFactory(OpenHandsExecutorConfig(model="test"))

    with pytest.raises(OpenHandsAdapterError, match="Invalid OpenHands conversation ID"):
        factory.create(
            workspace_path=tmp_path,
            conversation_id="not-a-uuid",
        )


def test_adapter_converts_run_exception_to_result():
    conversation = FakeConversation()

    def failing_run():
        raise RuntimeError("model unavailable")

    conversation.run = failing_run
    adapter = OpenHandsConversationAdapter(conversation)

    result = adapter.run()

    assert result.conversation_id == str(conversation.id)
    assert result.error == "model unavailable"


def test_missing_optional_dependencies_have_a_clear_error(monkeypatch):
    def missing_sdk():
        raise OpenHandsAdapterError(
            'OpenHands SDK dependencies are not installed. Install with: pip install -e ".[agents]"'
        )

    monkeypatch.setattr(OpenHandsExecutorFactory, "_load_sdk", staticmethod(missing_sdk))

    factory = OpenHandsExecutorFactory(OpenHandsExecutorConfig(model="test"))

    with pytest.raises(OpenHandsAdapterError, match=r"\.\[agents\]"):
        factory.create(workspace_path=Path("."))



def test_factory_attaches_configured_fallback_strategy(monkeypatch, tmp_path):
    class FakeProfileStore:
        def __init__(self, base_dir):
            self.base_dir = base_dir

        def save(self, name, llm, include_secrets=False):
            assert include_secrets is True

    class FakeFallbackStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    sdk = fake_sdk()
    sdk.update(
        {
            "LLMProfileStore": FakeProfileStore,
            "FallbackStrategy": FakeFallbackStrategy,
        }
    )
    monkeypatch.setattr(
        OpenHandsExecutorFactory,
        "_load_sdk",
        staticmethod(lambda: sdk),
    )

    factory = OpenHandsExecutorFactory(
        OpenHandsExecutorConfig(
            model="gemini/gemini-3.8-flash",
            api_key="sk-gemini",
            fallbacks=(
                OpenHandsLLMFallbackConfig(
                    model="xai/grok-4.7",
                    api_key="sk-grok",
                    base_url="https://api.x.ai/v1",
                ),
            ),
        )
    )

    factory.create(workspace_path=tmp_path)
    llm = FakeConversationFactory.created[-1].kwargs["agent"].kwargs["llm"]

    assert isinstance(llm.kwargs["fallback_strategy"], FakeFallbackStrategy)
    assert llm.kwargs["fallback_strategy"].kwargs["fallback_llms"] == [
        "executor-fallback-1"
    ]
