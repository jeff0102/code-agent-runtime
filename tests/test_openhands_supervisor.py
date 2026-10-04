from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from runtime.openhands_supervisor import (
    OpenHandsSupervisorAdapter,
    OpenHandsSupervisorConfig,
    OpenHandsSupervisorError,
    OpenHandsSupervisorFactory,
    parse_supervisor_decision,
)
from runtime.protocol import SupervisorDecisionType


class FakeConversation:
    def __init__(self, response):
        self.id = UUID("33333333-3333-3333-3333-333333333333")
        self.response = response
        self.interrupted = False
        self.closed = False

    def ask_agent(self, question):
        self.question = question
        return self.response

    def interrupt(self):
        self.interrupted = True

    def close(self):
        self.closed = True


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeConversationFactory:
    created = []

    def __new__(cls, **kwargs):
        conversation = FakeConversation(
            '{"decision":"ACCEPT","task_complete":true,'
            '"instructions":[],"blocking_reason":null}'
        )
        conversation.kwargs = kwargs
        cls.created.append(conversation)
        return conversation


class FakeSDK:
    Agent = FakeAgent
    Conversation = FakeConversationFactory
    LLM = FakeLLM


def test_parse_supervisor_decision_accepts_plain_json():
    decision = parse_supervisor_decision(
        '{"decision":"ACCEPT","task_complete":true,"instructions":[],"blocking_reason":null}'
    )
    assert decision.decision is SupervisorDecisionType.ACCEPT


def test_parse_supervisor_decision_accepts_markdown_fence():
    response = (
        chr(96) * 3
        + 'json\n{"decision":"BLOCK","task_complete":false,'
        '"instructions":[],"blocking_reason":"Tests are unavailable."}\n'
        + chr(96) * 3
    )
    decision = parse_supervisor_decision(response)
    assert decision.decision is SupervisorDecisionType.BLOCK
    assert decision.blocking_reason == "Tests are unavailable."


def test_parse_supervisor_decision_rejects_invalid_json():
    with pytest.raises(Exception):
        parse_supervisor_decision("not json")


def test_config_requires_model():
    with pytest.raises(ValueError, match="model"):
        OpenHandsSupervisorConfig(model="")


def test_adapter_reviews_and_controls_conversation():
    conversation = FakeConversation(
        '{"decision":"REVISE","task_complete":false,'
        '"instructions":["Fix the parser."],"blocking_reason":null}'
    )
    adapter = OpenHandsSupervisorAdapter(conversation)

    result = adapter.review("Review this implementation.")

    assert result.conversation_id == str(conversation.id)
    assert result.decision.decision is SupervisorDecisionType.REVISE
    assert result.raw_response.startswith('{"decision"')

    adapter.interrupt()
    adapter.close()
    assert conversation.interrupted
    assert conversation.closed


def test_factory_creates_agent_without_tools(monkeypatch, tmp_path):
    FakeConversationFactory.created.clear()
    monkeypatch.setattr(
        OpenHandsSupervisorFactory,
        "_load_sdk",
        staticmethod(lambda: {
            "Agent": FakeAgent,
            "Conversation": FakeConversationFactory,
            "LLM": FakeLLM,
        }),
    )

    factory = OpenHandsSupervisorFactory(
        OpenHandsSupervisorConfig(
            model="openai/gemini-primary",
            api_key="sk-dummy",
            base_url="http://litellm:4000/v1",
            persistence_dir=tmp_path / "sessions",
        )
    )
    adapter = factory.create(
        reviewer_workspace=tmp_path,
        conversation_id="44444444-4444-4444-4444-444444444444",
    )

    conversation = FakeConversationFactory.created[-1]
    agent = conversation.kwargs["agent"]

    assert adapter.conversation_id == str(conversation.id)
    assert agent.kwargs["tools"] == []
    assert agent.kwargs["llm"].kwargs["model"] == "openai/gemini-primary"
    assert conversation.kwargs["persistence_dir"] == str(tmp_path / "sessions")
    assert conversation.kwargs["conversation_id"] == UUID(
        "44444444-4444-4444-4444-444444444444"
    )


def test_factory_rejects_missing_reviewer_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(
        OpenHandsSupervisorFactory,
        "_load_sdk",
        staticmethod(lambda: FakeSDK.__dict__),
    )
    factory = OpenHandsSupervisorFactory(
        OpenHandsSupervisorConfig(model="test")
    )

    with pytest.raises(OpenHandsSupervisorError, match="workspace"):
        factory.create(reviewer_workspace=tmp_path / "missing")
