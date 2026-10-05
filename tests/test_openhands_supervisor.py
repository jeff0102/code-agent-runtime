from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from runtime.openhands_fallback import OpenHandsLLMFallbackConfig
from runtime.openhands_supervisor import (
    OpenHandsSupervisorAdapter,
    OpenHandsSupervisorConfig,
    OpenHandsSupervisorError,
    OpenHandsSupervisorFactory,
    parse_supervisor_decision,
    parse_supervisor_plan,
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
        '{"schema_version":1,"message_type":"supervisor_decision","decision":"ACCEPT","task_complete":true,"instructions":[],"blocking_reason":null}'
    )
    assert decision.decision is SupervisorDecisionType.ACCEPT


def test_parse_supervisor_decision_accepts_markdown_fence():
    response = (
        chr(96) * 3
        + 'json\n{"schema_version":1,"message_type":"supervisor_decision","decision":"BLOCK","task_complete":false,'
        '"instructions":[],"blocking_reason":"Tests are unavailable."}\n'
        + chr(96) * 3
    )
    decision = parse_supervisor_decision(response)
    assert decision.decision is SupervisorDecisionType.BLOCK
    assert decision.blocking_reason == "Tests are unavailable."


def test_parse_supervisor_decision_rejects_invalid_json():
    with pytest.raises(Exception):
        parse_supervisor_decision("not json")

def test_parse_supervisor_decision_canonicalizes_observed_message_type_alias():
    decision = parse_supervisor_decision(
        '{"message_type":"SupervisorDecision","decision":"ACCEPT",'
        '"task_complete":true,"instructions":[],"blocking_reason":null}'
    )
    assert decision.decision is SupervisorDecisionType.ACCEPT
    assert decision.instructions == []


def test_parse_supervisor_decision_rejects_unrecognized_message_type():
    with pytest.raises(Exception, match="Invalid SupervisorDecision message_type"):
        parse_supervisor_decision(
            '{"schema_version":1,"message_type":"other","decision":"ACCEPT",'
            '"task_complete":true,"instructions":[],"blocking_reason":null}'
        )

def test_parse_supervisor_decision_accept_defaults_missing_instructions():
    decision = parse_supervisor_decision(
        '{"decision":"ACCEPT","task_complete":true,"blocking_reason":null}'
    )
    assert decision.decision is SupervisorDecisionType.ACCEPT
    assert decision.instructions == []


def test_parse_supervisor_decision_maps_required_revisions_to_instructions():
    decision = parse_supervisor_decision(
        '{"decision":"REVISE","summary":"Evidence is incomplete.",'
        '"rationale":["The endpoint test was not reported."],'
        '"required_revisions":["Run the endpoint test and report its result."]}'
    )
    assert decision.decision is SupervisorDecisionType.REVISE
    assert decision.instructions[-1] == "Run the endpoint test and report its result."
    assert "Evidence is incomplete." in decision.instructions[0]
    assert "The endpoint test was not reported." in decision.instructions[1]

@pytest.mark.parametrize(
    "payload,expected,complete",
    [
        ('{"decision":"ACCEPT","instructions":[]}', SupervisorDecisionType.ACCEPT, True),
        (
            '{"type":"SupervisorDecision","status":"REVISE","instructions":["Fix the issue."]}',
            SupervisorDecisionType.REVISE,
            False,
        ),
    ],
)
def test_parse_supervisor_decision_normalizes_known_response_shapes(
    payload, expected, complete
):
    decision = parse_supervisor_decision(payload)
    assert decision.decision is expected
    assert decision.task_complete is complete


def test_parse_supervisor_decision_block_still_requires_reason():
    with pytest.raises(Exception, match="blocking_reason"):
        parse_supervisor_decision('{"decision":"BLOCK","instructions":[]}')

def test_parse_supervisor_decision_unwraps_nested_decision():
    decision = parse_supervisor_decision(
        '{"type":"SupervisorDecision","decision":{'
        '"decision":"BLOCK","task_complete":false,"instructions":[],'
        '"blocking_reason":"Missing validation evidence."}}'
    )
    assert decision.decision is SupervisorDecisionType.BLOCK
    assert decision.blocking_reason == "Missing validation evidence."


def test_config_requires_model():
    with pytest.raises(ValueError, match="model"):
        OpenHandsSupervisorConfig(model="")


def test_adapter_reviews_and_controls_conversation():
    conversation = FakeConversation(
        '{"schema_version":1,"message_type":"supervisor_decision","decision":"REVISE","task_complete":false,'
        '"instructions":["Fix the parser."],"blocking_reason":null}'
    )
    adapter = OpenHandsSupervisorAdapter(conversation)

    result = adapter.review("Review this implementation.")

    assert result.conversation_id == str(conversation.id)
    assert result.decision.decision is SupervisorDecisionType.REVISE
    assert result.raw_response.startswith('{"schema_version"')

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
    assert agent.kwargs["llm"].kwargs["timeout"] == 90
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


def test_parse_supervisor_plan_accepts_next_task():
    plan = parse_supervisor_plan(
        '{"schema_version":1,"message_type":"supervisor_plan","action":"NEXT_TASK",'
        '"title":"Add configuration loader","objective":"Load runtime configuration.",'
        '"instructions":"Implement the loader.","acceptance_criteria":"Loader is tested.",'
        '"blocking_reason":null}'
    )
    assert plan.title == "Add configuration loader"


@pytest.mark.parametrize(
    "payload,expected_instruction,expected_detail",
    [
        (
            '{"type":"SupervisorTaskPlan","status":"TASK","task_id":"one",'
            '"title":"Bootstrap","objective":"Start app",'
            '"instructions":["Create the app."],"acceptance_criteria":["Tests pass."],'
            '"validation":["Run focused tests."],"constraints":["Skip deployment."]}',
            "- Create the app.",
            "Validation:\n- Run focused tests.",
        ),
        (
            '{"type":"SupervisorTaskPlan","status":"READY","task":{'
            '"title":"Bootstrap","objective":"Start app",'
            '"scope":["Add FastAPI app.","Expose GET /health."],'
            '"acceptance_criteria":["Health returns 200."],'
            '"validation":["Run focused tests."]}}',
            "- Add FastAPI app.\n- Expose GET /health.",
            "Validation:\n- Run focused tests.",
        ),
    ],
)
def test_parse_supervisor_task_plan_variants(
    payload, expected_instruction, expected_detail
):
    plan = parse_supervisor_plan(payload)
    assert plan.title == "Bootstrap"
    assert expected_instruction in plan.instructions
    assert expected_detail in plan.instructions
    assert plan.acceptance_criteria


def test_parse_supervisor_task_plan_accepts_latest_nested_scope_shape():
    payload = (
        '{"type":"SupervisorTaskPlan","status":"READY","task":{'
        '"title":"Implement the minimal FastAPI health endpoint",'
        '"objective":"Establish the first runnable bootstrap slice.",'
        '"scope":["Add the smallest reasonable FastAPI application structure.",'
        '"Expose a GET /health endpoint returning a stable success response."],'
        '"acceptance_criteria":["The application starts.","GET /health succeeds."],'
        '"validation":["Run focused tests."]}}'
    )
    plan = parse_supervisor_plan(payload)
    assert plan.title == "Implement the minimal FastAPI health endpoint"
    assert "GET /health" in plan.instructions


def test_adapter_review_error_preserves_raw_response():
    response = '{"decision":"UNKNOWN","instructions":[]}'
    adapter = OpenHandsSupervisorAdapter(FakeConversation(response))

    with pytest.raises(OpenHandsSupervisorError) as exc_info:
        adapter.review("Review this implementation.")

    assert exc_info.value.raw_response == response

def test_adapter_plans_next_task():
    response = (
        '{"schema_version":1,"message_type":"supervisor_plan","action":"NEXT_TASK",'
        '"title":"Add configuration loader","objective":"Load runtime configuration.",'
        '"instructions":"Implement the loader.","acceptance_criteria":"Loader is tested.",'
        '"blocking_reason":null}'
    )
    conversation = FakeConversation(response)
    adapter = OpenHandsSupervisorAdapter(conversation)

    result = adapter.plan("Plan the next task.")

    assert result.plan.title == "Add configuration loader"





def test_factory_attaches_configured_fallback_strategy(monkeypatch, tmp_path):
    class FakeProfileStore:
        def __init__(self, base_dir):
            self.base_dir = base_dir

        def save(self, name, llm, include_secrets=False):
            assert include_secrets is True

    class FakeFallbackStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    sdk = {
        "Agent": FakeAgent,
        "Conversation": FakeConversationFactory,
        "LLM": FakeLLM,
        "LLMProfileStore": FakeProfileStore,
        "FallbackStrategy": FakeFallbackStrategy,
    }
    monkeypatch.setattr(
        OpenHandsSupervisorFactory,
        "_load_sdk",
        staticmethod(lambda: sdk),
    )

    factory = OpenHandsSupervisorFactory(
        OpenHandsSupervisorConfig(
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

    factory.create(reviewer_workspace=tmp_path)
    llm = FakeConversationFactory.created[-1].kwargs["agent"].kwargs["llm"]

    assert isinstance(llm.kwargs["fallback_strategy"], FakeFallbackStrategy)
    assert llm.kwargs["fallback_strategy"].kwargs["fallback_llms"] == [
        "supervisor-fallback-1"
    ]
