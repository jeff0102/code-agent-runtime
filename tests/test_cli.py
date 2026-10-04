from runtime.cli import _parse_validation, _read_fallbacks_from_env, build_parser


def test_cli_parses_validation_command():
    command = _parse_validation("tests::python -m pytest tests/unit")
    assert command.name == "tests"
    assert command.argv == ("python", "-m", "pytest", "tests/unit")


def test_cli_reads_numbered_fallbacks_from_environment(monkeypatch):
    monkeypatch.setenv("OPENHANDS_EXECUTOR_FALLBACK_1_MODEL", "xai/grok-4.7")
    monkeypatch.setenv(
        "OPENHANDS_EXECUTOR_FALLBACK_1_API_KEY",
        "sk-grok",
    )
    monkeypatch.setenv(
        "OPENHANDS_EXECUTOR_FALLBACK_1_BASE_URL",
        "https://api.x.ai/v1",
    )
    monkeypatch.setenv(
        "OPENHANDS_EXECUTOR_FALLBACK_2_MODEL",
        "gemini/gemini-3.7-flash",
    )
    monkeypatch.setenv(
        "OPENHANDS_EXECUTOR_FALLBACK_2_API_KEY",
        "sk-gemini",
    )

    fallbacks = _read_fallbacks_from_env("OPENHANDS_EXECUTOR")

    assert [(item.model, item.api_key, item.base_url) for item in fallbacks] == [
        ("xai/grok-4.7", "sk-grok", "https://api.x.ai/v1"),
        ("gemini/gemini-3.7-flash", "sk-gemini", None),
    ]


def test_cli_requires_workspace():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--workspace",
            "/tmp/repo",
            "--executor-model",
            "provider/model",
            "--supervisor-model",
            "provider/model",
        ]
    )

    assert str(args.workspace) == "/tmp/repo"
    assert args.max_iterations == 5
