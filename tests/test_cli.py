from runtime.cli import _parse_validation, build_parser


def test_cli_parses_validation_command():
    command = _parse_validation("tests::python -m pytest tests/unit")
    assert command.name == "tests"
    assert command.argv == ("python", "-m", "pytest", "tests/unit")


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
