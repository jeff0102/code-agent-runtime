.DEFAULT_GOAL := help

.PHONY: help install run-wsl run-win recover

help:
	@echo "Available targets:"
	@echo "  install   - Upgrade pip and install runtime, agent, and development dependencies"
	@echo "  run-wsl   - Run the orchestrator with the configured WSL workspace and state paths"
	@echo "  run-win   - Run the orchestrator with the configured Windows workspace and state paths"
	@echo "  recover   - Stash tracked changes, switch to main, and reapply the stash"

install:
	python -m pip install --upgrade pip
	pip install -e ".[agents,dev]"

run-wsl:
	python3 -m runtime.cli --workspace /mnt/c/Users/jeffe/Projects/open-job-radar --state-path /mnt/c/Users/jeffe/PycharmProjects/code-agent-runtime/state --max-iterations 8 --max-tasks 100 --validation "pytest::python3 -m pytest -q" --enable-push

run-win:
	python -m runtime.cli --workspace "C:\Users\jeffe\Projects\open-job-radar" --state-path "C:\Users\jeffe\PycharmProjects\code-agent-runtime\state" --max-iterations 8 --max-tasks 100 --validation "pytest::python -m pytest -q" --enable-push

recover:
	git stash
	git checkout main
	-git stash pop
