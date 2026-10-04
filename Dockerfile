FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY runtime /app/runtime
RUN pip install --no-cache-dir -e ".[agents]"

ENTRYPOINT ["code-agent-runtime"]
