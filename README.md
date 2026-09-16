# KingoGPT

KingoGPT provides a Python client and automation tools for the KingoGPT web service, an OpenAI-compatible API server, and a resumable workspace agent. The API server and agent can run independently. Hermes integration is maintained separately in [`Eruin4/hermingo`](https://github.com/Eruin4/hermingo), with the matching Hermes Agent patch documented under `integrations/hermes-agent/`.

## Features

- KingoGPT API client and token refresh through Playwright
- OpenAI-compatible Chat Completions and Responses endpoints, including streaming and function-call conversion
- Workspace agent with bounded file access, backups, resumable runs, command approval, and optional verification commands
- Optional iCampus data collection helpers

## Install

Python 3.10 or newer is required. For the API server and client:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m playwright install chromium
```

For the agent and test suite:

```bash
python -m pip install -e '.[agent,test]'
```

## Configure KingoGPT access

Create `state/kingogpt_config.json` locally with your account credentials. Keep the `state/` directory private; it contains credentials, token caches, browser profiles, and run data and is excluded from Git.

```json
{
  "id": "your-account-id",
  "password": "your-password"
}
```

Capture a token when needed:

```bash
kingogpt-token-capture \
  --cache-file state/kingogpt_token_cache.json \
  --config-file state/kingogpt_config.json \
  --profile-dir state/kingogpt_chrome_profile
```

## Run the OpenAI-compatible server

Set a strong API key in your local environment and bind the service to an address appropriate for your deployment:

```bash
export KINGOGPT_SERVER_API_KEY='replace-with-a-secret'
export KINGOGPT_SERVER_HOST=127.0.0.1
kingogpt-openai-server
```

The server provides `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/responses`. It supports streaming and validates supported function-call payloads. It does not execute functions supplied by API clients.

A Docker Compose example is included. Copy `deploy/openai-server.env.example` to a private local environment file before starting the service:

```bash
cp deploy/openai-server.env.example state/openai-server.env
docker compose up -d --build
```

## Run the workspace agent

The agent uses the KingoGPT API by default. Local mode and command execution must be selected explicitly:

```bash
kingogpt-agent --local --workspace /path/to/project 'Explain this project'
kingogpt-agent --local --workspace /path/to/project --allow-write \
  --exec ask --verify-command 'python -m unittest discover -s tests -q' \
  'Review the code, fix a bug, and verify the change'
```

Command execution runs with the host user's permissions; it is not an operating-system sandbox. `ask` requests approval for commands, while `deny` is the default. For long tasks, the agent saves a checkpoint that can be resumed with `--resume`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests use temporary workspaces and mocked upstream responses. The scripts under `scripts/` are optional smoke probes; they make live requests when run and may consume account or API quota.
