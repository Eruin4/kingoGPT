# internal_agent.server

This package exposes KingoGPT as an OpenAI-compatible HTTP API using FastAPI.

## Entry Point

```bash
uvicorn internal_agent.server.openai_compat:app --host 0.0.0.0 --port 8000
```

Docker uses this entry point by default.

## Implemented Endpoints

| Endpoint | Status | Notes |
| --- | --- | --- |
| `GET /health` | Complete | Simple liveness check. |
| `GET /v1/models` | Minimal | Returns `kingogpt-web`. |
| `GET /v1/models/{model_id}` | Minimal | Returns `kingogpt-web` or OpenAI-style 404. |
| `POST /v1/chat/completions` | Primary | Main Hermes compatibility path. |
| `POST /v1/responses` | Minimal | Accepts Responses-style input/instructions/tools fields. |
| `POST /v1/agent/chat/completions` | Experimental | Routes to the standalone JSON-action agent. Not used by normal Hermes traffic. |

## Chat Completions Behavior

`messages` are converted into a plain prompt:

```text
USER:
...

ASSISTANT:
...

ASSISTANT:
```

System messages are collected separately and passed to `AzureWebLLM.complete`.
The adapter decides how to send them to KingoGPT.

This endpoint does not use `internal_agent.standalone` by default.

History is trimmed with:

- `KINGOGPT_MAX_HISTORY_MESSAGES`
- `KINGOGPT_MAX_PROMPT_CHARS`

## Responses API Behavior

`/v1/responses` currently maps `input` and `instructions` into the same `AzureWebLLM` call path.
The response shape includes `output`, `output_text`, status, and model fields.

Hosted OpenAI tools are not executed here yet.
`tools` and `tool_choice` are accepted so clients do not fail on unknown fields.

## Debugging Client Compatibility

Set:

```bash
KINGOGPT_DEBUG_OPENAI_REQUESTS=true
```

The server logs request summaries:

```text
openai_request endpoint=/v1/responses model=kingogpt-web stream=False tools=['web_search'] tool_choice='auto'
```

This is the quickest way to verify what Hermes sends.
