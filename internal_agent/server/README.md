# internal_agent.server

This package exposes KingoGPT as an OpenAI-compatible HTTP API using FastAPI.

## Entry Point

```bash
uvicorn internal_agent.server.openai_compat:app --host 0.0.0.0 --port 8000
```

Docker uses this entry point by default.
For local agent runtimes, the short alias is also available:

```bash
uvicorn kingogpt_openai_server:app --host 127.0.0.1 --port 8008
```

## Implemented Endpoints

| Endpoint | Status | Notes |
| --- | --- | --- |
| `GET /health` | Complete | Simple liveness check. |
| `GET /v1/models` | Minimal | Returns `kingogpt-web`. |
| `GET /v1/models/{model_id}` | Minimal | Returns `kingogpt-web` or OpenAI-style 404. |
| `POST /v1/chat/completions` | Primary | Main Hermes compatibility path. |
| `POST /v1/responses` | Minimal | Accepts Responses-style input/instructions/tools fields and emits OpenAI-style streaming events. |
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

When `tools` are supplied, the server appends a rigid JSON contract to the system
prompt. KingoGPT can then return either:

```json
{"type":"tool_call","name":"read","arguments":{"path":"README.md"}}
```

or:

```json
{"type":"final","content":"answer"}
```

The server maps `tool_call` responses to OpenAI-style `message.tool_calls`
with `assistant.content = null` and `finish_reason = "tool_calls"`.
The older `{"call":"exec","args":{...}}` and `{"reply":"..."}` shapes are still
accepted for compatibility.
When a client sends an assistant message with prior `tool_calls`, the server
strips non-OpenAI fields before rendering them back into the KingoGPT prompt.

History is trimmed with:

- `KINGOGPT_MAX_HISTORY_MESSAGES`
- `KINGOGPT_MAX_PROMPT_CHARS`

## Responses API Behavior

`/v1/responses` currently maps `input` and `instructions` into the same `AzureWebLLM` call path.
It accepts plain strings, message objects, and common content blocks such as `input_text`.
The response shape includes `output`, `output_text`, status, model, and approximate usage fields.

Hosted OpenAI tools are not executed here yet.
`tools` and `tool_choice` are accepted so clients do not fail on unknown fields.

Both Chat Completions and Responses streaming return complete text in a single delta because KingoGPT's upstream web API is wrapped as one completed answer.

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
