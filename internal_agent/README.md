# internal_agent

`internal_agent` contains the KingoGPT-backed runtime used by this repository.
The active production path is the OpenAI-compatible server.
It has two main faces:

- `server/`: the OpenAI-compatible HTTP surface used by Hermes and other clients.
- `standalone/`: a small JSON-action agent loop kept for local/internal testing.

The shared LLM adapter lives in `llm/`, and common defaults live in `config.py`.

## Package Map

| Path | Purpose |
| --- | --- |
| `config.py` | Shared paths and runtime defaults. |
| `llm/` | Adapter from internal prompts to the KingoGPT web API/SSE automation. |
| `server/` | FastAPI OpenAI-compatible API (`/v1/chat/completions`, `/v1/responses`, `/v1/models`). |
| `standalone/` | CLI and local JSON-action agent implementation. Not used by the normal production `/v1/chat/completions` path. |
| `standalone/agent/` | Planner loop, JSON parser, prompts, and step state. |
| `standalone/tools/` | MVP local tools used only by the standalone agent. |
| `tools/` | Legacy/empty namespace. New tool code should live under `standalone/tools/` unless promoted intentionally. |

## Runtime State

Runtime state is outside the package, under `state/` by default:

- `kingogpt_token_cache.json`
- `kingogpt_config.json`
- `kingogpt_chrome_profile/`

These paths are defined in `config.py` and can be overridden with environment variables in the server path.

## Production Entry Point

The Docker image starts:

```bash
uvicorn internal_agent.server.openai_compat:app --host 0.0.0.0 --port 8000
```

The standalone agent is not the default server path. It is kept for experiments and local agent-loop testing. It is reached only if a developer runs the CLI or calls the experimental `/v1/agent/chat/completions` endpoint.
