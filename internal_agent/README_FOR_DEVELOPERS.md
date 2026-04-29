# Developer Guide

This guide is for people changing the KingoGPT OpenAI-compatible provider.

## Current Architecture

```text
Hermes / OpenAI-compatible client
        |
        v
internal_agent.server.openai_compat
        |
        v
internal_agent.llm.AzureWebLLM
        |
        v
kingogpt_api_solver.py
        |
        v
KingoGPT web API
```

The server aims to make KingoGPT look like an OpenAI API provider. KingoGPT itself is still the text model; API compatibility, history trimming, and future tool orchestration are handled here.

## Active vs Experimental Code

The normal deployed Docker entry point is:

```text
internal_agent.server.openai_compat:app
```

Hermes should use `/v1/chat/completions` or `/v1/responses` on that app.
The `standalone/` package is not on the normal production path. It is experimental support code for a JSON-action agent loop and is used only by:

- `python -m internal_agent.standalone.app`
- `POST /v1/agent/chat/completions`
- unit tests around the old agent loop

## Important Invariants

- `/v1/chat/completions` is the primary Hermes path.
- `/v1/responses` is a minimal compatibility layer, not a complete hosted-tools implementation yet.
- `tools` and `tool_choice` are accepted and logged when debug is enabled, but hosted tools are not executed by this server yet.
- The standalone JSON-action agent is not a substitute for OpenAI tool calling until it is explicitly wired into the OpenAI-compatible endpoints.
- `AzureWebLLM` is serialized with a lock because it updates token/session cache state.
- Long conversations are trimmed before being sent upstream.
- System prompts are currently embedded into the KingoGPT query text by the adapter because the upstream `instruction` field has not behaved like a reliable OpenAI system prompt.
- `KINGOGPT_AUTO_DELETE_THREAD=false` is useful for debugging upstream KingoGPT chat rooms, but production may prefer deletion to avoid stale thread state.

## Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `KINGOGPT_TOKEN_CACHE` | `state/kingogpt_token_cache.json` | Token cache path inside the runtime. |
| `KINGOGPT_TOKEN_CONFIG` | `state/kingogpt_config.json` | Login credential config path. |
| `KINGOGPT_PROFILE_DIR` | `state/kingogpt_chrome_profile` | Playwright profile directory. |
| `KINGOGPT_CHAT_ROOM_ID` | unset | Force a specific KingoGPT room. |
| `KINGOGPT_SCENARIO_ID` | solver default | Override upstream scenario/workflow. |
| `KINGOGPT_REQUEST_TIMEOUT` | `120` | Upstream request read timeout in seconds. |
| `KINGOGPT_TOKEN_REFRESH_TIMEOUT` | `300` | Playwright token refresh timeout. |
| `KINGOGPT_NO_AUTO_REFRESH_TOKEN` | `false` | Disable automatic token refresh. |
| `KINGOGPT_IGNORE_EXPIRY` | `false` | Ignore token expiry checks. |
| `KINGOGPT_REUSE_THREAD` | `false` | Reuse upstream KingoGPT thread state. |
| `KINGOGPT_AUTO_DELETE_THREAD` | `true` | Delete upstream thread after each request unless reusing. |
| `KINGOGPT_ECHO_LLM` | `false` | Print upstream streaming chunks. |
| `KINGOGPT_DEBUG_OPENAI_REQUESTS` | `false` | Log endpoint/model/tools/tool_choice summaries. |
| `KINGOGPT_MAX_HISTORY_MESSAGES` | `16` | Max non-system history blocks sent upstream. |
| `KINGOGPT_MAX_PROMPT_CHARS` | `12000` | Max non-system prompt characters sent upstream. |

## Testing

Run focused tests after changing this package:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_runtime tests.test_kingogpt_api_solver
.\.venv\Scripts\python.exe -m py_compile internal_agent\server\openai_compat.py internal_agent\llm\azure_web_adapter.py kingogpt_api_solver.py
```

## Deployment Notes

Use `deploy/sync_to_server.ps1` and then `deploy/remote_bootstrap.sh` for normal deployment.
If the server has temporary runtime-only changes, copy individual files instead of syncing the whole repository.

## Roadmap For Full OpenAI Replacement

1. Observe Hermes requests with `KINGOGPT_DEBUG_OPENAI_REQUESTS=true`.
2. Confirm whether Hermes uses `/v1/chat/completions` or `/v1/responses` for tools.
3. Implement the missing OpenAI surface area in `server/openai_compat.py`.
4. Decide whether tools are executed by Hermes from `tool_calls` or hosted by this server.
5. If this server executes tools, add real sandboxing before enabling Python or shell execution.
