# internal_agent.llm

This package contains the LLM adapter used by the OpenAI-compatible server.
The standalone CLI can also instantiate it, but that is an experimental/local path rather than normal deployment.

## Main Class

`AzureWebLLM` in `azure_web_adapter.py` wraps `kingogpt_api_solver.py`.
It exposes a small interface:

```python
complete(prompt: str, *, system_prompt: str | None = None) -> str
stream(prompt: str) -> Iterator[str]
```

## Responsibilities

- Load or refresh the KingoGPT token cache.
- Resolve user/profile information required by the upstream API.
- Route prompts to `kingogpt.chat_via_api`.
- Maintain prompt-hash-based room/thread state in the token cache.
- Optionally reuse or delete upstream KingoGPT chat threads.
- Serialize calls with a lock so cache/thread state is not mutated concurrently.

## System Prompt Handling

The adapter currently embeds `system_prompt` into the prompt text as:

```text
SYSTEM:
...

USER:
...
```

It passes `instruction=None` to the upstream solver. This is intentional: the upstream `instruction` field has not behaved like a reliable OpenAI system-prompt channel during testing.

## Thread Behavior

- `reuse_thread=False`: do not reuse prior `chatThreadId`; system prompt is sent each request.
- `reuse_thread=True`: reuse cached `chatThreadId`; the system prompt is skipped after existing state is found.
- `auto_delete_thread=True`: delete the resolved upstream thread after each request, unless `reuse_thread=True`.

Use `auto_delete_thread=False` while debugging KingoGPT web chat rooms.
