# internal_agent.standalone

This package contains a local CLI and standalone JSON-action agent.
It is not the production Hermes server path and is not used by `/v1/chat/completions`.

Production traffic goes through:

```text
internal_agent.server.openai_compat
```

The standalone package is invoked only when a developer runs the CLI or when the experimental `/v1/agent/chat/completions` endpoint is called directly.

## CLI

Run:

```bash
python -m internal_agent.standalone.app
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--max-steps N` | Limit tool/action loop steps. |
| `--reuse-thread` | Reuse upstream KingoGPT chat thread state. |
| `--keep-thread` | Do not delete the upstream KingoGPT thread after calls. |
| `--echo-llm` | Print upstream KingoGPT streaming output. |
| `--fake-response TEXT` | Use a static LLM response for local parser tests. |
| `--fake-final-answer TEXT` | Return a valid final JSON action with the given answer. |

## Runtime Flow

```text
Task input
   -> Agent.run()
   -> AzureWebLLM.complete() or StaticLLM
   -> JSON action
   -> local tool execution
   -> final answer
```

## When To Use

- Testing JSON-action prompts.
- Testing parser/repair behavior.
- Trying MVP local tools.
- Debugging agent-loop ideas before promoting them to the OpenAI-compatible server.

Do not assume this package is used by `/v1/chat/completions`; that endpoint uses `server/openai_compat.py` and `llm/AzureWebLLM`.
