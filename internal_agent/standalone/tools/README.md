# internal_agent.standalone.tools

This package contains MVP tools for the standalone JSON-action agent.

## Registered Tools

| Tool | File | Purpose |
| --- | --- | --- |
| `search_docs` | `search_docs.py` | Placeholder search result for agent-loop testing. |
| `run_python` | `python_runner.py` | Very small restricted Python runner for demos/tests. |

`registry.py` maps action names to tool functions and validates that args are objects.

## Python Runner Limits

`run_python` is intentionally tiny:

- Only a small builtin set is exposed.
- Output is captured from `print`.
- Locals are returned for inspection.
- Exceptions are returned as text.

This is not a production sandbox. It must not be exposed as a hosted tool for untrusted clients without a real sandbox, filesystem limits, timeout limits, and output limits.

## Adding A Tool

1. Add a function in this package.
2. Register it in `registry.TOOLS`.
3. Add the action name to `standalone/agent/parser.py`.
4. Update `standalone/agent/prompts.py`.
5. Add tests in `tests/test_agent_runtime.py`.

