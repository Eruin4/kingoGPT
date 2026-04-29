# internal_agent.standalone.agent

This package implements the MVP JSON-action agent loop.

## Files

| File | Purpose |
| --- | --- |
| `loop.py` | Agent loop, repair attempt, tool execution, max-step handling. |
| `parser.py` | Extracts and validates JSON action objects from imperfect model output. |
| `prompts.py` | Planner prompt and repair prompt builders. |
| `state.py` | Builds compact step-history entries. |

## Action Contract

The parser accepts:

```json
{"action":"search_docs","args":{"query":"..."}}
{"action":"run_python","args":{"code":"..."}}
{"action":"final","args":{"answer":"..."}}
```

It also accepts:

```json
{"reply":"..."}
```

and converts it to a `final` action.

## Repair Behavior

If the model output is not valid JSON or fails validation:

1. `Agent` builds a repair prompt with the bad output and error.
2. The LLM gets one repair attempt.
3. If repair still fails and text fallback is enabled, the original text becomes a final answer.

## Safety Notes

This is a lightweight experimental loop. Tool execution is local and intentionally small.
Do not expose new high-risk tools here without sandboxing and tests.

