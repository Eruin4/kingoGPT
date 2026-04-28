from typing import Any

from internal_agent.tools.python_runner import run_python
from internal_agent.tools.search_docs import search_docs


TOOLS = {
    "search_docs": search_docs,
    "run_python": run_python,
}


def execute_tool(action: str, args: dict[str, Any]) -> str:
    if action not in TOOLS:
        raise ValueError(f"Unknown tool: {action}")
    if not isinstance(args, dict):
        raise ValueError("Tool args must be object")

    return TOOLS[action](**args)

