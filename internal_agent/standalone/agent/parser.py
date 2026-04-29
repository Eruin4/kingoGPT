import json
import re
from typing import Any


ALLOWED_ACTIONS = {"search_docs", "run_python", "final"}


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from imperfect model output."""
    stripped = _strip_markdown_fence(text)
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None

    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(obj, dict):
            raise ValueError("JSON value must be an object")
        return obj

    if last_error is not None:
        raise ValueError(f"No valid JSON object found: {last_error}") from last_error
    raise ValueError("No JSON object found")


def validate_action(obj: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Action must be object")

    if "action" not in obj and isinstance(obj.get("reply"), str):
        return {"action": "final", "args": {"answer": obj["reply"]}}

    action = obj.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("Missing action")

    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    args = obj.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("args must be object")

    if action == "final":
        answer = args.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('final args must include non-empty string "answer"')

    obj["args"] = args
    return obj
