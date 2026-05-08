import json
import uuid
from typing import Any


STRICT_TOOL_CONTRACT = """You are connected to an external agent runtime.

When you need an action, return exactly:

{"type":"tool_call","name":"<tool_name>","arguments":{...}}

When you are done, return exactly:

{"type":"final","content":"<answer>"}

No markdown. No prose outside JSON."""


def render_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
    blocks: list[str] = []

    if tools:
        blocks.append("AVAILABLE TOOLS\n" + json.dumps(tools, ensure_ascii=False))

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if role == "system":
            blocks.append(f"SYSTEM\n{_content_to_text(content)}")
        elif role == "user":
            blocks.append(f"USER\n{_content_to_text(content)}")
        elif role == "assistant":
            assistant_text = _content_to_text(content)
            if not assistant_text and message.get("tool_calls"):
                assistant_text = json.dumps(message["tool_calls"], ensure_ascii=False)
            blocks.append(f"ASSISTANT\n{assistant_text}")
        elif role == "tool":
            blocks.append(
                f"TOOL RESULT {message.get('tool_call_id', '')}\n{_content_to_text(content)}"
            )
        else:
            blocks.append(f"{role.upper()}\n{_content_to_text(content)}")

    return "\n\n".join(blocks)


def render_tool_contract(tools: list[dict[str, Any]] | None = None) -> str:
    blocks = [STRICT_TOOL_CONTRACT]
    if tools:
        blocks.append("AVAILABLE TOOLS\n" + json.dumps(tools, ensure_ascii=False))
    return "\n\n".join(blocks)


def convert_kingogpt_json_to_openai_message(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        obj = _extract_json_object(raw)
        if obj is None:
            return {"role": "assistant", "content": raw}

    if obj.get("type") == "final":
        return {"role": "assistant", "content": obj.get("content", "")}

    if obj.get("type") == "tool_call":
        return _tool_call_message(obj.get("name"), obj.get("arguments", {}))

    if obj.get("call"):
        return _tool_call_message(obj.get("call"), obj.get("args", {}))

    if "reply" in obj:
        return {"role": "assistant", "content": obj["reply"]}

    return {"role": "assistant", "content": raw}


def sanitize_openai_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") or {}
        sanitized.append(
            {
                "id": str(tool_call.get("id") or f"call_kingogpt_{index + 1}"),
                "type": "function",
                "function": {
                    "name": str(function.get("name") or ""),
                    "arguments": _arguments_to_json_string(function.get("arguments", {})),
                },
            }
        )
    return sanitized


def finish_reason_for_message(message: dict[str, Any]) -> str:
    return "tool_calls" if message.get("tool_calls") else "stop"


def _tool_call_message(name: Any, arguments: Any) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        return {
            "role": "assistant",
            "content": json.dumps(
                {"error": "Invalid KingoGPT tool call: missing function name"},
                ensure_ascii=False,
            ),
        }

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_kingogpt_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _arguments_to_json_string(arguments),
                },
            }
        ],
    }


def _arguments_to_json_string(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        try:
            json.loads(arguments)
            return arguments
        except Exception:
            return json.dumps({"value": arguments}, ensure_ascii=False)
    return json.dumps(arguments, ensure_ascii=False)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if "```" in text:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
            try:
                obj = json.loads(candidate)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("input_text") or item.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") in {"input_image", "image_url"}:
                    parts.append("[image]")
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text") or content.get("input_text") or content.get("output_text")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False)
    return "" if content is None else str(content)
