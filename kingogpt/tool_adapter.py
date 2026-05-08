import json
import uuid
from typing import Any


STRICT_TOOL_CONTRACT = """You may use client-side tools, but only when the task needs outside state.

If the user asks something you can answer from the conversation, answer normally.

If the user asks for current environment, files, shell output, or other information
you cannot know from the conversation, choose exactly one available tool by returning
only this JSON object:

{"type":"tool_call","name":"<tool_name>","arguments":{...}}

The client tool catalog below is the complete set of tools you may use. If you
would otherwise inspect the environment with Python, shell, markdown code, or a
hidden tool comment, return a tool_call for one of the listed tools instead.
Never use internal at-sign tool syntax.

After a tool result is provided, answer the user based on that result.

Do not claim to run tools yourself. Do not invent tool results. Do not include
markdown code blocks or HTML comments as substitutes for tool calls."""


def render_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
    blocks: list[str] = []

    if tools:
        blocks.append("CLIENT TOOL CATALOG\n" + json.dumps(tools, ensure_ascii=False))

    for message in messages:
        if not isinstance(message, dict):
            blocks.append(f"USER\n{_content_to_text(message)}")
            continue
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
        blocks.append("CLIENT TOOL CATALOG\n" + render_available_tools(tools))
    return "\n\n".join(blocks)


def render_available_tools(tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = function.get("description")
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") if isinstance(parameters, dict) else {}
        required = parameters.get("required") if isinstance(parameters, dict) else []
        argument_parts: list[str] = []
        if isinstance(properties, dict):
            for argument_name, schema in properties.items():
                if not isinstance(argument_name, str):
                    continue
                schema_type = schema.get("type") if isinstance(schema, dict) else "any"
                marker = "required" if argument_name in required else "optional"
                argument_parts.append(f"{argument_name}: {schema_type} ({marker})")
        argument_text = ", ".join(argument_parts) if argument_parts else "no arguments"
        example = json.dumps(
            {"type": "tool_call", "name": name, "arguments": example_arguments(parameters)},
            ensure_ascii=False,
        )
        if isinstance(description, str) and description:
            lines.append(
                f"Tool {name}: {description} Arguments: {argument_text}. "
                f"Example: {example}"
            )
        else:
            lines.append(f"Tool {name}: Arguments: {argument_text}. Example: {example}")
    return "\n".join(lines)


def example_arguments(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {}
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return {}
    required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
    names = [name for name in required if isinstance(name, str)]
    if not names:
        names = [name for name in properties if isinstance(name, str)]
    arguments: dict[str, Any] = {}
    for name in names[:3]:
        schema = properties.get(name)
        arguments[name] = example_argument_value(name, schema if isinstance(schema, dict) else {})
    return arguments


def example_argument_value(name: str, schema: dict[str, Any]) -> Any:
    if name in {"command", "cmd", "shell_command"}:
        return "pwd"
    if name in {"path", "directory", "cwd"}:
        return "."
    if name in {"pattern", "glob"}:
        return "*"
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0])
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return schema["enum"][0]
    if schema_type == "integer":
        return 30 if name == "timeout" else 1
    if schema_type == "number":
        return 1
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}
    return "value"


def convert_kingogpt_json_to_openai_message(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        obj = _extract_json_object(raw)
        if obj is None:
            return {"role": "assistant", "content": raw}
    if not isinstance(obj, dict):
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


def parse_kingogpt_json_message(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return _extract_json_object(raw)
    return obj if isinstance(obj, dict) else None


def sanitize_openai_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
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
