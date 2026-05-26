import json
import re
import uuid
from typing import Any


STRICT_TOOL_CONTRACT = """CRITICAL INSTRUCTION: follow this tool protocol exactly.

You are connected to a client that executes tools on your behalf.
You do NOT have a Python sandbox, file system, or shell.
You CANNOT run code yourself. The ONLY way to interact with the outside world
is by returning a JSON tool_call object.

When the user asks for information that requires accessing files, running commands,
listing directories, checking the environment, or any action you cannot answer
from conversation alone, you MUST respond with ONLY this JSON object and nothing else:

{"type":"tool_call","name":"TOOL_NAME_HERE","arguments":{...}}

When you can answer from conversation context alone, respond with ONLY:
{"type":"final","content":"your answer here"}

Rules:
- Your entire response must be a single JSON object. No markdown. No explanation.
- NEVER use code fences or code blocks of any language.
- NEVER use HTML comment syntax.
- NEVER use internal tool syntax of any kind.
- NEVER write Python code like os.listdir() or subprocess.run().
- NEVER reference /mnt/data. You have no sandbox.
- NEVER fabricate tool results. Wait for the client to return the result.
- After a tool result is provided, answer based on that result using the final JSON format.

The tool catalog is listed below. Use ONLY these tools."""


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
        if not isinstance(required, list):
            required = []
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
                f"- {name}: {description} | args: {argument_text} | "
                f"example: {example}"
            )
        else:
            lines.append(f"- {name}: args: {argument_text} | example: {example}")
    lines.append("\nRespond with ONLY a JSON object. No other text.")
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

    # Alternative shapes the model sometimes produces
    if obj.get("type") == "function" and obj.get("name"):
        return _tool_call_message(obj["name"], obj.get("arguments", {}))

    if obj.get("call"):
        return _tool_call_message(obj.get("call"), obj.get("args", {}))

    if obj.get("function") and isinstance(obj["function"], dict):
        func = obj["function"]
        if func.get("name"):
            return _tool_call_message(func["name"], func.get("arguments", {}))

    if obj.get("tool") and isinstance(obj.get("tool"), str):
        return _tool_call_message(obj["tool"], obj.get("arguments", obj.get("args", {})))

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


def infer_tool_call_from_failed_response(
    raw: str,
    tools: list[dict[str, Any]],
    user_text: str,
) -> dict[str, Any] | None:
    """Best-effort heuristic: extract intent from a failed response and build a tool_call.

    When the KingoGPT model ignores the JSON tool_call contract and instead writes
    Python code, bash commands, or ``<!-- tools: ... -->`` comments, this function
    tries to figure out what tool it *meant* to call and returns a synthetic
    OpenAI-format tool_call message.  Returns ``None`` if no confident match.
    """
    functions = {}
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        func = tool.get("function") or {}
        name = func.get("name")
        if isinstance(name, str) and name:
            functions[name] = func

    if not functions:
        return None

    user_lowered = user_text.lower()

    # --- Heuristic 1: extract a shell command from code blocks in the response ---
    shell_command = None
    bash_match = re.search(r'```(?:bash|sh|shell)\s*\n(.+?)\n```', raw, re.DOTALL)
    if bash_match:
        shell_command = bash_match.group(1).strip().splitlines()[0].strip()

    # Try Python subprocess patterns
    if not shell_command:
        sub_match = re.search(
            r'subprocess\.(?:run|call|check_output)\(\[?["\']([\w\s./-]+'  # noqa: E501
            r'["\'])',
            raw,
        )
        if sub_match:
            shell_command = sub_match.group(1)

    # Try os.listdir / os.getcwd / ls patterns inside Python code
    if not shell_command:
        if 'os.listdir' in raw or 'os.getcwd' in raw:
            shell_command = 'ls -la'
        elif re.search(r'os\.(?:system|popen)\(["\'](.*?)["\']\)', raw):
            match = re.search(r'os\.(?:system|popen)\(["\'](.*?)["\']\)', raw)
            if match:
                shell_command = match.group(1)

    if shell_command and "terminal" in functions:
        return _tool_call_message("terminal", {"command": shell_command})

    # --- Heuristic 2: user asked for files/directory listing ---
    # Only match against user_text, not the raw error response.
    # Use regex to match word combinations with intervening words.
    dir_match = (
        re.search(r'\blist\b.*\b(?:file|dir|folder)', user_lowered)
        or re.search(r'\bfiles?\b.*\b(?:in|of|on|from)\b.*\b(?:dir|folder|home)', user_lowered)
        or re.search(r'\b(?:what|show|which)\b.*\bfiles?\b', user_lowered)
        or re.search(r'\b(?:current|working|home)\s+directory\b', user_lowered)
        or re.search(r'\bls\b', user_lowered)
        or re.search(r'파일.{0,4}(?:목록|리스트|뭐|어떤|있)', user_lowered)
        or re.search(r'디렉토리.{0,4}(?:목록|내용|확인)', user_lowered)
        or re.search(r'폴더.{0,4}(?:목록|내용)', user_lowered)
    )
    if dir_match:
        if "search_files" in functions:
            return _tool_call_message("search_files", {
                "pattern": "*", "target": "files", "path": ".", "limit": 50,
            })
        if "terminal" in functions:
            return _tool_call_message("terminal", {"command": "ls -la"})

    # --- Heuristic 3: user asked to read a file ---
    read_keywords = {'read', 'show', 'cat', 'view', 'open', 'content', '읽', '보여', '내용'}
    if any(kw in user_lowered for kw in read_keywords) and "read_file" in functions:
        # Try to extract a filename from the user prompt
        file_match = re.search(
            r'["\']?([\w./-]+\.(?:py|md|txt|json|yaml|yml|toml|sh|js|ts|css|html))["\']?',
            user_text,
        )
        if file_match:
            return _tool_call_message("read_file", {"path": file_match.group(1)})

    # --- Heuristic 4: fallback for Python code that looks like file operations ---
    if '```python' in raw.lower() and "terminal" in functions:
        # The model wanted to run Python; map it to a terminal command
        python_match = re.search(r'```python\s*\n(.+?)\n```', raw, re.DOTALL)
        if python_match:
            code = python_match.group(1).strip()
            # If it's a short snippet, wrap in python -c
            if len(code) < 200 and '\n' not in code:
                return _tool_call_message("terminal", {"command": f'python3 -c "{code}"'})
            return _tool_call_message("terminal", {"command": "ls -la"})

    return None


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
