import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
}


CASES = [
    {
        "name": "shallow_direct_answer",
        "expect_tool_calls": False,
        "prompt": (
            "You have tools available if they are genuinely needed, but answer "
            "directly when you can. What is 2 + 2?"
        ),
    },
    {
        "name": "environment_inspection",
        "expect_tool_calls": True,
        "prompt": (
            "You have tools available if they are genuinely needed. What is the "
            "current working directory? Use a tool only if you need to inspect "
            "the environment."
        ),
    },
]


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_tool_calls(message: dict[str, Any], tools: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    available_names = tool_names(tools)
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return ["message.tool_calls must be a non-empty list"]
    for index, tool_call in enumerate(tool_calls):
        prefix = f"tool_calls[{index}]"
        if tool_call.get("type") != "function":
            errors.append(f"{prefix}.type must be 'function'")
        if not isinstance(tool_call.get("id"), str) or not tool_call["id"]:
            errors.append(f"{prefix}.id must be a non-empty string")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            errors.append(f"{prefix}.function must be an object")
            continue
        if function.get("name") not in available_names:
            errors.append(f"{prefix}.function.name must match one of the provided tools")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            errors.append(f"{prefix}.function.arguments must be a JSON string")
            continue
        try:
            decoded = json.loads(arguments)
        except Exception:
            errors.append(f"{prefix}.function.arguments must parse as JSON")
            continue
        if not isinstance(decoded, dict):
            errors.append(f"{prefix}.function.arguments must decode to an object")
        elif not isinstance(decoded.get("command"), str) or not decoded["command"]:
            errors.append(f"{prefix}.function.arguments.command must be a non-empty string")
    return errors


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "tools": [TERMINAL_TOOL],
        "tool_choice": "auto",
    }
    return post_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout=timeout)


def validate_choice(
    choice: dict[str, Any],
    *,
    expect_tool_calls: bool,
) -> tuple[list[str], bool]:
    message = choice["message"]
    has_tool_calls = bool(message.get("tool_calls"))
    errors: list[str] = []
    if expect_tool_calls != has_tool_calls:
        expected = "tool_calls" if expect_tool_calls else "no tool_calls"
        actual = "tool_calls" if has_tool_calls else "no tool_calls"
        errors.append(f"expected {expected}, got {actual}")
    if has_tool_calls:
        if message.get("content") is not None:
            errors.append("assistant.content must be null when tool_calls are present")
        if choice.get("finish_reason") != "tool_calls":
            errors.append("finish_reason must be 'tool_calls'")
        errors.extend(validate_tool_calls(message, [TERMINAL_TOOL]))
    elif choice.get("finish_reason") == "tool_calls":
        errors.append("finish_reason must not be 'tool_calls' without message.tool_calls")
    return errors, has_tool_calls


def run_case(base_url: str, model: str, case: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = chat_completion(
        base_url,
        model,
        [{"role": "user", "content": case["prompt"]}],
        timeout,
    )
    choice = response["choices"][0]
    message = choice["message"]
    errors, has_tool_calls = validate_choice(choice, expect_tool_calls=case["expect_tool_calls"])
    result = {
        "name": case["name"],
        "prompt": case["prompt"],
        "finish_reason": choice.get("finish_reason"),
        "has_tool_calls": has_tool_calls,
        "errors": errors,
        "message": message,
    }
    if case["expect_tool_calls"] and has_tool_calls and not errors:
        followup = run_followup(base_url, model, case["prompt"], message, timeout)
        result["followup"] = followup
        result["errors"].extend(f"followup: {error}" for error in followup["errors"])
    return result


def run_followup(
    base_url: str,
    model: str,
    original_prompt: str,
    assistant_message: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    tool_call = assistant_message["tool_calls"][0]
    tool_result = (
        "command: pwd\n"
        "exit_code: 0\n"
        "stdout:\n"
        "C:\\Users\\ppggh\\.antigravity\\kingoGPT\n"
    )
    response = chat_completion(
        base_url,
        model,
        [
            {"role": "user", "content": original_prompt},
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result,
            },
        ],
        timeout,
    )
    choice = response["choices"][0]
    message = choice["message"]
    errors, has_tool_calls = validate_choice(choice, expect_tool_calls=False)
    content = message.get("content") or ""
    if "kingoGPT" not in content and "antigravity" not in content:
        errors.append("final answer does not appear to use the tool result")
    return {
        "finish_reason": choice.get("finish_reason"),
        "has_tool_calls": has_tool_calls,
        "errors": errors,
        "message": message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe Discord-like tool behavior on the OpenAI-compatible KingoGPT server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", default="kingogpt-web")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    results = []
    try:
        for case in CASES:
            results.append(run_case(args.base_url, args.model, case, args.timeout))
    except urllib.error.URLError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps({"base_url": args.base_url, "results": results}, indent=2))
    return 1 if any(result["errors"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
