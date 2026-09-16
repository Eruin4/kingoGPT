"""Translate external application decisions into validated OpenAI function calls.

KingoGPT's workflow owns its built-in tools. External functions are represented
as operation IDs so the model writes a decision for the caller to execute.
No operation is inferred from prose, examples, or the original user request.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry

from kingogpt.tool_adapter import _content_to_text

class ToolProtocolError(ValueError):
    pass


def operation_catalog(tools: list[dict]) -> dict[str, dict]:
    return {f"op_{i + 1}": tool["function"] for i, tool in enumerate(tools)}


def decision_prompt(messages: list[dict], tools: list[dict], tool_choice: Any) -> str:
    catalog = operation_catalog(tools)
    name_to_id = {function["name"]: key for key, function in catalog.items()}
    history = []
    goals = []
    instructions = []
    for message in messages:
        if message["role"] in ("system", "developer"):
            instructions.append(_content_to_text(message.get("content")))
            continue
        if message["role"] == "user":
            goals.append(_content_to_text(message.get("content")))
        item = {"role": message["role"], "content": message.get("content")}
        if message.get("tool_calls"):
            item["operations_requested"] = [
                {"id": call.get("id"),
                 "operation": name_to_id.get(call["function"]["name"], call["function"]["name"]),
                 "parameters": call["function"].get("arguments", "{}")}
                for call in message["tool_calls"]
            ]
        if message.get("tool_call_id"):
            item["operation_result_for"] = message["tool_call_id"]
        history.append(item)
    entries = [
        f"- {key} (application function {function['name']}): {function.get('description') or function['name']}. "
        f"Parameters JSON schema: {json.dumps(function.get('parameters') or {}, ensure_ascii=False)}"
        for key, function in catalog.items()
    ]
    requirement = "Select the operation needed to obtain missing information or perform the user's task."
    if tool_choice == "required" or isinstance(tool_choice, dict):
        requirement += " An operation is required this turn; finish is not allowed."
    return (
        "Choose the next operation for a separate local program that will execute it and return actual results.\n"
        "Task: " + (goals[-1] if goals else "Continue the task from the conversation below.") + "\n"
        "Application instructions: " + "\n".join(instructions) + "\n"
        "Operations supported by the application:\n" + "\n".join(entries) + "\n"
        "Results so far: " + (json.dumps(history, ensure_ascii=False) if history else "none.") + "\n"
        'Write a decision document: {"tools":[],"operation":"op_1","parameters":{...}} '
        'using the selected operation ID and its parameters, or {"tools":[],"operation":"finish","answer":"..."} '
        "when the task is complete. " + requirement + " "
        "Include the tools field as an empty array because this document requests no built-in tool from this chat. "
        "Produce exactly one decision object for one step. Code or document text requested by the user "
        "belongs inside parameter strings, never in a separate example or a separate response. "
        "After selecting an operation, stop and wait for its actual result from the application. "
        "Do not invent results. After an operation result, select the next operation if work remains."
    )


def validate_schema(schema: Any) -> None:
    if not isinstance(schema, dict):
        raise ValueError("Function parameters must be a JSON schema object.")
    Draft202012Validator.check_schema(schema)
    # External references could cause a network lookup while validating a call.
    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("$ref", "$dynamicRef") and isinstance(child, str) and not child.startswith("#"):
                    raise ValueError("Only local schema references are supported.")
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(schema)


def _reject_non_json_number(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def _decode_document(raw: str) -> dict:
    # Parse the outer document first: literal tags in file content are data.
    try:
        obj = json.loads(raw, parse_constant=_reject_non_json_number)
    except ValueError:
        sections = re.findall(r"^```(?:json)?\s*\n(.*?)\n```\s*$", raw, re.MULTILINE | re.DOTALL)
        if not sections:
            sections = re.findall(r"<decision>\s*([\s\S]*?)\s*</decision>", raw)
        if len(sections) != 1:
            raise ToolProtocolError("Expected exactly one JSON decision document.")
        try:
            obj = json.loads(sections[0], parse_constant=_reject_non_json_number)
        except ValueError as exc:
            raise ToolProtocolError("Expected a valid JSON decision document.") from exc
    if not isinstance(obj, dict):
        raise ToolProtocolError("Expected a JSON decision document.")
    return obj


def parse_decision(raw: str, tools: list[dict]) -> dict:
    obj = _decode_document(raw)
    catalog = operation_catalog(tools)
    operation = obj.get("operation")
    if operation == "finish" or obj.get("type") == "final":
        answer = obj.get("answer") if operation == "finish" else obj.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ToolProtocolError("A final decision needs a non-empty answer string.")
        return {"role": "assistant", "content": answer}
    if operation is not None:
        if not isinstance(operation, str) or operation not in catalog:
            raise ToolProtocolError("Unknown operation ID.")
        requests = [{"name": catalog[operation]["name"], "arguments": obj.get("parameters")}]
    elif obj.get("type") in ("tool_call", "function"):
        requests = [{"name": obj.get("name"), "arguments": obj.get("arguments")}]
    elif isinstance(obj.get("tool_calls"), list) and obj["tool_calls"]:
        requests = [call.get("function", call) if isinstance(call, dict) else {} for call in obj["tool_calls"]]
    else:
        raise ToolProtocolError("Expected an operation or a final answer.")

    functions = {function["name"]: function for function in catalog.values()}
    calls = []
    for request in requests:
        if not isinstance(request, dict):
            raise ToolProtocolError("Invalid function call object.")
        name, arguments = request.get("name"), request.get("arguments")
        if not isinstance(name, str) or name not in functions:
            raise ToolProtocolError("Unknown function name.")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments, parse_constant=_reject_non_json_number)
            except ValueError as exc:
                raise ToolProtocolError("Function arguments must be valid JSON.") from exc
        if not isinstance(arguments, dict):
            raise ToolProtocolError("Function arguments must be an object.")
        schema = functions[name].get("parameters") or {}
        try:
            errors = list(Draft202012Validator(schema, registry=Registry()).iter_errors(arguments))
        except Exception as exc:
            raise ToolProtocolError("Could not resolve the function parameters schema.") from exc
        if errors:
            # Only schema path/type is fed back; argument values may contain secrets.
            error = errors[0]
            location = ".".join(map(str, error.absolute_path)) or "arguments"
            raise ToolProtocolError(f"{name}: {location} violates {error.validator}.")
        calls.append({"id": f"call_{uuid.uuid4().hex}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
    return {"role": "assistant", "content": None, "tool_calls": calls}
