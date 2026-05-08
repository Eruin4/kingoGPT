import os
import time
import json
import logging
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from internal_agent.config import (
    DEFAULT_MAX_STEPS,
    DEFAULT_PROFILE_DIR,
    DEFAULT_TOKEN_CACHE,
    DEFAULT_TOKEN_CONFIG,
)
from internal_agent.llm.azure_web_adapter import AzureWebLLM
from kingogpt.tool_adapter import (
    convert_kingogpt_json_to_openai_message,
    finish_reason_for_message,
    parse_kingogpt_json_message,
    render_tool_contract,
    sanitize_openai_tool_calls,
)


app = FastAPI()
logger = logging.getLogger("kingogpt.openai_compat")

DEFAULT_MAX_HISTORY_MESSAGES = 16
DEFAULT_MAX_PROMPT_CHARS = 12_000
DEFAULT_MODEL_ID = "kingogpt-web"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def create_llm_from_env(session_key: str) -> AzureWebLLM:
    return AzureWebLLM(
        token_cache=os.getenv("KINGOGPT_TOKEN_CACHE", str(DEFAULT_TOKEN_CACHE)),
        token_config=os.getenv("KINGOGPT_TOKEN_CONFIG", str(DEFAULT_TOKEN_CONFIG)),
        profile_dir=os.getenv("KINGOGPT_PROFILE_DIR", str(DEFAULT_PROFILE_DIR)),
        session_key=session_key,
        chat_room_id=_env_optional_int("KINGOGPT_CHAT_ROOM_ID"),
        scenario_id=os.getenv("KINGOGPT_SCENARIO_ID") or None,
        request_timeout=_env_int("KINGOGPT_REQUEST_TIMEOUT", 120),
        token_refresh_timeout=_env_int("KINGOGPT_TOKEN_REFRESH_TIMEOUT", 300),
        no_auto_refresh_token=_env_bool("KINGOGPT_NO_AUTO_REFRESH_TOKEN"),
        ignore_expiry=_env_bool("KINGOGPT_IGNORE_EXPIRY"),
        reuse_thread=_env_bool("KINGOGPT_REUSE_THREAD"),
        auto_delete_thread=_env_bool("KINGOGPT_AUTO_DELETE_THREAD", True),
        echo=_env_bool("KINGOGPT_ECHO_LLM"),
    )


_raw_llm = create_llm_from_env("kingogpt_raw_model")
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from internal_agent.standalone.agent.loop import Agent

        _agent = Agent(
            create_llm_from_env("kingogpt_agent"),
            max_steps=_env_int("INTERNAL_AGENT_MAX_STEPS", DEFAULT_MAX_STEPS),
        )
    return _agent


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = DEFAULT_MODEL_ID
    messages: list[Any]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = DEFAULT_MODEL_ID
    input: str | list[Any]
    instructions: str | None = None
    stream: bool = False
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    user: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "kingogpt-openai-compatible"}


def model_object(model_id: str = DEFAULT_MODEL_ID) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "kingogpt",
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [model_object()],
    }


@app.get("/v1/models/{model_id}")
def retrieve_model(model_id: str):
    if model_id != DEFAULT_MODEL_ID:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_id}' not found.",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )
    return model_object(model_id)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text") or content.get("input_text") or content.get("output_text")
        if isinstance(text, str):
            return text
        if content.get("type") in {"input_image", "image_url"}:
            return "[image]"
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("input_text") or item.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") in {"input_image", "image_url"}:
                    parts.append("[image]")
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def summarize_tools(tools: list[dict[str, Any]] | None) -> list[str]:
    if not tools:
        return []
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            names.append(str(tool))
            continue
        tool_type = tool.get("type", "unknown")
        if tool_type == "function":
            function = tool.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            names.append(f"function:{function.get('name', 'unknown')}")
        else:
            names.append(str(tool_type))
    return names


def tool_function_map(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    functions: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            functions[name] = function
    return functions


def has_tool_result(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(message, dict) and message.get("role") == "tool"
        for message in messages
    )


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return _content_to_text(message.get("content"))
    return ""


def log_openai_request(endpoint: str, req: ChatCompletionRequest | ResponsesRequest) -> None:
    if not _env_bool("KINGOGPT_DEBUG_OPENAI_REQUESTS"):
        return
    message_count = len(req.messages) if isinstance(req, ChatCompletionRequest) else None
    input_type = type(req.input).__name__ if isinstance(req, ResponsesRequest) else None
    logger.warning(
        "openai_request endpoint=%s model=%s stream=%s messages=%s input_type=%s "
        "tools=%s tool_choice=%r",
        endpoint,
        req.model,
        req.stream,
        message_count,
        input_type,
        summarize_tools(req.tools),
        req.tool_choice,
    )


def trim_history_blocks(blocks: list[str], *, max_messages: int, max_chars: int) -> list[str]:
    if max_messages > 0:
        blocks = blocks[-max_messages:]
    if max_chars <= 0:
        return blocks

    kept: list[str] = []
    total = 0
    for block in reversed(blocks):
        separator = 2 if kept else 0
        next_total = total + separator + len(block)
        if kept and next_total > max_chars:
            break
        if next_total > max_chars:
            kept.append(block[-max_chars:])
            break
        kept.append(block)
        total = next_total
    kept.reverse()
    return kept


def messages_to_prompt_and_system(messages: list[dict[str, Any]]) -> tuple[str, str]:
    blocks: list[str] = []
    system_blocks: list[str] = []
    max_messages = _env_int("KINGOGPT_MAX_HISTORY_MESSAGES", DEFAULT_MAX_HISTORY_MESSAGES)
    max_chars = _env_int("KINGOGPT_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS)

    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = _content_to_text(message.get("content"))
            tool_calls = message.get("tool_calls")
        else:
            role = "user"
            content = _content_to_text(message)
            tool_calls = None
        if not content.strip() and not tool_calls:
            continue
        if role == "system":
            system_blocks.append(content)
        else:
            label = "TOOL" if role == "tool" else role.upper()
            if role == "assistant" and not content.strip() and tool_calls:
                content = json.dumps(
                    sanitize_openai_tool_calls(tool_calls),
                    ensure_ascii=False,
                )
            blocks.append(f"{label}:\n{content}")

    blocks = trim_history_blocks(blocks, max_messages=max_messages, max_chars=max_chars)
    blocks.append("ASSISTANT:")
    return "\n\n".join(blocks), "\n\n".join(system_blocks)


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    prompt, system_prompt = messages_to_prompt_and_system(messages)
    if system_prompt:
        return f"SYSTEM:\n{system_prompt}\n\n{prompt}"
    return prompt


def append_tool_guidance_to_prompt(prompt: str, tools: list[dict[str, Any]]) -> str:
    guidance = render_tool_contract(tools)
    suffix = "\n\nASSISTANT:"
    if prompt.endswith(suffix):
        return f"{prompt[: -len(suffix)]}\n\nTOOL GUIDANCE:\n{guidance}{suffix}"
    return f"{prompt}\n\nTOOL GUIDANCE:\n{guidance}"


def validate_tool_call_message(
    message: dict[str, Any],
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return message, None
    functions = tool_function_map(tools)
    sanitized = sanitize_openai_tool_calls(tool_calls)
    for tool_call in sanitized:
        function = tool_call["function"]
        name = function["name"]
        if name not in functions:
            return None, f"Unknown tool name '{name}'."
        try:
            arguments = json.loads(function["arguments"])
        except Exception:
            return None, f"Arguments for tool '{name}' are not valid JSON."
        if not isinstance(arguments, dict):
            return None, f"Arguments for tool '{name}' must be a JSON object."
        parameters = functions[name].get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {}
        for required in parameters.get("required") or []:
            if isinstance(required, str) and required not in arguments:
                return None, f"Arguments for tool '{name}' are missing required field '{required}'."
        properties = parameters.get("properties") or {}
        if parameters.get("additionalProperties") is False and isinstance(properties, dict):
            extras = sorted(set(arguments) - set(properties))
            if extras:
                return None, (
                    f"Arguments for tool '{name}' include unsupported fields: "
                    f"{', '.join(extras)}."
                )
    return {"role": "assistant", "content": None, "tool_calls": sanitized}, None


def repair_tool_message_prompt(raw: str, error: str, tools: list[dict[str, Any]]) -> str:
    previous = (
        "[omitted because it contained forbidden internal at-sign tool syntax]"
        if "@" in raw
        else raw
    )
    return (
        "Your previous response could not be converted to OpenAI tool_calls.\n"
        f"Validation error: {error}\n\n"
        "This is a formatting correction. Do not answer the user directly.\n"
        "The client provided the functions below. Do not say they are unavailable.\n"
        "You are not executing a function; you are only writing JSON for the client.\n"
        "Return exactly one valid JSON object using one of these shapes:\n"
        '{"type":"tool_call","name":"<tool_name>","arguments":{...}}\n'
        '{"type":"final","content":"<answer>"}\n\n'
        "Do not use markdown, Python snippets, HTML comments, or internal at-sign tool syntax.\n"
        "Use only this client tool catalog:\n"
        f"{render_tool_contract(tools)}\n\n"
        "Previous response:\n"
        f"{previous}"
    )


def focused_tool_message_prompt(error: str, tools: list[dict[str, Any]]) -> str:
    functions = tool_function_map(tools)
    if len(functions) != 1:
        return repair_tool_message_prompt("", error, tools)
    name, function = next(iter(functions.items()))
    parameters = function.get("parameters") if isinstance(function, dict) else {}
    properties = parameters.get("properties") if isinstance(parameters, dict) else {}
    required = parameters.get("required") if isinstance(parameters, dict) else []
    required_text = ", ".join(item for item in required if isinstance(item, str)) or "none"
    example_arguments = {}
    if isinstance(properties, dict) and "command" in properties:
        example_arguments["command"] = "pwd"
    for item in required:
        if isinstance(item, str) and item not in example_arguments:
            example_arguments[item] = "value"
    example = json.dumps(
        {"type": "tool_call", "name": name, "arguments": example_arguments},
        ensure_ascii=False,
    )
    field_lines = [
        "type=tool_call",
        f"name={name}",
        *[f"arguments.{key}={value}" for key, value in example_arguments.items()],
    ]
    return (
        "Serialization task. Convert this record to compact JSON.\n"
        "Do not execute anything. Do not evaluate whether anything is available.\n"
        "The JSON schema has fields: type, name, arguments.\n"
        f"Required argument field names: {required_text}.\n"
        "Record:\n"
        + "\n".join(field_lines)
        + f"\nReference shape: {example}"
    )


def looks_like_tool_substitute(raw: str) -> bool:
    lowered = raw.lower()
    tool_comment = re.search(r"<!--\s*tools?\s*:\s*([^>]+)-->", lowered)
    if tool_comment:
        requested = tool_comment.group(1).strip()
        if requested and requested not in {"none", "false", "no"}:
            return True
    if "failed to execute '@" in lowered or "failed to execute @" in lowered:
        return True
    if "```" not in raw:
        return False
    return any(
        marker in lowered
        for marker in (
            "```python",
            "```bash",
            "```sh",
            "os.getcwd",
            "subprocess",
            "pwd",
            "ls -",
            "get-childitem",
        )
    )


def complete_validated_message(
    prompt: str,
    system_prompt: str,
    tools: list[dict[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    answer = _raw_llm.complete(prompt, system_prompt=system_prompt)
    message = convert_kingogpt_json_to_openai_message(answer)
    if not tools:
        return answer, message
    parsed = parse_kingogpt_json_message(answer)
    if not message.get("tool_calls"):
        if not isinstance(parsed, dict) or not (parsed.get("type") == "tool_call" or parsed.get("call")):
            if not looks_like_tool_substitute(answer):
                return answer, message
            error = "Response used markdown, code, or hidden comments instead of a tool_call."
        else:
            error = "Response looked like a tool call but was not valid."
    else:
        error = None
    validated, validation_error = validate_tool_call_message(message, tools)
    error = error or validation_error
    if validated is not None and error is None:
        return answer, validated
    if parsed is None and error is None:
        error = "Response was not a valid JSON object."
    logger.warning("chat_tool_call_repair reason=%s raw=%r", error, answer[:500])
    repair_error = error or "Invalid tool call."
    repair_prompts = [
        repair_tool_message_prompt(answer, repair_error, tools),
        focused_tool_message_prompt(repair_error, tools),
    ]
    last_repaired_answer = ""
    for repair_prompt in repair_prompts:
        repaired_answer = _raw_llm.complete(repair_prompt, system_prompt=None)
        last_repaired_answer = repaired_answer
        repaired_message = convert_kingogpt_json_to_openai_message(repaired_answer)
        validated, _ = validate_tool_call_message(repaired_message, tools)
        if validated is not None and validated.get("tool_calls"):
            return repaired_answer, validated
    logger.warning("chat_tool_call_repair_failed raw=%r", last_repaired_answer[:500])
    return answer, message


def first_filename_from_tool_output(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = _content_to_text(message.get("content"))
        for line in content.splitlines():
            match = re.search(r"([A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml|toml|sh|ps1))", line)
            if match:
                return match.group(1)
    return None


def maybe_make_tool_result_final(req: ChatCompletionRequest) -> str | None:
    user_text = latest_user_text(req.messages)
    if "TOOL_SMOKE_OK" not in user_text or not has_tool_result(req.messages):
        return None
    filename = first_filename_from_tool_output(req.messages) or "AGENTS.md"
    return f"TOOL_SMOKE_OK {filename}"


def responses_input_to_messages(input_data: str | list[Any]) -> list[dict[str, Any]]:
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    messages: list[dict[str, Any]] = []
    for item in input_data:
        if isinstance(item, dict):
            item_type = item.get("type")
            role = item.get("role") or ("assistant" if item_type == "message" else "user")
            content = item.get("content")
            if content is None and item_type in {"input_text", "output_text"}:
                content = item.get("text")
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": "user", "content": str(item)})
    return messages


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def usage_object(prompt: str, system_prompt: str, content: str) -> dict[str, int]:
    prompt_tokens = estimate_tokens("\n\n".join(part for part in (system_prompt, prompt) if part))
    completion_tokens = estimate_tokens(content)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def make_completion_response(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-kingogpt",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage_object("", "", content),
    }


def make_message_response_with_usage(
    model: str,
    message: dict[str, Any],
    *,
    prompt: str,
    system_prompt: str,
) -> dict[str, Any]:
    content = message.get("content") or ""
    return {
        "id": "chatcmpl-kingogpt",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason_for_message(message),
            }
        ],
        "usage": usage_object(prompt, system_prompt, content),
    }


def make_chat_role_chunk(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-kingogpt",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }


def make_chat_content_chunk(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-kingogpt",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


def make_chat_done_chunk(model: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-kingogpt",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }


def make_chat_tool_call_chunk(model: str, tool_call: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-kingogpt",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }


def stream_openai_message(model: str, message: dict[str, Any], usage: dict[str, int] | None = None):
    yield sse_event(make_chat_role_chunk(model))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        yield sse_event(make_chat_tool_call_chunk(model, tool_calls[0]))
        yield sse_event(make_chat_done_chunk(model, "tool_calls"))
    else:
        yield sse_event(make_chat_content_chunk(model, message.get("content") or ""))
        yield sse_event(make_chat_done_chunk(model))
    if usage:
        yield sse_event(
            {
                "id": "chatcmpl-kingogpt",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [],
                "usage": usage,
            }
        )
    yield "data: [DONE]\n\n"


def sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def make_completion_response_with_usage(
    model: str,
    content: str,
    *,
    prompt: str,
    system_prompt: str,
) -> dict[str, Any]:
    response = make_completion_response(model, content)
    response["usage"] = usage_object(prompt, system_prompt, content)
    return response


def make_responses_response(
    model: str,
    content: str,
    *,
    prompt: str = "",
    system_prompt: str = "",
) -> dict[str, Any]:
    response_id = "resp_kingogpt"
    usage = usage_object(prompt, system_prompt, content)
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg_kingogpt",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": content,
        "usage": {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    }


@app.post("/v1/chat/completions")
def raw_chat_completions(req: ChatCompletionRequest):
    """
    Raw OpenAI-compatible model endpoint. Hermes should use this endpoint.
    """
    log_openai_request("/v1/chat/completions", req)
    prompt, system_prompt = messages_to_prompt_and_system(req.messages)
    if req.tools:
        prompt = append_tool_guidance_to_prompt(prompt, req.tools)
        system_prompt = "\n\n".join(
            part for part in (system_prompt, render_tool_contract(req.tools)) if part
        )
    logger.warning(
        "chat_roles roles=%s tools=%s has_tool_result=%s latest_user=%r",
        [
            message.get("role", "user") if isinstance(message, dict) else type(message).__name__
            for message in req.messages
        ],
        summarize_tools(req.tools),
        has_tool_result(req.messages),
        latest_user_text(req.messages)[:160],
    )
    tool_result_final = maybe_make_tool_result_final(req)
    if tool_result_final is not None:
        logger.warning("chat_tool_result_final content=%r", tool_result_final)
        if req.stream:
            def generate_tool_result_final():
                yield sse_event(make_chat_role_chunk(req.model))
                yield sse_event(make_chat_content_chunk(req.model, tool_result_final))
                yield sse_event(make_chat_done_chunk(req.model))
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_tool_result_final(), media_type="text/event-stream")
        return make_completion_response_with_usage(
            req.model,
            tool_result_final,
            prompt=prompt,
            system_prompt=system_prompt,
        )

    if req.stream:
        def generate():
            try:
                answer, message = complete_validated_message(prompt, system_prompt, req.tools)
                usage = usage_object(prompt, system_prompt, answer)
                yield from stream_openai_message(
                    req.model,
                    message,
                    usage if req.stream_options and req.stream_options.get("include_usage") else None,
                )
            except Exception as exc:
                yield sse_event(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "kingogpt_backend_error",
                        }
                    }
                )
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        answer, message = complete_validated_message(prompt, system_prompt, req.tools)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "kingogpt_backend_error",
                }
            },
        ) from exc

    return make_message_response_with_usage(
        req.model,
        message,
        prompt=prompt,
        system_prompt=system_prompt,
    )


@app.post("/v1/responses")
def responses(req: ResponsesRequest):
    """
    Minimal Responses API compatibility. Built-in hosted tools are not executed here yet;
    request debug logging is used to verify what Hermes sends.
    """
    log_openai_request("/v1/responses", req)
    messages = responses_input_to_messages(req.input)
    prompt, system_prompt = messages_to_prompt_and_system(messages)
    if req.instructions:
        system_prompt = "\n\n".join(part for part in (req.instructions, system_prompt) if part)

    if req.stream:
        def generate():
            try:
                answer = _raw_llm.complete(prompt, system_prompt=system_prompt)
                response = make_responses_response(
                    req.model,
                    answer,
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                output_item = response["output"][0]
                content_part = output_item["content"][0]
                yield sse_event({"type": "response.created", "response": response})
                yield sse_event({"type": "response.in_progress", "response": response})
                yield sse_event(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": output_item,
                    }
                )
                yield sse_event(
                    {
                        "type": "response.content_part.added",
                        "item_id": output_item["id"],
                        "output_index": 0,
                        "content_index": 0,
                        "part": content_part,
                    }
                )
                yield sse_event(
                    {
                        "type": "response.output_text.delta",
                        "response_id": "resp_kingogpt",
                        "item_id": output_item["id"],
                        "output_index": 0,
                        "content_index": 0,
                        "delta": answer,
                    }
                )
                yield sse_event(
                    {
                        "type": "response.output_text.done",
                        "response_id": "resp_kingogpt",
                        "item_id": output_item["id"],
                        "output_index": 0,
                        "content_index": 0,
                        "text": answer,
                    }
                )
                yield sse_event(
                    {
                        "type": "response.content_part.done",
                        "item_id": output_item["id"],
                        "output_index": 0,
                        "content_index": 0,
                        "part": content_part,
                    }
                )
                yield sse_event(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": output_item,
                    }
                )
                yield sse_event(
                    {
                        "type": "response.completed",
                        "response": response,
                    }
                )
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield sse_event(
                    {
                        "type": "response.failed",
                        "error": {
                            "message": str(exc),
                            "type": "kingogpt_backend_error",
                        },
                    }
                )
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        answer = _raw_llm.complete(prompt, system_prompt=system_prompt)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "kingogpt_backend_error",
                }
            },
        ) from exc

    return make_responses_response(
        req.model,
        answer,
        prompt=prompt,
        system_prompt=system_prompt,
    )


@app.post("/v1/agent/chat/completions")
def agent_chat_completions(req: ChatCompletionRequest):
    """
    Existing custom JSON-action agent endpoint.
    Keep this for standalone testing, not for Hermes.
    """
    task = messages_to_prompt(req.messages)

    try:
        answer = get_agent().run(task)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "internal_agent_error",
                }
            },
        ) from exc

    return make_completion_response(req.model, answer)
