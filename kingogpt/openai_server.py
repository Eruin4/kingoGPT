"""Standalone OpenAI-compatible HTTP gateway for KingoGPT.

The gateway deliberately does not reuse KingoGPT chat threads.  Every request
contains the complete caller-provided conversation, which prevents context from
leaking between API clients and makes retries deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import re
import time
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kingogpt import api_solver
from kingogpt.tool_adapter import (
    finish_reason_for_message,
    render_messages,
)
from kingogpt.tool_protocol import (
    ToolProtocolError, decision_prompt, parse_decision, validate_schema,
)


class CompletionProvider(Protocol):
    def complete(
        self,
        prompt: str,
        instruction: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class ServerSettings:
    model_id: str = "kingogpt"
    api_key: str | None = None
    token_cache: Path = Path("state/kingogpt_token_cache.json")
    token_config: Path = Path("state/kingogpt_config.json")
    profile_dir: Path = Path("state/kingogpt_chrome_profile")
    scenario_id: str = api_solver.DEFAULT_SCENARIO_ID
    chat_room_id: int | None = None
    request_timeout: int = 120
    token_refresh_timeout: int = 300
    max_concurrency: int = 1
    tool_attempts: int = 3

    @classmethod
    def from_env(cls) -> "ServerSettings":
        room_value = os.getenv("KINGOGPT_CHAT_ROOM_ID", "").strip()
        return cls(
            model_id=os.getenv("KINGOGPT_MODEL_ID", "kingogpt").strip() or "kingogpt",
            api_key=os.getenv("KINGOGPT_SERVER_API_KEY") or None,
            token_cache=Path(os.getenv("KINGOGPT_TOKEN_CACHE", "state/kingogpt_token_cache.json")),
            token_config=Path(os.getenv("KINGOGPT_TOKEN_CONFIG", "state/kingogpt_config.json")),
            profile_dir=Path(os.getenv("KINGOGPT_PROFILE_DIR", "state/kingogpt_chrome_profile")),
            scenario_id=os.getenv("KINGOGPT_SCENARIO_ID", api_solver.DEFAULT_SCENARIO_ID),
            chat_room_id=int(room_value) if room_value else None,
            request_timeout=_positive_env_int("KINGOGPT_REQUEST_TIMEOUT", 120),
            token_refresh_timeout=_positive_env_int("KINGOGPT_TOKEN_REFRESH_TIMEOUT", 300),
            max_concurrency=_positive_env_int("KINGOGPT_MAX_CONCURRENCY", 1),
            tool_attempts=_positive_env_int("KINGOGPT_TOOL_ATTEMPTS", 3),
        )


class KingoGPTProvider:
    """A stateless provider backed by the existing KingoGPT web API client."""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self._auth_lock = threading.Lock()
        self._auth_state = None
        self._auth_time = 0
        self.args = argparse.Namespace(
            access_token=None,
            token_cache=str(settings.token_cache),
            token_config=str(settings.token_config),
            profile_dir=str(settings.profile_dir),
            token_refresh_timeout=settings.token_refresh_timeout,
            no_auto_refresh_token=False,
            chat_room_id=settings.chat_room_id,
            scenario_id=settings.scenario_id,
            request_timeout=settings.request_timeout,
            ignore_expiry=False,
        )

    def _credentials(self, rejected_token=None):
        # Share a validated identity between turns; refresh once when concurrent
        # requests reject the same token. Never cache across provider instances.
        with self._auth_lock:
            cached = self._auth_state
            if cached and (rejected_token is None or cached[1] != rejected_token):
                try:
                    api_solver.ensure_token_is_fresh(cached[1], ignore_expiry=False)
                    if time.monotonic() - self._auth_time < 60:
                        return cached
                except Exception:
                    pass
            if rejected_token is not None:
                cache = api_solver.refresh_token_cache(self.args)
                token = api_solver.resolve_access_token(self.args, cache)
                claims = api_solver.ensure_token_is_fresh(token, ignore_expiry=False)
                user = api_solver.fetch_user_profile(token)
                result = cache, token, claims, user
            else:
                result = api_solver.load_or_refresh_token(self.args)
            cache, token, claims, user = result
            if not user.get("id"):
                user = {**user, "id": claims.get("userId"), "userId": claims.get("userId")}
            self._auth_state = cache, token, claims, user
            self._auth_time = time.monotonic()
            return self._auth_state

    def complete(self, prompt, instruction=None, on_chunk=None) -> str:
        cache, token, _, user = self._credentials()
        emitted = False

        def emit(chunk):
            nonlocal emitted
            emitted = True
            if on_chunk is not None:
                on_chunk(chunk)

        for attempt in range(2):
            try:
                answer, _, _ = api_solver.chat_via_api(
                    token, user, prompt, self.args, instruction=instruction,
                    chat_room_id=self.settings.chat_room_id or cache.get("chat_room_id") or api_solver.DEFAULT_CHAT_ROOM_ID,
                    chat_thread_id=None, on_chunk=emit, verbose=False,
                )
                return answer
            except Exception as exc:
                # Retrying after visible text would splice two different answers.
                if attempt or emitted or not api_solver.should_auto_refresh_token(exc):
                    raise
                cache, token, _, user = self._credentials(rejected_token=token)


class OpenAIAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code


def create_app(
    *,
    settings: ServerSettings | None = None,
    provider: CompletionProvider | None = None,
) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    provider = provider or KingoGPTProvider(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    app = FastAPI(title="KingoGPT OpenAI-compatible API", version="0.2.0")
    app.state.settings = settings
    app.state.provider = provider

    @app.exception_handler(OpenAIAPIError)
    async def handle_openai_error(_request: Request, exc: OpenAIAPIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_error_payload(exc))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        wrapped = OpenAIAPIError(
            f"KingoGPT upstream request failed: {exc}",
            status_code=502,
            error_type="server_error",
            code="upstream_error",
        )
        return JSONResponse(status_code=wrapped.status_code, content=_error_payload(wrapped))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": settings.model_id}

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _authorize(settings, authorization)
        return {"object": "list", "data": [_model_object(settings.model_id)]}

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(
        model_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _authorize(settings, authorization)
        _validate_model(model_id, settings)
        return _model_object(settings.model_id)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Any:
        _authorize(settings, authorization)
        body = await _json_body(request)
        prepared = _prepare_chat_request(body, settings)
        if body.get("stream") is True:
            return StreamingResponse(
                _stream_chat_completion(provider, semaphore, settings, prepared, body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        raw, message = await _complete_message(provider, semaphore, settings, prepared)
        return _chat_completion_payload(
            model=settings.model_id,
            message=message,
            prompt=prepared["prompt"],
            raw=raw,
        )

    @app.post("/v1/responses")
    async def responses(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Any:
        _authorize(settings, authorization)
        body = await _json_body(request)
        chat_body = _responses_to_chat_request(body)
        prepared = _prepare_chat_request(chat_body, settings)
        if body.get("stream") is True:
            return StreamingResponse(
                _stream_response(provider, semaphore, settings, prepared, body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        raw, message = await _complete_message(provider, semaphore, settings, prepared)
        return _response_payload(
            model=settings.model_id,
            message=message,
            prompt=prepared["prompt"],
            raw=raw,
            request_body=body,
        )

    return app


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _authorize(settings: ServerSettings, authorization: str | None) -> None:
    if not settings.api_key:
        return
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if not hmac.compare_digest(provided, settings.api_key):
        raise OpenAIAPIError(
            "Incorrect API key provided.",
            status_code=401,
            error_type="invalid_request_error",
            code="invalid_api_key",
        )


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OpenAIAPIError("Invalid JSON body.") from exc
    if not isinstance(body, dict):
        raise OpenAIAPIError("Request body must be a JSON object.")
    return body


def _validate_model(model: Any, settings: ServerSettings) -> None:
    if model != settings.model_id:
        raise OpenAIAPIError(
            f"The model `{model}` does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )


def _prepare_chat_request(body: dict[str, Any], settings: ServerSettings) -> dict[str, Any]:
    _validate_model(body.get("model"), settings)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenAIAPIError("`messages` must be a non-empty array.", param="messages")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise OpenAIAPIError(
                f"Invalid message at index {index}.",
                param=f"messages.{index}",
            )

    tools = _normalize_tools(body.get("tools"))
    tool_choice = body.get("tool_choice")
    if tool_choice == "none":
        tools = []
    elif isinstance(tool_choice, dict):
        selected = ((tool_choice.get("function") or {}).get("name"))
        if not isinstance(selected, str) or selected not in _tool_names(tools):
            raise OpenAIAPIError("`tool_choice` references an unknown function.", param="tool_choice")
        tools = [tool for tool in tools if (tool.get("function") or {}).get("name") == selected]
    elif tool_choice not in (None, "auto", "required"):
        raise OpenAIAPIError("Unsupported `tool_choice` value.", param="tool_choice")
    if tool_choice == "required" and not tools:
        raise OpenAIAPIError("`tool_choice=required` requires at least one tool.", param="tool_choice")

    return {
        "prompt": decision_prompt(messages, tools, tool_choice) if tools else render_messages(messages),
        "instruction": None,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
    }


def _responses_to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("previous_response_id") or body.get("conversation"):
        raise OpenAIAPIError(
            "Send the complete input history; server-side conversation storage is not supported.",
            param="previous_response_id" if body.get("previous_response_id") else "conversation",
        )
    model = body.get("model")
    input_value = body.get("input")
    if isinstance(input_value, str):
        messages: list[dict[str, Any]] = [{"role": "user", "content": input_value}]
    elif isinstance(input_value, list) and input_value:
        messages = []
        for index, item in enumerate(input_value):
            if not isinstance(item, dict):
                raise OpenAIAPIError(f"Invalid input item at index {index}.", param=f"input.{index}")
            item_type = item.get("type")
            if item_type in (None, "message") and isinstance(item.get("role"), str):
                messages.append({"role": item["role"], "content": item.get("content")})
            elif item_type == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item.get("call_id") or item.get("id") or f"call_{index}",
                                "type": "function",
                                "function": {
                                    "name": item.get("name"),
                                    "arguments": item.get("arguments", "{}"),
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }
                )
            else:
                raise OpenAIAPIError(
                    f"Unsupported input item type `{item_type}`.",
                    param=f"input.{index}.type",
                )
    else:
        raise OpenAIAPIError("`input` must be a string or non-empty array.", param="input")

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    return {
        "model": model,
        "messages": messages,
        "tools": _responses_tools_to_chat_tools(body.get("tools")),
        "tool_choice": _responses_tool_choice(body.get("tool_choice")),
        "stream": body.get("stream", False),
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
    }


def _responses_tools_to_chat_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise OpenAIAPIError("`tools` must be an array.", param="tools")
    converted: list[dict[str, Any]] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise OpenAIAPIError(
                "Only function tools are currently supported.",
                param=f"tools.{index}",
            )
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("parameters") or {},
                    "strict": tool.get("strict"),
                },
            }
        )
    return converted


def _responses_tool_choice(value: Any) -> Any:
    if not isinstance(value, dict) or value.get("type") != "function":
        return value
    return {"type": "function", "function": {"name": value.get("name")}}


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise OpenAIAPIError("`tools` must be an array.", param="tools")
    result: list[dict[str, Any]] = []
    for index, tool in enumerate(value):
        function = tool.get("function") if isinstance(tool, dict) else None
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
        ):
            raise OpenAIAPIError(f"Invalid function tool at index {index}.", param=f"tools.{index}")
        try:
            validate_schema(function.get("parameters", {}))
        except Exception as exc:
            raise OpenAIAPIError(
                "Invalid function parameters schema (only local references are supported).",
                param=f"tools.{index}.function.parameters",
            ) from exc
        if function["name"] in _tool_names(result):
            raise OpenAIAPIError("Duplicate function name.", param=f"tools.{index}.function.name")
        result.append(tool)
    return result


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {(tool.get("function") or {}).get("name") for tool in tools}


def _message_from_raw(raw: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    if not tools:
        return {"role": "assistant", "content": _clean_upstream_text(raw)}
    # Remove only trailing metadata outside the decision. A function argument
    # may itself contain an HTML tools comment (e.g. when writing source files).
    decision = _strip_trailing_tool_metadata(raw)
    message = parse_decision(decision.strip(), tools)
    if isinstance(message.get("content"), str):
        message["content"] = _strip_trailing_tool_metadata(message["content"])
    return message


def _strip_trailing_tool_metadata(text: str) -> str:
    return re.sub(r"\s*<!--\s*tools\s*:(?:(?!-->)[\s\S])*-->\s*$", "", text, flags=re.IGNORECASE)


async def _complete_message(provider, semaphore, settings, prepared):
    """Retry invalid decisions, never synthesize actions from failed prose."""
    prompt = prepared["prompt"]
    attempts = settings.tool_attempts if prepared["tools"] else 1
    for attempt in range(attempts):
        async with semaphore:
            raw = await asyncio.to_thread(provider.complete, prompt, prepared["instruction"], None)
        try:
            message = _message_from_raw(raw, prepared["tools"])
            _enforce_tool_choice(message, prepared["tool_choice"])
            if prepared.get("parallel_tool_calls") is False and len(message.get("tool_calls", [])) > 1:
                raise ToolProtocolError("Choose exactly one operation this turn.")
            return raw, message
        except (ToolProtocolError, OpenAIAPIError) as exc:
            if attempt + 1 == attempts:
                raise OpenAIAPIError(
                    f"The upstream model did not return a valid decision after {attempts} attempts: {exc}",
                    status_code=502, error_type="server_error", code="tool_call_failed",
                ) from exc
            prompt = prepared["prompt"] + (
                "\nThe application could not validate the previous decision: " + str(exc)
                + "\nPrevious response (quoted data): " + json.dumps(raw[:8000], ensure_ascii=False)
                + '\nInclude a corrected JSON decision document, including "tools":[], in your answer. '
                "No operation has been executed by this request."
            )


def _enforce_tool_choice(message: dict[str, Any], tool_choice: Any) -> None:
    if tool_choice != "required" and not isinstance(tool_choice, dict):
        return
    if message.get("tool_calls"):
        return
    raise OpenAIAPIError(
        "The upstream model did not produce the required function call.",
        status_code=502,
        error_type="server_error",
        param="tool_choice",
        code="tool_call_failed",
    )


_TOOLS_COMMENT = re.compile(r"\s*<!--\s*tools\s*:[\s\S]*?-->\s*", re.IGNORECASE)
_TOOLS_COMMENT_START = re.compile(r"<!--\s*tools\s*:", re.IGNORECASE)


def _clean_upstream_text(text: Any) -> str:
    return _TOOLS_COMMENT.sub("", text if isinstance(text, str) else str(text or "")).rstrip()


class _StreamingTextCleaner:
    """Remove upstream-only markers even when split across SSE chunks."""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, chunk: str, *, final: bool = False) -> str:
        self.buffer += chunk
        self.buffer = _TOOLS_COMMENT.sub("", self.buffer)
        if final:
            output = self.buffer.rstrip()
            self.buffer = ""
            return output
        starts = list(_TOOLS_COMMENT_START.finditer(self.buffer))
        if starts:
            marker_start = starts[-1].start()
            output = self.buffer[:marker_start]
            self.buffer = self.buffer[marker_start:]
            return output
        # Retain enough text to recognize a marker prefix split across chunks.
        if len(self.buffer) <= 16:
            return ""
        output = self.buffer[:-16]
        self.buffer = self.buffer[-16:]
        return output


def _chat_completion_payload(
    *,
    model: str,
    message: dict[str, Any],
    prompt: str,
    raw: str,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason_for_message(message),
            }
        ],
        "usage": _usage(prompt, raw),
        "system_fingerprint": "kingogpt-gateway",
    }


async def _stream_chat_completion(
    provider: CompletionProvider,
    semaphore: asyncio.Semaphore,
    settings: ServerSettings,
    prepared: dict[str, Any],
    body: dict[str, Any],
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def event(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": settings.model_id,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
            "system_fingerprint": "kingogpt-gateway",
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield event({"role": "assistant", "content": ""})
    tools = prepared["tools"]

    try:
        if tools:
            raw, message = await _complete_message(provider, semaphore, settings, prepared)
            if message.get("tool_calls"):
                for index, call in enumerate(message["tool_calls"]):
                    yield event(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": call["function"],
                                }
                            ]
                        }
                    )
                yield event({}, "tool_calls")
            else:
                yield event({"content": message.get("content") or ""})
                yield event({}, "stop")
        else:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            cleaner = _StreamingTextCleaner()

            def on_chunk(chunk: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))

            async def run_provider() -> None:
                try:
                    async with semaphore:
                        raw_text = await asyncio.to_thread(
                            provider.complete,
                            prepared["prompt"],
                            prepared["instruction"],
                            on_chunk,
                        )
                    await queue.put(("done", raw_text))
                except Exception as exc:  # streamed failures must be encoded in-band
                    await queue.put(("error", exc))

            task = asyncio.create_task(run_provider())
            raw = ""
            while True:
                kind, value = await queue.get()
                if kind == "chunk":
                    cleaned = cleaner.feed(value)
                    if cleaned:
                        raw += cleaned
                        yield event({"content": cleaned})
                elif kind == "done":
                    cleaned = cleaner.feed("", final=True)
                    if cleaned:
                        raw += cleaned
                        yield event({"content": cleaned})
                    if not raw and value:
                        raw = _clean_upstream_text(value)
                        if raw:
                            yield event({"content": raw})
                    break
                else:
                    raise value
            await task
            yield event({}, "stop")

        if (body.get("stream_options") or {}).get("include_usage"):
            usage_payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.model_id,
                "choices": [],
                "usage": _usage(prepared["prompt"], raw),
            }
            yield f"data: {json.dumps(usage_payload, ensure_ascii=False)}\n\n"
    except Exception as exc:
        error = exc if isinstance(exc, OpenAIAPIError) else OpenAIAPIError(
            f"KingoGPT upstream request failed: {exc}",
            status_code=502,
            error_type="server_error",
            code="upstream_error",
        )
        yield f"data: {json.dumps(_error_payload(error), ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _response_payload(
    *,
    model: str,
    message: dict[str, Any],
    prompt: str,
    raw: str,
    request_body: dict[str, Any],
    response_id: str | None = None,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if message.get("tool_calls"):
        for call in message["tool_calls"]:
            output.append(
                {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
            )
    else:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": message.get("content") or "",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    usage = _usage(prompt, raw)
    return {
        "id": response_id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": request_body.get("instructions"),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "model": model,
        "output": output,
        "parallel_tool_calls": bool(request_body.get("parallel_tool_calls", True)),
        "previous_response_id": request_body.get("previous_response_id"),
        "reasoning": request_body.get("reasoning"),
        "store": False,
        "temperature": request_body.get("temperature"),
        "text": request_body.get("text") or {"format": {"type": "text"}},
        "tool_choice": request_body.get("tool_choice") or "auto",
        "tools": request_body.get("tools") or [],
        "top_p": request_body.get("top_p"),
        "truncation": request_body.get("truncation") or "disabled",
        "usage": {
            "input_tokens": usage["prompt_tokens"],
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage["completion_tokens"],
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage["total_tokens"],
        },
        "user": request_body.get("user"),
        "metadata": request_body.get("metadata") or {},
    }


async def _stream_response(
    provider: CompletionProvider,
    semaphore: asyncio.Semaphore,
    settings: ServerSettings,
    prepared: dict[str, Any],
    body: dict[str, Any],
):
    response_id = f"resp_{uuid.uuid4().hex}"
    sequence = 0

    def sse(event_type: str, payload: dict[str, Any]) -> str:
        nonlocal sequence
        event = {"type": event_type, "sequence_number": sequence, **payload}
        sequence += 1
        return f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    initial = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "model": settings.model_id,
        "output": [],
        "error": None,
    }
    yield sse("response.created", {"response": initial})
    tools = prepared["tools"]
    raw = ""

    try:
        if tools:
            raw, message = await _complete_message(provider, semaphore, settings, prepared)
            completed = _response_payload(
                model=settings.model_id,
                message=message,
                prompt=prepared["prompt"],
                raw=raw,
                request_body=body,
                response_id=response_id,
            )
            for output_index, item in enumerate(completed["output"]):
                yield sse(
                    "response.output_item.added",
                    {"output_index": output_index, "item": {
                        **item, "status": "in_progress",
                        **({"arguments": ""} if item["type"] == "function_call" else {"content": []}),
                    }},
                )
                if item["type"] == "function_call":
                    yield sse(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": item["id"],
                            "output_index": output_index,
                            "delta": item["arguments"],
                        },
                    )
                    yield sse(
                        "response.function_call_arguments.done",
                        {
                            "item_id": item["id"],
                            "output_index": output_index,
                            "arguments": item["arguments"],
                        },
                    )
                else:
                    part = item["content"][0]
                    base = {"item_id": item["id"], "output_index": output_index, "content_index": 0}
                    yield sse("response.content_part.added", {**base, "part": {**part, "text": ""}})
                    yield sse("response.output_text.delta", {**base, "delta": part["text"], "logprobs": []})
                    yield sse("response.output_text.done", {**base, "text": part["text"], "logprobs": []})
                    yield sse("response.content_part.done", {**base, "part": part})
                yield sse("response.output_item.done", {"output_index": output_index, "item": item})
        else:
            item_id = f"msg_{uuid.uuid4().hex}"
            in_progress_item = {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            yield sse("response.output_item.added", {"output_index": 0, "item": in_progress_item})
            part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
            yield sse(
                "response.content_part.added",
                {"item_id": item_id, "output_index": 0, "content_index": 0, "part": part},
            )
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            cleaner = _StreamingTextCleaner()

            def on_chunk(chunk: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))

            async def run_provider() -> None:
                try:
                    async with semaphore:
                        answer = await asyncio.to_thread(
                            provider.complete,
                            prepared["prompt"],
                            prepared["instruction"],
                            on_chunk,
                        )
                    await queue.put(("done", answer))
                except Exception as exc:
                    await queue.put(("error", exc))

            task = asyncio.create_task(run_provider())
            while True:
                kind, value = await queue.get()
                if kind == "chunk":
                    cleaned = cleaner.feed(value)
                    if cleaned:
                        raw += cleaned
                        yield sse(
                            "response.output_text.delta",
                            {
                                "item_id": item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": cleaned,
                                "logprobs": [],
                            },
                        )
                elif kind == "done":
                    cleaned = cleaner.feed("", final=True)
                    if cleaned:
                        raw += cleaned
                        yield sse(
                            "response.output_text.delta",
                            {
                                "item_id": item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": cleaned,
                                "logprobs": [],
                            },
                        )
                    if not raw and value:
                        raw = _clean_upstream_text(value)
                        if raw:
                            yield sse(
                                "response.output_text.delta",
                                {
                                    "item_id": item_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": raw,
                                    "logprobs": [],
                                },
                            )
                    break
                else:
                    raise value
            await task
            message = {"role": "assistant", "content": raw}
            completed = _response_payload(
                model=settings.model_id,
                message=message,
                prompt=prepared["prompt"],
                raw=raw,
                request_body=body,
                response_id=response_id,
            )
            final_item = completed["output"][0]
            final_item["id"] = item_id
            yield sse(
                "response.output_text.done",
                {
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": raw,
                    "logprobs": [],
                },
            )
            yield sse(
                "response.content_part.done",
                {
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": final_item["content"][0],
                },
            )
            yield sse("response.output_item.done", {"output_index": 0, "item": final_item})

        yield sse("response.completed", {"response": completed})
    except Exception as exc:
        error = {
            "code": exc.code if isinstance(exc, OpenAIAPIError) else "upstream_error",
            "message": f"KingoGPT upstream request failed: {exc}",
            "param": None,
            "type": "server_error",
        }
        yield sse("error", {"error": error})


def _usage(prompt: str, completion: str) -> dict[str, int]:
    # The upstream does not report tokenizer counts.  A deterministic estimate is
    # more useful to clients than omitting the standard field entirely.
    prompt_tokens = _estimated_tokens(prompt)
    completion_tokens = _estimated_tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _estimated_tokens(text: str) -> int:
    return 0 if not text else max(1, (len(text) + 3) // 4)


def _model_object(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "kingogpt",
    }


def _error_payload(exc: OpenAIAPIError) -> dict[str, Any]:
    return {
        "error": {
            "message": str(exc),
            "type": exc.error_type,
            "param": exc.param,
            "code": exc.code,
        }
    }


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "kingogpt.openai_server:app",
        host=os.getenv("KINGOGPT_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("KINGOGPT_SERVER_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
