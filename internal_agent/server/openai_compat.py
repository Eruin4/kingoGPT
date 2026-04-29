import os
import time
import json
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


app = FastAPI()


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

    model: str = "kingogpt-web"
    messages: list[dict[str, Any]]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "kingogpt-openai-compatible"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "kingogpt-web",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "kingogpt",
            }
        ],
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def messages_to_prompt_and_system(messages: list[dict[str, Any]]) -> tuple[str, str]:
    blocks: list[str] = []
    system_blocks: list[str] = []

    for message in messages:
        role = message.get("role", "user")
        content = _content_to_text(message.get("content"))
        if not content.strip():
            continue
        if role == "system":
            system_blocks.append(content)
        else:
            blocks.append(f"{role.upper()}:\n{content}")

    blocks.append("ASSISTANT:")
    return "\n\n".join(blocks), "\n\n".join(system_blocks)


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    prompt, system_prompt = messages_to_prompt_and_system(messages)
    if system_prompt:
        return f"SYSTEM:\n{system_prompt}\n\n{prompt}"
    return prompt


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
    }


def sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
def raw_chat_completions(req: ChatCompletionRequest):
    """
    Raw OpenAI-compatible model endpoint. Hermes should use this endpoint.
    """
    prompt, system_prompt = messages_to_prompt_and_system(req.messages)

    if req.stream:
        def generate():
            try:
                answer = _raw_llm.complete(prompt, system_prompt=system_prompt)

                yield sse_event(
                    {
                        "id": "chatcmpl-kingogpt",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )

                yield sse_event(
                    {
                        "id": "chatcmpl-kingogpt",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": answer,
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )

                yield sse_event(
                    {
                        "id": "chatcmpl-kingogpt",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )

                yield "data: [DONE]\n\n"
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

    return make_completion_response(req.model, answer)


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
