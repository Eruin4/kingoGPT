import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from internal_agent.agent.loop import Agent
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


def create_agent_from_env() -> Agent:
    llm = AzureWebLLM(
        token_cache=os.getenv("KINGOGPT_TOKEN_CACHE", str(DEFAULT_TOKEN_CACHE)),
        token_config=os.getenv("KINGOGPT_TOKEN_CONFIG", str(DEFAULT_TOKEN_CONFIG)),
        profile_dir=os.getenv("KINGOGPT_PROFILE_DIR", str(DEFAULT_PROFILE_DIR)),
        session_key=os.getenv("INTERNAL_AGENT_SESSION_KEY", "internal_agent_api"),
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
    return Agent(llm, max_steps=_env_int("INTERNAL_AGENT_MAX_STEPS", DEFAULT_MAX_STEPS))


_agent = create_agent_from_env()


class ChatCompletionRequest(BaseModel):
    model: str = "internal-azure-web-agent"
    messages: list[dict[str, Any]]
    stream: bool = False


def get_agent() -> Agent:
    return _agent


@app.get("/health")
def health():
    return {"status": "ok", "service": "internal-azure-web-agent"}


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


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    user_messages = [
        _content_to_text(message.get("content"))
        for message in req.messages
        if message.get("role") == "user"
    ]
    task = user_messages[-1] if user_messages else ""
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

    return {
        "id": "chatcmpl-internal-agent",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
    }
