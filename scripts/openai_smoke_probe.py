"""End-to-end smoke probe for a running KingoGPT OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def request_json(base_url: str, path: str, api_key: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base_url + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def stream_chat(base_url: str, api_key: str, model: str) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "한 단어로 인사해줘"}],
        "stream": True,
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    chunks: list[str] = []
    saw_done = False
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                saw_done = True
                break
            payload = json.loads(data)
            if payload.get("error"):
                raise RuntimeError(payload["error"].get("message", "stream failed"))
            choices = payload.get("choices") or []
            if choices:
                chunks.append((choices[0].get("delta") or {}).get("content") or "")
    if not saw_done:
        raise RuntimeError("stream ended without [DONE]")
    return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.getenv("KINGOGPT_SERVER_API_KEY"))
    parser.add_argument("--check-tools", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or KINGOGPT_SERVER_API_KEY is required")

    try:
        health = request_json(args.base_url, "/health", args.api_key)
        model = health["model"]
        models = request_json(args.base_url, "/v1/models", args.api_key)
        if not any(item.get("id") == model for item in models.get("data", [])):
            raise RuntimeError("configured model is missing from /v1/models")

        chat = request_json(
            args.base_url,
            "/v1/chat/completions",
            args.api_key,
            {
                "model": model,
                "messages": [{"role": "user", "content": "한 문장으로 인사해줘"}],
            },
        )
        chat_text = chat["choices"][0]["message"].get("content") or ""
        if not chat_text.strip():
            raise RuntimeError("Chat Completions returned empty content")

        streamed_text = stream_chat(args.base_url, args.api_key, model)
        if not streamed_text.strip():
            raise RuntimeError("streaming Chat Completions returned empty content")

        response = request_json(
            args.base_url,
            "/v1/responses",
            args.api_key,
            {"model": model, "input": "한 문장으로 자신을 소개해줘"},
        )
        output_text = "".join(
            part.get("text", "")
            for item in response.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        if not output_text.strip():
            raise RuntimeError("Responses API returned empty output_text")

        for label, text in (("chat", chat_text), ("stream", streamed_text), ("response", output_text)):
            if "<!-- tools:" in text.lower():
                raise RuntimeError(f"{label} leaked an upstream tools marker: {text[-240:]!r}")

        tool_name = None
        if args.check_tools:
            tool_response = request_json(
                args.base_url,
                "/v1/chat/completions",
                args.api_key,
                {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Call get_weather for Seoul. Do not answer directly.",
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get the current weather for a city.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                    "tool_choice": "required",
                },
            )
            tool_choice = tool_response["choices"][0]
            calls = tool_choice["message"].get("tool_calls") or []
            if tool_choice.get("finish_reason") != "tool_calls" or not calls:
                raise RuntimeError("required function tool did not return tool_calls")
            tool_name = calls[0]["function"]["name"]
            if tool_name != "get_weather":
                raise RuntimeError(f"unexpected function tool `{tool_name}`")
    except (KeyError, IndexError, RuntimeError, urllib.error.URLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    passed = "health, models, chat, chat stream, responses"
    if args.check_tools:
        passed += ", function tools"
    print(f"PASS: {passed}")
    print(f"chat={chat_text[:120]!r}")
    print(f"stream={streamed_text[:120]!r}")
    print(f"response={output_text[:120]!r}")
    if args.check_tools:
        print(f"tool={tool_name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
