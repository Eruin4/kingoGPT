import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

QUERY_URL = "https://kingogpt.skku.edu/v2/athena/chats/m1/queries"
IDENTIX_ME_URL = "https://kingogpt.skku.edu/v2/identix/users/me"
CHAT_THREAD_URL = "https://kingogpt.skku.edu/v2/datahub/chats/threads/{thread_id}"
DEFAULT_SCENARIO_ID = "robi-gpt-dev:workflow_c0hfnXS236g4FKO"
DEFAULT_CHAT_ROOM_ID = 14
DEFAULT_TOKEN_CACHE = Path(__file__).with_name("kingogpt_token_cache.json")
TOKEN_EXPIRY_SKEW_SECONDS = 300
TEXT_ACTION_CONTRACT = "\n".join(
    [
        "## KingoClaw JSON Action Contract",
        "Every reply must be a single JSON object.",
        "When a listed server action is immediately required, do not say that you lack access.",
        'For a tool request, reply with: {"call":"<exact tool name>","args":{...}}',
        'If no server action is required, reply with: {"reply":"<your answer as a plain string>"}',
        "Do not put any prose before or after the JSON object.",
        'Example final action: {"reply":"I checked the workspace and found the requested summary."}',
        'Example read action: {"call":"read","args":{"path":"/home/eruin/.openclaw/workspace/HEARTBEAT.md"}}',
        'Example memory action: {"call":"memory_search","args":{"query":"recent decisions about KingoGPT"}}',
        'Example exec action: {"call":"exec","args":{"command":"pwd"}}',
        'Example message action: {"call":"message","args":{"action":"send","message":"Done.","channel":"discord"}}',
    ]
)
USER_TOOL_REMINDER = "\n".join(
    [
        'Reply with {"call":"<exact tool name>","args":{...}} when a listed tool or command is needed.',
        'Reply with {"reply":"<your answer as a plain string>"} when no tool is needed.',
        "Do not say you cannot access the file, command, or workspace when a listed tool can do it.",
    ]
)


def configure_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send prompts to KingoGPT using an access token and persistent room/thread state."
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text to send to KingoGPT.")
    parser.add_argument(
        "--access-token",
        dest="access_token",
        default=os.getenv("KINGOGPT_ACCESS_TOKEN"),
        help="KingoGPT accessToken value from browser localStorage.",
    )
    parser.add_argument(
        "--token-cache",
        default=str(DEFAULT_TOKEN_CACHE),
        help="Path to the token cache JSON written by kingogpt_token_capture.py.",
    )
    parser.add_argument(
        "--token-config",
        default=str(Path(__file__).with_name("kingogpt_config.json")),
        help="JSON config file with KingoGPT login credentials for automatic token refresh.",
    )
    parser.add_argument(
        "--profile-dir",
        default=str(Path(__file__).with_name("kingogpt_chrome_profile")),
        help="Persistent Playwright profile directory used for automatic token refresh.",
    )
    parser.add_argument(
        "--token-refresh-timeout",
        type=int,
        default=300,
        help="Seconds to wait for automatic KingoGPT login/token capture.",
    )
    parser.add_argument(
        "--no-auto-refresh-token",
        action="store_true",
        help="Do not run kingogpt_token_capture.py when the token cache is missing or stale.",
    )
    parser.add_argument(
        "--chat-room-id",
        type=int,
        default=None,
        help="Force a specific chat room ID instead of prompt-hash-based routing.",
    )
    parser.add_argument(
        "--scenario-id",
        default=DEFAULT_SCENARIO_ID,
        help="Scenario ID used by the Athena API.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Seconds to wait for the API stream after the connection is established.",
    )
    parser.add_argument(
        "--ignore-expiry",
        action="store_true",
        help="Skip the local access token expiry check.",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Dynamic system prompt text used only when bootstrapping a new KingoGPT room.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Read the dynamic system prompt from a UTF-8 text file.",
    )
    parser.add_argument(
        "--session-key",
        default="default",
        help="Logical session key used to scope prompt-hash room/thread reuse.",
    )
    return parser.parse_args()


def decode_jwt_payload(token: str) -> dict:
    try:
        payload_segment = token.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        return json.loads(decoded)
    except Exception as exc:
        raise RuntimeError("Failed to decode access token JWT.") from exc


def load_token_cache(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse token cache file: {path}") from exc


def write_token_cache(path_str: str, cache: dict) -> None:
    path = Path(path_str)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_access_token(args: argparse.Namespace, cache: dict) -> str:
    token = (args.access_token or cache.get("access_token") or "").strip()
    if not token:
        cache_path = Path(args.token_cache)
        raise RuntimeError(
            "access token is missing. "
            f"Create `{cache_path.name}` first or pass --access-token."
        )
    return token


def ensure_token_is_fresh(token: str, *, ignore_expiry: bool) -> dict:
    claims = decode_jwt_payload(token)
    exp = claims.get("exp")
    if ignore_expiry or not exp:
        return claims

    now = int(time.time())
    if exp <= now + TOKEN_EXPIRY_SKEW_SECONDS:
        expires_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp))
        raise RuntimeError(
            "access token is expired or about to expire "
            f"(exp={expires_at}). Refresh it with kingogpt_token_capture.py."
        )

    return claims


def should_auto_refresh_token(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        needle in message
        for needle in (
            "access token is missing",
            "expired or about to expire",
            "http 401",
            "http 403",
            "returned html instead of json",
            "session expired",
            "auth failed",
        )
    )


def refresh_token_cache(args: argparse.Namespace) -> dict:
    if args.no_auto_refresh_token:
        raise RuntimeError("Automatic token refresh is disabled.")

    print("[*] Refreshing KingoGPT login token...")
    try:
        import kingogpt_token_capture
    except ImportError as exc:
        raise RuntimeError(
            "Automatic token refresh requires kingogpt_token_capture.py and Playwright."
        ) from exc

    capture_args = argparse.Namespace(
        cache_file=args.token_cache,
        config_file=args.token_config,
        profile_dir=args.profile_dir,
        timeout=args.token_refresh_timeout,
        login_id=None,
        password=None,
    )
    result = asyncio.run(kingogpt_token_capture.refresh_token_cache(capture_args))
    if not result.get("access_token"):
        raise RuntimeError("Token refresh completed without an access token.")
    return result


def load_or_refresh_token(args: argparse.Namespace) -> tuple[dict, str, dict, dict]:
    try:
        cache = load_token_cache(args.token_cache)
        token = resolve_access_token(args, cache)
        claims = ensure_token_is_fresh(token, ignore_expiry=args.ignore_expiry)
        user = fetch_user_profile(token)
        return cache, token, claims, user
    except Exception as exc:
        if args.ignore_expiry or args.no_auto_refresh_token or not should_auto_refresh_token(exc):
            raise

        cache = refresh_token_cache(args)
        token = resolve_access_token(args, cache)
        claims = ensure_token_is_fresh(token, ignore_expiry=False)
        user = fetch_user_profile(token)
        return cache, token, claims, user


def fetch_user_profile(token: str) -> dict:
    response = requests.get(
        IDENTIX_ME_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=20,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        snippet = response.text[:500]
        raise RuntimeError(
            f"User profile lookup failed: HTTP {response.status_code} {snippet}"
        ) from exc

    payload = response.json()
    documents = ((payload.get("data") or {}).get("documents")) or []
    if not documents:
        raise RuntimeError("User profile response did not include any documents.")

    document = documents[0]
    groups = document.get("groups") or []
    primary_group_name = groups[0]["name"] if groups else None

    return {
        "id": document.get("authUsersId"),
        "loginId": document.get("username"),
        "name": document.get("name"),
        "email": document.get("email"),
        "groupName": primary_group_name,
        "userId": document.get("authUsersId"),
        "status": document.get("status"),
    }


def resolve_dynamic_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt_file:
        return Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    return (args.system_prompt or "").strip()


def create_prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def build_bootstrap_system_prompt(system_prompt: str) -> str:
    trimmed = system_prompt.strip()
    return "\n\n".join(part for part in (TEXT_ACTION_CONTRACT, trimmed) if part)


def build_state_key(session_key: str, prompt_hash: str) -> str:
    return f"{session_key}::{prompt_hash}"


def read_session_prompt_state(cache: dict, state_key: str) -> dict | None:
    prompt_state = cache.get("session_prompt_state") or {}
    entry = prompt_state.get(state_key)
    return entry if isinstance(entry, dict) else None


def write_session_prompt_state(
    cache: dict,
    state_key: str,
    *,
    prompt_hash: str,
    chat_room_id: int | None,
    chat_thread_id: int | None,
) -> None:
    cache.setdefault("session_prompt_state", {})
    cache["session_prompt_state"][state_key] = {
        "promptHash": prompt_hash,
        "chatRoomId": chat_room_id,
        "chatThreadId": chat_thread_id,
        "lastUsedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if chat_room_id is not None:
        cache["last_chat_room_id"] = int(chat_room_id)
    if chat_thread_id is not None:
        cache["last_chat_threads_id"] = int(chat_thread_id)


def delete_session_prompt_state(cache: dict, state_key: str) -> None:
    prompt_state = cache.get("session_prompt_state") or {}
    if state_key in prompt_state:
        del prompt_state[state_key]
        cache["session_prompt_state"] = prompt_state


def build_request_prompt(prompt: str, system_prompt: str | None) -> str:
    blocks: list[str] = []
    if system_prompt is not None:
        blocks.append(f"SYSTEM\n{build_bootstrap_system_prompt(system_prompt)}")
    blocks.append(f"USER\n{USER_TOOL_REMINDER}\n\n{prompt.strip()}")
    return "\n\n".join(blocks).strip()


def build_payload(
    user: dict,
    prompt: str,
    args: argparse.Namespace,
    *,
    chat_room_id: int | None = None,
    chat_thread_id: int | None = None,
) -> dict:
    param_filters = {"dataGb": "C"}
    if user.get("loginId"):
        param_filters["loginId"] = user["loginId"]
    if user.get("name"):
        param_filters["name"] = user["name"]
    if user.get("email"):
        param_filters["email"] = user["email"]
    if user.get("groupName"):
        param_filters["group"] = user["groupName"]
    if user.get("userId"):
        param_filters["userId"] = user["userId"]
    if user.get("status"):
        param_filters["status"] = user["status"]

    payload = {
        "app_type": "browser",
        "device_type": "pc",
        "users_id": user["id"],
        "scenarios_id": args.scenario_id,
        "llms": {
            "model_config": {
                "provider": "azure",
                "model_name": "gpt-5.2",
                "deployment_name": "gpt-5.2-gs-2025-12-11",
            },
            "reply_style_prompt": "normal",
        },
        "queries": {"type": "text", "text": prompt},
        "param_filters": param_filters,
        "sse_status_enabled": True,
    }
    if chat_room_id is not None:
        payload["chat_rooms_id"] = int(chat_room_id)
    if chat_thread_id is not None:
        payload["chat_threads_id"] = int(chat_thread_id)
    return payload


def parse_optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def extract_identifiers(event: dict) -> tuple[int | None, int | None]:
    data = event.get("data") or {}
    documents = data.get("documents") or []
    document = documents[0] if documents else {}

    room_id = (
        parse_optional_int(document.get("chat_rooms_id"))
        or parse_optional_int(document.get("chatRoomsId"))
        or parse_optional_int(data.get("chat_rooms_id"))
        or parse_optional_int(data.get("chatRoomsId"))
        or parse_optional_int(event.get("chat_rooms_id"))
        or parse_optional_int(event.get("chatRoomsId"))
    )

    thread_id = (
        parse_optional_int(document.get("chat_threads_id"))
        or parse_optional_int(document.get("chatThreadsId"))
        or parse_optional_int(data.get("chat_threads_id"))
        or parse_optional_int(data.get("chatThreadsId"))
        or parse_optional_int(event.get("chat_threads_id"))
        or parse_optional_int(event.get("chatThreadsId"))
    )

    return room_id, thread_id


def extract_stream_text(event: dict) -> str:
    data = event.get("data") or {}
    documents = data.get("documents") or []
    if documents:
        document = documents[0] or {}
        replies = document.get("replies") or {}
        text = replies.get("text")
        if isinstance(text, str):
            return text

    choices = event.get("choices") or []
    if choices:
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content

        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content

    content = event.get("content")
    if isinstance(content, str):
        return content

    text = event.get("text")
    if isinstance(text, str):
        return text

    return ""


def should_reset_prompt_state(message: str) -> bool:
    lowered = message.lower()
    if "http 4" not in lowered and "http 5" not in lowered:
        return False
    return any(
        needle in lowered
        for needle in (
            "chat_threads_id",
            "chat_rooms_id",
            "chat thread",
            "chat room",
            "thread id",
            "room id",
        )
    )


def delete_chat_thread(token: str, thread_id: int) -> None:
    response = requests.delete(
        CHAT_THREAD_URL.format(thread_id=int(thread_id)),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=20,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        snippet = response.text[:500]
        raise RuntimeError(
            f"Chat thread delete failed: HTTP {response.status_code} {snippet}"
        ) from exc


def chat_via_api(
    token: str,
    user: dict,
    prompt: str,
    args: argparse.Namespace,
    *,
    chat_room_id: int | None = None,
    chat_thread_id: int | None = None,
) -> tuple[str, int | None, int | None]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    payload = build_payload(
        user,
        prompt,
        args,
        chat_room_id=chat_room_id,
        chat_thread_id=chat_thread_id,
    )

    print("[*] API response started:")
    print("-" * 40)
    response = requests.post(
        QUERY_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=(10, args.request_timeout),
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        snippet = response.text[:500]
        raise RuntimeError(f"API request failed: HTTP {response.status_code} {snippet}") from exc

    full_text = ""
    resolved_room_id = chat_room_id
    resolved_thread_id = chat_thread_id
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue

        data_str = line[5:].strip()
        if not data_str:
            continue
        if data_str == "[DONE]":
            break

        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_room_id, event_thread_id = extract_identifiers(event)
        resolved_room_id = resolved_room_id or event_room_id
        resolved_thread_id = resolved_thread_id or event_thread_id

        chunk = extract_stream_text(event)
        if chunk:
            print(chunk, end="", flush=True)
            full_text += chunk

    print("\n" + "-" * 40)
    return full_text, resolved_room_id, resolved_thread_id


def main() -> int:
    configure_output()
    args = parse_args()
    user_prompt = " ".join(args.prompt).strip() or input("Prompt: ").strip()
    dynamic_system_prompt = resolve_dynamic_system_prompt(args)
    prompt_hash = create_prompt_hash(build_bootstrap_system_prompt(dynamic_system_prompt))
    state_key = build_state_key(args.session_key, prompt_hash)

    try:
        cache, token, claims, user = load_or_refresh_token(args)
        if not user.get("id"):
            user["id"] = claims.get("userId")
            user["userId"] = claims.get("userId")

        existing_state = read_session_prompt_state(cache, state_key)
        forced_room_id = args.chat_room_id
        default_room_id = cache.get("chat_room_id") or DEFAULT_CHAT_ROOM_ID
        current_room_id = (
            forced_room_id
            or (existing_state or {}).get("chatRoomId")
            or default_room_id
        )
        current_thread_id = (existing_state or {}).get("chatThreadId")

        request_prompt = build_request_prompt(
            user_prompt,
            None if existing_state else dynamic_system_prompt,
        )

        try:
            try:
                _, resolved_room_id, resolved_thread_id = chat_via_api(
                    token,
                    user,
                    request_prompt,
                    args,
                    chat_room_id=current_room_id,
                    chat_thread_id=current_thread_id,
                )
            except Exception as exc:
                if args.no_auto_refresh_token or not should_auto_refresh_token(exc):
                    raise
                cache = refresh_token_cache(args)
                token = resolve_access_token(args, cache)
                claims = ensure_token_is_fresh(token, ignore_expiry=False)
                user = fetch_user_profile(token)
                if not user.get("id"):
                    user["id"] = claims.get("userId")
                    user["userId"] = claims.get("userId")
                _, resolved_room_id, resolved_thread_id = chat_via_api(
                    token,
                    user,
                    request_prompt,
                    args,
                    chat_room_id=current_room_id,
                    chat_thread_id=current_thread_id,
                )
        except Exception as exc:
            message = str(exc)
            if not existing_state or not should_reset_prompt_state(message):
                raise

            delete_session_prompt_state(cache, state_key)
            reset_request_prompt = build_request_prompt(
                user_prompt,
                dynamic_system_prompt,
            )
            _, resolved_room_id, resolved_thread_id = chat_via_api(
                token,
                user,
                reset_request_prompt,
                args,
                chat_room_id=current_room_id,
            )

        write_session_prompt_state(
            cache,
            state_key,
            prompt_hash=prompt_hash,
            chat_room_id=resolved_room_id,
            chat_thread_id=resolved_thread_id,
        )
        if "chat_room_id" not in cache and DEFAULT_CHAT_ROOM_ID is not None:
            cache["chat_room_id"] = DEFAULT_CHAT_ROOM_ID
        write_token_cache(args.token_cache, cache)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"[!] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
