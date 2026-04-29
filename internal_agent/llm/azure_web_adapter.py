import argparse
import contextlib
import io
import threading
from collections.abc import Iterator
from pathlib import Path

import kingogpt_api_solver as kingogpt

from internal_agent.config import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SESSION_KEY,
    DEFAULT_TOKEN_CACHE,
    DEFAULT_TOKEN_CONFIG,
    DEFAULT_TOKEN_REFRESH_TIMEOUT_SECONDS,
)


class AzureWebLLM:
    """LLM adapter backed by the existing KingoGPT API/SSE automation code."""

    def __init__(
        self,
        *,
        token_cache: str | Path = DEFAULT_TOKEN_CACHE,
        token_config: str | Path = DEFAULT_TOKEN_CONFIG,
        profile_dir: str | Path = DEFAULT_PROFILE_DIR,
        session_key: str = DEFAULT_SESSION_KEY,
        chat_room_id: int | None = None,
        scenario_id: str | None = None,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        token_refresh_timeout: int = DEFAULT_TOKEN_REFRESH_TIMEOUT_SECONDS,
        no_auto_refresh_token: bool = False,
        ignore_expiry: bool = False,
        reuse_thread: bool = False,
        auto_delete_thread: bool = True,
        echo: bool = False,
    ):
        self.session_key = session_key
        self.reuse_thread = reuse_thread
        self.auto_delete_thread = auto_delete_thread
        self.echo = echo
        self._lock = threading.Lock()
        self._args = argparse.Namespace(
            access_token=None,
            token_cache=str(token_cache),
            token_config=str(token_config),
            profile_dir=str(profile_dir),
            token_refresh_timeout=token_refresh_timeout,
            no_auto_refresh_token=no_auto_refresh_token,
            chat_room_id=chat_room_id,
            scenario_id=scenario_id or kingogpt.DEFAULT_SCENARIO_ID,
            request_timeout=request_timeout,
            ignore_expiry=ignore_expiry,
            system_prompt="",
            system_prompt_file=None,
            session_key=session_key,
        )

    def _state_key(self, system_prompt: str | None = None) -> str:
        prompt_hash = kingogpt.create_prompt_hash(system_prompt or "internal-agent-adapter-v1")
        return kingogpt.build_state_key(self.session_key, prompt_hash)

    def _prompt_with_system(self, prompt: str, system_prompt: str | None) -> str:
        if not system_prompt or not system_prompt.strip():
            return prompt
        return f"SYSTEM:\n{system_prompt.strip()}\n\n{prompt}"

    def _chat_via_api(
        self,
        token: str,
        user: dict,
        prompt: str,
        *,
        system_prompt: str | None,
        chat_room_id: int | None,
        chat_thread_id: int | None,
    ) -> tuple[str, int | None, int | None]:
        api_prompt = self._prompt_with_system(prompt, system_prompt)
        if self.echo:
            return kingogpt.chat_via_api(
                token,
                user,
                api_prompt,
                self._args,
                instruction=None,
                chat_room_id=chat_room_id,
                chat_thread_id=chat_thread_id,
            )

        with contextlib.redirect_stdout(io.StringIO()):
            return kingogpt.chat_via_api(
                token,
                user,
                api_prompt,
                self._args,
                instruction=None,
                chat_room_id=chat_room_id,
                chat_thread_id=chat_thread_id,
            )

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Send prompt to KingoGPT and return the full streamed answer."""
        with self._lock:
            return self._complete_locked(prompt, system_prompt=system_prompt)

    def _complete_locked(self, prompt: str, *, system_prompt: str | None = None) -> str:
        state_key = self._state_key(system_prompt)
        cache, token, claims, user = kingogpt.load_or_refresh_token(self._args)
        if not user.get("id"):
            user["id"] = claims.get("userId")
            user["userId"] = claims.get("userId")

        existing_state = kingogpt.read_session_prompt_state(cache, state_key)
        current_room_id = (
            self._args.chat_room_id
            or (existing_state or {}).get("chatRoomId")
            or cache.get("chat_room_id")
            or kingogpt.DEFAULT_CHAT_ROOM_ID
        )
        current_thread_id = (
            (existing_state or {}).get("chatThreadId") if self.reuse_thread else None
        )
        system_prompt_for_request = (
            None if self.reuse_thread and existing_state else system_prompt
        )

        try:
            try:
                text, resolved_room_id, resolved_thread_id = self._chat_via_api(
                    token,
                    user,
                    prompt,
                    system_prompt=system_prompt_for_request,
                    chat_room_id=current_room_id,
                    chat_thread_id=current_thread_id,
                )
            except Exception as exc:
                if self._args.no_auto_refresh_token or not kingogpt.should_auto_refresh_token(exc):
                    raise
                cache = kingogpt.refresh_token_cache(self._args)
                token = kingogpt.resolve_access_token(self._args, cache)
                claims = kingogpt.ensure_token_is_fresh(token, ignore_expiry=False)
                user = kingogpt.fetch_user_profile(token)
                if not user.get("id"):
                    user["id"] = claims.get("userId")
                    user["userId"] = claims.get("userId")
                text, resolved_room_id, resolved_thread_id = self._chat_via_api(
                    token,
                    user,
                    prompt,
                    system_prompt=system_prompt_for_request,
                    chat_room_id=current_room_id,
                    chat_thread_id=current_thread_id,
                )
        except Exception as exc:
            if not existing_state or not kingogpt.should_reset_prompt_state(str(exc)):
                raise

            kingogpt.delete_session_prompt_state(cache, state_key)
            text, resolved_room_id, resolved_thread_id = self._chat_via_api(
                token,
                user,
                prompt,
                system_prompt=system_prompt,
                chat_room_id=current_room_id,
                chat_thread_id=None,
            )

        self._delete_thread_if_needed(token, resolved_thread_id)

        kingogpt.write_session_prompt_state(
            cache,
            state_key,
            prompt_hash=kingogpt.create_prompt_hash(system_prompt or "internal-agent-adapter-v1"),
            chat_room_id=resolved_room_id,
            chat_thread_id=resolved_thread_id if self.reuse_thread else None,
        )
        if "chat_room_id" not in cache and kingogpt.DEFAULT_CHAT_ROOM_ID is not None:
            cache["chat_room_id"] = kingogpt.DEFAULT_CHAT_ROOM_ID
        kingogpt.write_token_cache(self._args.token_cache, cache)
        return text

    def stream(self, prompt: str) -> Iterator[str]:
        """MVP stream compatibility: yield the completed response once."""
        yield self.complete(prompt)

    def _delete_thread_if_needed(self, token: str, thread_id: int | None) -> None:
        if not self.auto_delete_thread or self.reuse_thread or thread_id is None:
            return

        try:
            kingogpt.delete_chat_thread(token, thread_id)
        except Exception as exc:
            if self.echo:
                print(f"[!] Failed to delete KingoGPT thread {thread_id}: {exc}")
