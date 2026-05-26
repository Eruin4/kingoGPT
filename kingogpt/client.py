import argparse
from pathlib import Path

from kingogpt import api_solver


class KingoGPTClient:
    """Small importable wrapper around the KingoGPT API solver.

    Unlike the previous minimal implementation, this version has full parity
    with the CLI ``main()`` function:
    - Auto-refresh on auth failure during API calls (not just during initial load)
    - Session/prompt state management for room/thread reuse across calls
    - Stale thread recovery via ``should_reset_prompt_state``
    """

    def __init__(
        self,
        *,
        token_cache: str | Path = Path("state/kingogpt_token_cache.json"),
        token_config: str | Path = Path("state/kingogpt_config.json"),
        profile_dir: str | Path = Path("state/kingogpt_chrome_profile"),
        session_key: str = "kingogpt-client",
        chat_room_id: int | None = None,
        scenario_id: str | None = None,
        request_timeout: int = 120,
        token_refresh_timeout: int = 300,
        no_auto_refresh_token: bool = False,
        ignore_expiry: bool = False,
    ) -> None:
        self._args = argparse.Namespace(
            access_token=None,
            token_cache=str(token_cache),
            token_config=str(token_config),
            profile_dir=str(profile_dir),
            token_refresh_timeout=token_refresh_timeout,
            no_auto_refresh_token=no_auto_refresh_token,
            chat_room_id=chat_room_id,
            scenario_id=scenario_id or api_solver.DEFAULT_SCENARIO_ID,
            request_timeout=request_timeout,
            ignore_expiry=ignore_expiry,
            system_prompt="",
            system_prompt_file=None,
            session_key=session_key,
        )

    def chat(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Send *prompt* to KingoGPT and return the streamed answer text.

        Handles token refresh, session state reuse, and stale thread recovery
        identically to the CLI ``main()`` function.
        """
        dynamic_system_prompt = (system_prompt or "").strip()
        prompt_hash = api_solver.create_prompt_hash(dynamic_system_prompt)
        state_key = api_solver.build_state_key(self._args.session_key, prompt_hash)

        cache, token, claims, user = api_solver.load_or_refresh_token(self._args)
        if not user.get("id"):
            user["id"] = claims.get("userId")
            user["userId"] = claims.get("userId")

        existing_state = api_solver.read_session_prompt_state(cache, state_key)
        current_room_id = (
            self._args.chat_room_id
            or (existing_state or {}).get("chatRoomId")
            or cache.get("chat_room_id")
            or api_solver.DEFAULT_CHAT_ROOM_ID
        )
        current_thread_id = (existing_state or {}).get("chatThreadId")

        request_prompt = api_solver.build_request_prompt(
            prompt,
            None if existing_state else dynamic_system_prompt or None,
        )

        try:
            try:
                answer, resolved_room_id, resolved_thread_id = api_solver.chat_via_api(
                    token,
                    user,
                    request_prompt,
                    self._args,
                    instruction=None,
                    chat_room_id=current_room_id,
                    chat_thread_id=current_thread_id,
                )
            except Exception as exc:
                if self._args.no_auto_refresh_token or not api_solver.should_auto_refresh_token(exc):
                    raise
                cache = api_solver.refresh_token_cache(self._args)
                token = api_solver.resolve_access_token(self._args, cache)
                claims = api_solver.ensure_token_is_fresh(token, ignore_expiry=False)
                user = api_solver.fetch_user_profile(token)
                if not user.get("id"):
                    user["id"] = claims.get("userId")
                    user["userId"] = claims.get("userId")
                answer, resolved_room_id, resolved_thread_id = api_solver.chat_via_api(
                    token,
                    user,
                    request_prompt,
                    self._args,
                    instruction=None,
                    chat_room_id=current_room_id,
                    chat_thread_id=current_thread_id,
                )
        except Exception as exc:
            message = str(exc)
            if not existing_state or not api_solver.should_reset_prompt_state(message):
                raise

            api_solver.delete_session_prompt_state(cache, state_key)
            reset_request_prompt = api_solver.build_request_prompt(
                prompt,
                dynamic_system_prompt or None,
            )
            answer, resolved_room_id, resolved_thread_id = api_solver.chat_via_api(
                token,
                user,
                reset_request_prompt,
                self._args,
                instruction=None,
                chat_room_id=current_room_id,
            )

        api_solver.write_session_prompt_state(
            cache,
            state_key,
            prompt_hash=prompt_hash,
            chat_room_id=resolved_room_id,
            chat_thread_id=resolved_thread_id,
        )
        if "chat_room_id" not in cache and api_solver.DEFAULT_CHAT_ROOM_ID is not None:
            cache["chat_room_id"] = api_solver.DEFAULT_CHAT_ROOM_ID
        api_solver.write_token_cache(self._args.token_cache, cache)
        return answer
