import argparse
from pathlib import Path

from kingogpt import api_solver


class KingoGPTClient:
    """Small importable wrapper around the KingoGPT API solver."""

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
        cache, token, claims, user = api_solver.load_or_refresh_token(self._args)
        if not user.get("id"):
            user["id"] = claims.get("userId")
            user["userId"] = claims.get("userId")

        request_prompt = api_solver.build_request_prompt(prompt, system_prompt)
        answer, room_id, thread_id = api_solver.chat_via_api(
            token,
            user,
            request_prompt,
            self._args,
            instruction=None,
            chat_room_id=self._args.chat_room_id,
            chat_thread_id=None,
        )
        if "chat_room_id" not in cache and room_id is not None:
            cache["chat_room_id"] = room_id
        if thread_id is not None:
            cache["last_chat_thread_id"] = thread_id
        api_solver.write_token_cache(self._args.token_cache, cache)
        return answer
