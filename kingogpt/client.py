from pathlib import Path

from internal_agent.config import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SESSION_KEY,
    DEFAULT_TOKEN_CACHE,
    DEFAULT_TOKEN_CONFIG,
    DEFAULT_TOKEN_REFRESH_TIMEOUT_SECONDS,
)
from internal_agent.llm.azure_web_adapter import AzureWebLLM


class KingoGPTClient:
    """Importable client wrapper for the KingoGPT web/API backend."""

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
    ) -> None:
        self._llm = AzureWebLLM(
            token_cache=token_cache,
            token_config=token_config,
            profile_dir=profile_dir,
            session_key=session_key,
            chat_room_id=chat_room_id,
            scenario_id=scenario_id,
            request_timeout=request_timeout,
            token_refresh_timeout=token_refresh_timeout,
            no_auto_refresh_token=no_auto_refresh_token,
            ignore_expiry=ignore_expiry,
            reuse_thread=reuse_thread,
            auto_delete_thread=auto_delete_thread,
            echo=echo,
        )

    def chat(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return self._llm.complete(prompt, system_prompt=system_prompt)
