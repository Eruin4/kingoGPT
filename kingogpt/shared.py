"""Shared utilities used by both api_solver and token_capture.

This module is the single source of truth for functions that were previously
duplicated across the two scripts.  Both modules re-export the symbols they
used to define locally, so downstream callers are unaffected.
"""

import base64
import json
import os
import tempfile
import time
import sys
from pathlib import Path

import requests

from kingogpt.exceptions import (
    AuthenticationError,
    TokenCacheCorruptError,
    UserProfileError,
)
IDENTIX_ME_URL = "https://kingogpt.skku.edu/v2/identix/users/me"


def configure_output() -> None:
    """Ensure stdout/stderr use UTF-8 with lossy replacement."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def decode_jwt_payload(token: str) -> dict:
    """Decode and return the payload of a JWT without verification."""
    try:
        payload_segment = token.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        return json.loads(decoded)
    except Exception as exc:
        raise RuntimeError("Failed to decode access token JWT.") from exc  # noqa: keep RuntimeError for back-compat


def fetch_user_profile(token: str) -> dict:
    """Fetch the authenticated user profile from the KingoGPT identity API."""
    response = requests.get(
        IDENTIX_ME_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=20,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        snippet = response.text[:500]
        status = response.status_code
        if status in (401, 403):
            raise AuthenticationError(
                f"User profile lookup failed: HTTP {status} {snippet}"
            ) from exc
        raise UserProfileError(
            f"User profile lookup failed: HTTP {status} {snippet}"
        ) from exc

    payload = response.json()
    documents = ((payload.get("data") or {}).get("documents")) or []
    if not documents:
        raise UserProfileError("User profile response did not include any documents.")

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


def load_token_cache(path_str: str) -> dict:
    """Load a token cache JSON file, returning an empty dict if absent."""
    path = Path(path_str)
    if not path.exists():
        return {}

    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            raise TokenCacheCorruptError("Token cache must be a JSON object.")
        return cache
    except json.JSONDecodeError as exc:
        raise TokenCacheCorruptError(f"Failed to parse token cache file: {path}") from exc


def write_token_cache(path_str: str, cache: dict) -> None:
    """Atomically write *cache* to the token cache JSON file."""
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".token-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(cache, stream, ensure_ascii=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def refresh_cached_tokens(cache: dict) -> dict:
    """Use the same refresh endpoint as the web app; never log token values."""
    with requests.post(
        "https://kingogpt.skku.edu/identity/token/refresh",
        json={"accessToken": cache["access_token"], "refreshToken": cache["refresh_token"]},
        timeout=(10, 20), allow_redirects=False,
    ) as response:
        if response.status_code != 200:
            raise AuthenticationError(f"Token refresh failed: HTTP {response.status_code}")
        payload = response.json()
    if not isinstance(payload, dict) or not all(isinstance(payload.get(k), str) and payload[k]
                                               for k in ("accessToken", "refreshToken")):
        raise AuthenticationError("Token refresh response is missing tokens.")
    claims = decode_jwt_payload(payload["accessToken"])
    if claims.get("exp", 0) <= time.time() + 300:
        raise AuthenticationError("Token refresh returned an expired token.")
    return {**cache, "access_token": payload["accessToken"], "refresh_token": payload["refreshToken"],
            "claims": claims, "expires_at": claims["exp"], "fetched_at": int(time.time())}
