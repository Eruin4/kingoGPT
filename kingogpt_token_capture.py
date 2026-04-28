import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from playwright.async_api import async_playwright

START_URL = "https://www.skku.edu/skku/kingoGPT.do"
IDENTIX_ME_URL = "https://kingogpt.skku.edu/v2/identix/users/me"
DEFAULT_TOKEN_CACHE = Path(__file__).with_name("kingogpt_token_cache.json")
DEFAULT_CONFIG_FILE = Path(__file__).with_name("kingogpt_config.json")
DEFAULT_PROFILE_DIR = Path(__file__).with_name("kingogpt_chrome_profile")
DEFAULT_TIMEOUT_SECONDS = 300
AUTO_LOGIN_WAIT_SECONDS = 60


def configure_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log in to SKKU KingoGPT with Playwright and write a reusable token cache."
    )
    parser.add_argument("--cache-file", default=str(DEFAULT_TOKEN_CACHE))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--port", type=int, default=9222, help=argparse.SUPPRESS)
    parser.add_argument("--chrome-path", default=os.getenv("KINGOGPT_CHROME_PATH"), help=argparse.SUPPRESS)
    parser.add_argument("--keep-open", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--id", dest="login_id", default=None)
    parser.add_argument("--password", dest="password", default=None)
    return parser.parse_args()


def decode_jwt_payload(token: str) -> dict:
    try:
        payload_segment = token.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        return json.loads(decoded)
    except Exception as exc:
        raise RuntimeError("Failed to decode access token JWT.") from exc


def load_json_file(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON file: {path}") from exc


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    config = load_json_file(args.config_file)
    login_id = (args.login_id or os.getenv("KINGOGPT_ID") or config.get("id") or "").strip()
    password = args.password or os.getenv("KINGOGPT_PW") or config.get("password") or ""
    return login_id, password


def fetch_user_profile(token: str) -> dict:
    response = requests.get(
        IDENTIX_ME_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    documents = ((payload.get("data") or {}).get("documents")) or []
    if not documents:
        raise RuntimeError("User profile response did not include any documents.")

    document = documents[0]
    groups = document.get("groups") or []
    return {
        "id": document.get("authUsersId"),
        "loginId": document.get("username"),
        "name": document.get("name"),
        "email": document.get("email"),
        "groupName": groups[0].get("name") if groups else None,
        "userId": document.get("authUsersId"),
        "status": document.get("status"),
    }


async def read_tokens_from_frame_or_page(target) -> dict | None:
    try:
        return await target.evaluate(
            """() => ({
                access_token: localStorage.getItem('accessToken'),
                refresh_token: localStorage.getItem('refreshToken'),
                chat_room_id: localStorage.getItem('CHAT_ROOM_ID'),
                current_url: location.href,
            })"""
        )
    except Exception:
        return None


async def wait_for_tokens(context, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for page in context.pages:
            tokens = await read_tokens_from_frame_or_page(page)
            if tokens and tokens.get("access_token"):
                return tokens
            for frame in page.frames:
                frame_tokens = await read_tokens_from_frame_or_page(frame)
                if frame_tokens and frame_tokens.get("access_token"):
                    return frame_tokens
        await asyncio.sleep(1)
    raise RuntimeError("Timed out while waiting for a KingoGPT accessToken.")


async def find_login_frame(page):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for frame in page.frames:
            if "kingoinfo.skku.edu/gaia/nxui/outdex.html" in frame.url:
                return frame
        await asyncio.sleep(0.5)
    return None


async def attempt_auto_login(page, login_id: str, password: str) -> bool:
    login_frame = await find_login_frame(page)
    if login_frame is None:
        return False

    try:
        await login_frame.wait_for_selector('input[id*="edtKingoID:input"]', timeout=20_000)
        await login_frame.evaluate(
            """([providedLoginId, providedPassword]) => {
                const bySuffix = (suffix) =>
                    Array.from(document.querySelectorAll('input')).find((el) =>
                        el.id.endsWith(suffix)
                    );
                const targets = [
                    [bySuffix('edtKingoID:input'), providedLoginId],
                    [bySuffix('edtPassWord:input'), providedPassword],
                    [bySuffix('edtKorNameDummy:input'), providedLoginId],
                    [bySuffix('edtPassWordDummy:input'), providedPassword],
                ];
                for (const [element, value] of targets) {
                    if (!element) continue;
                    element.focus();
                    element.value = value;
                    element.dispatchEvent(new Event('input', { bubbles: true }));
                    element.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }""",
            [login_id, password],
        )
        await login_frame.locator('input[id*="edtPassWord:input"]').press("Enter")
        await login_frame.evaluate(
            """() => {
                const loginButton = Array.from(document.querySelectorAll('div')).find((el) =>
                    el.id.endsWith('btnLogin')
                );
                if (loginButton) {
                    loginButton.click();
                }
            }"""
        )
        try:
            await login_frame.locator('div[id$="btnLogin"]').click(timeout=5_000)
        except Exception:
            pass

        deadline = time.monotonic() + AUTO_LOGIN_WAIT_SECONDS
        while time.monotonic() < deadline:
            page_tokens = await read_tokens_from_frame_or_page(page)
            frame_tokens = await read_tokens_from_frame_or_page(login_frame)
            if (page_tokens and page_tokens.get("access_token")) or (
                frame_tokens and frame_tokens.get("access_token")
            ):
                return True
            await asyncio.sleep(1)
    except Exception:
        return False
    return False


def write_cache(path_str: str, cache_data: dict) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")


async def refresh_token_cache(args: argparse.Namespace) -> dict:
    login_id, password = resolve_credentials(args)
    if not login_id or not password:
        raise RuntimeError(
            "Automatic login requires credentials. Pass --id and --password, "
            "set KINGOGPT_ID/KINGOGPT_PW, or put id/password in kingogpt_config.json."
        )

    user_data_dir = Path(args.profile_dir)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as playwright:
        launch_options = {
            "user_data_dir": str(user_data_dir),
            "headless": True,
            "args": ["--no-sandbox", "--disable-setuid-sandbox"],
        }
        if args.chrome_path:
            launch_options["executable_path"] = args.chrome_path

        context = await playwright.chromium.launch_persistent_context(**launch_options)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            print("[*] Opening KingoGPT...")
            await page.goto(START_URL, wait_until="domcontentloaded", timeout=60_000)

            print("[*] Attempting automatic login...")
            if not await attempt_auto_login(page, login_id, password):
                raise RuntimeError("Automatic KingoGPT login failed or timed out.")

            tokens = await wait_for_tokens(context, args.timeout)
            claims = decode_jwt_payload(tokens["access_token"])
            profile = await asyncio.to_thread(fetch_user_profile, tokens["access_token"])

            expires_at = claims.get("exp")
            cache_data = {
                "access_token": tokens["access_token"],
                "refresh_token": tokens.get("refresh_token"),
                "chat_room_id": int(tokens["chat_room_id"]) if tokens.get("chat_room_id") else None,
                "captured_from": tokens.get("current_url"),
                "fetched_at": int(time.time()),
                "expires_at": expires_at,
                "expires_at_iso": (
                    time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(expires_at))
                    if expires_at
                    else None
                ),
                "claims": claims,
                "user": profile,
            }
            write_cache(args.cache_file, cache_data)
            return cache_data
        finally:
            await context.close()


async def async_main(args: argparse.Namespace) -> int:
    cache_data = await refresh_token_cache(args)
    print(f"[*] Token cache written: {args.cache_file}")
    if cache_data.get("expires_at"):
        print(
            "[*] Access token expires at: "
            + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cache_data["expires_at"]))
        )
    return 0


def main() -> int:
    configure_output()
    try:
        return asyncio.run(async_main(parse_args()))
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"[!] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
