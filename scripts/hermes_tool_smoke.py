import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any


DEFAULT_PROMPT = (
    "Use your available tools to inspect the current working directory, then answer "
    "with the exact phrase TOOL_SMOKE_OK and one filename you found."
)


def run(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def latest_request_dump(home: pathlib.Path, since: float) -> pathlib.Path | None:
    sessions = home / "sessions"
    candidates = [
        path
        for path in sessions.glob("request_dump_*.json")
        if path.stat().st_mtime >= since
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_session_log(home: pathlib.Path, since: float) -> pathlib.Path | None:
    sessions = home / "sessions"
    candidates = [
        path
        for path in sessions.glob("session_*.json")
        if path.stat().st_mtime >= since
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def contains_tool_calls(value: Any) -> bool:
    if isinstance(value, dict):
        if "tool_calls" in value:
            return True
        return any(contains_tool_calls(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_tool_calls(item) for item in value)
    return False


def contains_tool_role(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("role") == "tool":
            return True
        return any(contains_tool_role(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_tool_role(item) for item in value)
    return False


def request_tool_count(path: pathlib.Path | None) -> int | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    body = data.get("request", {}).get("body", {})
    tools = body.get("tools")
    if isinstance(tools, list):
        return len(tools)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Hermes oneshot prompt through the same OpenAI-compatible provider "
            "used by Discord, then report whether tool metadata/tool calls appeared."
        )
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--hermes",
        default=os.getenv("HERMES_BIN", "/home/eruin/kingoGPT/hermes/venv/bin/python"),
        help="Hermes python executable or hermes command.",
    )
    parser.add_argument(
        "--home",
        default=os.getenv("HERMES_HOME", "/home/eruin/.hermes"),
        help="Hermes home directory.",
    )
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    hermes_home = pathlib.Path(args.home)
    started = time.time()

    if args.hermes.endswith("python"):
        command = [args.hermes, "-m", "hermes_cli.main", "-z", args.prompt]
    else:
        command = [args.hermes, "-z", args.prompt]

    proc = run(command, timeout=args.timeout)
    dump_path = latest_request_dump(hermes_home, started)
    session_path = latest_session_log(hermes_home, started)
    tool_count = request_tool_count(dump_path)
    output_has_tool_phrase = bool(re.search(r"TOOL_SMOKE_OK", proc.stdout))
    dump_has_tool_calls = False
    if dump_path is not None:
        dump_has_tool_calls = contains_tool_calls(
            json.loads(dump_path.read_text(encoding="utf-8"))
        )
    session_has_tool_calls = False
    session_has_tool_result = False
    if session_path is not None:
        session_data = json.loads(session_path.read_text(encoding="utf-8"))
        session_has_tool_calls = contains_tool_calls(session_data)
        session_has_tool_result = contains_tool_role(session_data)

    report = {
        "command_exit_code": proc.returncode,
        "prompt": args.prompt,
        "request_dump": str(dump_path) if dump_path else None,
        "session_log": str(session_path) if session_path else None,
        "request_tool_count": tool_count,
        "dump_has_tool_calls": dump_has_tool_calls,
        "session_has_tool_calls": session_has_tool_calls,
        "session_has_tool_result": session_has_tool_result,
        "output_has_tool_phrase": output_has_tool_phrase,
        "stdout_tail": proc.stdout[-2000:],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if proc.returncode != 0:
        return proc.returncode
    if not session_has_tool_calls or not session_has_tool_result:
        return 2
    if not output_has_tool_phrase:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
