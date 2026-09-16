"""Workspace functions and CLI for the KingoGPT agent."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


class Workspace:
    """File operations confined to an explicitly selected workspace."""

    def __init__(self, root: str | Path, *, allow_write: bool = False):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory.")
        self.allow_write = allow_write
        self.max_bytes = 128_000

    def _path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Use a path relative to the selected workspace.")
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Path escapes the selected workspace.")
        for part in resolved.relative_to(self.root).parts:
            if (part in {".git", ".ssh", ".aws", ".venv", "state", ".kingogpt", "node_modules", "__pycache__"}
                    or part.startswith(".env") or "token_cache" in part
                    or "chrome_profile" in part or part.endswith((".pem", ".key"))):
                raise ValueError("This path is excluded from agent file access.")
        return resolved

    def list_directory(self, path: str = ".") -> str:
        """List immediate children of a workspace-relative directory."""
        folder = self._path(path)
        items = []
        for item in sorted(folder.iterdir()):
            relative = str(item.relative_to(self.root))
            try:
                self._path(relative)
            except ValueError:
                continue
            items.append(relative + ("/" if item.is_dir() else ""))
            if len(items) == 200:
                break
        return json.dumps(items, ensure_ascii=False)

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file, at most 128 KB, using a workspace-relative path."""
        target = self._path(path)
        with target.open("rb") as file:
            data = file.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("File exceeds the 128 KB read limit.")
        return data.decode("utf-8")

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a UTF-8 file with the complete supplied content."""
        if not self.allow_write:
            raise PermissionError("File writing is disabled.")
        target = self._path(path)
        data = content.encode("utf-8")
        if len(data) > self.max_bytes:
            raise ValueError("Content exceeds the 128 KB write limit.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return json.dumps({"path": path, "bytes_written": len(data)})


def workspace_functions(workspace: Workspace):
    from agents import function_tool

    functions = [function_tool(workspace.list_directory), function_tool(workspace.read_file)]
    if workspace.allow_write:
        functions.append(function_tool(workspace.write_file))
    return functions


async def run_agent(args):
    from kingogpt.agent_runtime import run_cli
    return await run_cli(args)


def main():
    parser = argparse.ArgumentParser(description="Run an agent against your KingoGPT gateway.")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--allow-write", action="store_true", help="Enable file creation/overwrite in the selected workspace.")
    parser.add_argument("--local", action="store_true", help="Use KingoGPT directly; no separate API server needed.")
    parser.add_argument("--exec", dest="exec_policy", choices=("deny", "ask", "allow"), default="deny",
                        help="Command execution policy. NOT an OS sandbox; commands have your user privileges.")
    parser.add_argument("--verify-command", help="A fixed acceptance command that must pass before completion.")
    parser.add_argument("--command-timeout", type=int, default=60)
    parser.add_argument("--run-dir", help="New run directory (transcript, checkpoints, original-file backups).")
    parser.add_argument("--resume", help="Resume an existing run directory. Permissions must be granted again.")
    parser.add_argument("--base-url", default=os.getenv("KINGOGPT_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("KINGOGPT_SERVER_API_KEY", "local"))
    parser.add_argument("--max-turns", type=int, default=40, help="Additional model turns for this invocation.")
    parser.add_argument("--api-retries", type=int, default=3, help="Additional transient-error retries per model turn.")
    args = parser.parse_args()
    if not args.prompt and not args.resume:
        parser.error("provide a prompt or --resume")
    if args.resume and (args.run_dir or args.prompt):
        parser.error("--resume cannot be combined with --run-dir or a new prompt")
    if args.max_turns < 1 or not 0 <= args.api_retries <= 10 or not 1 <= args.command_timeout <= 300:
        parser.error("max-turns > 0, api-retries 0..10, command-timeout 1..300 required")
    if args.verify_command and args.exec_policy == "deny":
        parser.error("--verify-command requires --exec ask or --exec allow")
    try:
        code = asyncio.run(run_agent(args))
    except ImportError as exc:
        parser.exit(1, f"Install agent dependencies with: pip install -e '.[agent]' ({exc})\n")
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(1, f"Agent stopped: {exc}\n")
    else:
        parser.exit(code)


if __name__ == "__main__":
    main()
