"""Live acceptance probe: repair code, execute tests, pause, and resume.

Uses the configured KingoGPT account. All task files are disposable.
"""
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="kingogpt-runtime-probe-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        workspace.mkdir()
        nonce = secrets.token_hex(8)
        original = 'def add(a, b):\n    return a - b\n\n# preserve: ' + nonce + '\n'
        (workspace / "calc.py").write_text(original)
        tests = 'import unittest\nfrom calc import add\nclass TestAdd(unittest.TestCase):\n    def test_values(self):\n        self.assertEqual(add(17,25),42)\n        self.assertEqual(add(-3,8),5)\n'
        (workspace / "test_calc.py").write_text(tests)
        run = root / "run"
        base = [sys.executable, "-m", "kingogpt.agent", "--local", "--allow-write", "--exec", "allow"]
        env = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1"}
        prompt = ("Fix add in calc.py so the provided tests pass. Preserve the comment exactly. "
                  "Do not modify test_calc.py. Read actual files, make the smallest edit, "
                  "run python -m unittest discover -v, read back calc.py after testing, "
                  "and report the preserved reference. Work only in this temporary workspace. "
                  "Do not install dependencies or run network commands.")
        first = subprocess.run(base + ["--workspace", str(workspace), "--run-dir", str(run),
                                      "--max-turns", "1", "--verify-command", "python -m unittest discover -v", prompt],
                               cwd=project, env=env, timeout=300)
        if first.returncode != 2:
            raise RuntimeError(f"Expected a paused run, got exit {first.returncode}")
        state = json.loads((run / "state.json").read_text())
        if state["status"] != "paused" or state["turns"] != 1:
            raise RuntimeError("First checkpoint did not preserve the first model turn")
        resumed = subprocess.run(base + ["--resume", str(run), "--max-turns", "18"],
                                 cwd=project, env=env, timeout=900)
        if resumed.returncode != 0:
            raise RuntimeError(f"Resume failed with exit {resumed.returncode}")
        state = json.loads((run / "state.json").read_text())
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-v"], cwd=workspace,
                                env=env, capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError("Independent acceptance tests failed")
        if (workspace / "test_calc.py").read_text() != tests:
            raise RuntimeError("Agent changed acceptance tests")
        if nonce not in (workspace / "calc.py").read_text() or nonce not in state["final"]:
            raise RuntimeError("Agent did not preserve/report the real file reference")
        backup = state["backups"]["calc.py"]["file"]
        if (run / backup).read_text() != original:
            raise RuntimeError("Original file backup is incorrect")
        if state["verified_revision"] != state["revision"] or state["status"] != "completed":
            raise RuntimeError("Run completed without acceptance evidence")
        print(json.dumps({"status": "passed", "turns": state["turns"], "pause_resume": True,
                          "real_tests": True, "original_backup": True}), flush=True)


if __name__ == "__main__":
    main()
