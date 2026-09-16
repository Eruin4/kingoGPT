import asyncio
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from kingogpt.agent_runtime import AgentLoop, AgentWorkspace, RunStore, digest


def call(name, **arguments):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_" + os.urandom(8).hex(), "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name, "workspace")
        self.root.mkdir()
        self.store = RunStore(Path(self.temp.name, "run"), workspace=self.root, prompt="Fix the code and verify.")
        self.workspace = AgentWorkspace(self.root, self.store, allow_write=True)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_read_before_write_and_original_backup(self):
        target = self.root / "a.py"
        target.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.workspace.write_file("a.py", "replacement")
        self.workspace.read_file("a.py")
        self.workspace.write_file("a.py", "replacement")
        backup = self.store.state["backups"]["a.py"]
        self.assertEqual((self.store.root / backup["file"]).read_text(), "original")
        self.workspace.read_file("a.py")
        self.workspace.write_file("a.py", "third version")
        self.assertEqual((self.store.root / backup["file"]).read_text(), "original")
        self.assertIn("a.py", self.workspace.completion_blockers()[0])
        self.workspace.read_file("a.py")
        self.assertEqual(self.workspace.completion_blockers(), [])

    def test_concurrent_change_is_preserved(self):
        target = self.root / "a.txt"
        target.write_text("first")
        self.workspace.read_file("a.txt")
        target.write_text("user edit")
        with self.assertRaises(ValueError):
            self.workspace.write_file("a.txt", "agent edit")
        self.assertEqual(target.read_text(), "user edit")

    def test_line_ranges_allow_reading_beyond_model_output_limit(self):
        (self.root / "large.txt").write_text("first\n" + "x" * 21000 + "\ntail\n")
        self.assertEqual(self.workspace.read_file("large.txt", start_line=3, end_line=3), "tail\n")
        with self.assertRaises(ValueError):
            self.workspace.read_file("large.txt", start_line=0)

    def test_write_intent_survives_failure_before_replacement(self):
        from unittest.mock import patch
        from kingogpt.agent_runtime import atomic_write
        target = self.root / "a.txt"
        target.write_text("original")
        self.workspace.read_file("a.txt")
        def fail_target(path, data, mode=0o600):
            if path == target:
                raise OSError("simulated interruption before replacement")
            return atomic_write(path, data, mode)
        with patch("kingogpt.agent_runtime.atomic_write", side_effect=fail_target):
            with self.assertRaises(OSError):
                self.workspace.write_file("a.txt", "proposed")
        persisted = json.loads((self.store.root / "state.json").read_text())
        self.assertIn("a.txt", persisted["writes"])
        self.assertTrue(self.workspace.completion_blockers())
        self.assertEqual(target.read_text(), "original")

    def test_precise_replace_and_search(self):
        (self.root / "a.py").write_text("return a - b\n# preserve me\n")
        self.workspace.read_file("a.py")
        self.workspace.replace_text("a.py", "a - b", "a + b")
        self.assertEqual((self.root / "a.py").read_text(), "return a + b\n# preserve me\n")
        found = json.loads(self.workspace.search_files("preserve"))
        self.assertEqual(found["matches"][0]["line"], 2)
        with self.assertRaises(ValueError):
            self.workspace.replace_text("a.py", "does not exist", "x")

    def test_permission_and_special_paths(self):
        readonly = AgentWorkspace(self.root, self.store)
        for name in ("write_file", "replace_text", "run_command"):
            self.assertNotIn(name, [tool["function"]["name"] for tool in readonly.definitions()])
        with self.assertRaises(PermissionError):
            readonly.run_command("python -V")
        (self.root / "state").mkdir()
        (self.root / "state" / "secret").write_text("secret")
        (self.root / "link").symlink_to(self.root / "state" / "secret")
        with self.assertRaises(ValueError):
            readonly.read_file("link")
        (self.root / "pipe").mkfifo() if hasattr(Path, "mkfifo") else os.mkfifo(self.root / "pipe")
        with self.assertRaises(ValueError):
            readonly.read_file("pipe")
        self.assertEqual(json.loads(readonly.search_files("secret"))["matches"], [])

    def test_hardlink_not_overwritten(self):
        (self.root / "a").write_text("same")
        os.link(self.root / "a", self.root / "b")
        self.workspace.read_file("a")
        with self.assertRaises(ValueError):
            self.workspace.write_file("a", "different")

    def test_checkpoint_private_and_locked(self):
        self.assertEqual((self.store.root / "state.json").stat().st_mode & 0o777, 0o600)
        with self.assertRaises(BlockingIOError):
            RunStore(self.store.root)
        self.assertNotIn("api_key", self.store.state)

    def test_command_approval_and_real_output(self):
        denied = AgentWorkspace(self.root, self.store, exec_policy="ask", approve=lambda *_: False)
        with self.assertRaises(PermissionError):
            denied.run_command("python -V")
        allowed = AgentWorkspace(self.root, self.store, exec_policy="ask", approve=lambda *_: True)
        result = json.loads(allowed.run_command("python -c 'print(6 + 8 + 23)'"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"].strip(), "37")

    def test_command_timeout_and_output_limit(self):
        allowed = AgentWorkspace(self.root, self.store, exec_policy="allow", command_timeout=1)
        result = json.loads(allowed.run_command("python -c 'import time; time.sleep(10)'"))
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["ok"])
        result = json.loads(allowed.run_command("python -c 'print(\"x\" * 100000)'"))
        self.assertTrue(result["output_limit"])
        self.assertLessEqual(len(result["output"]), 32000)

    def test_command_does_not_inherit_api_secrets(self):
        from unittest.mock import patch
        allowed = AgentWorkspace(self.root, self.store, exec_policy="allow")
        with patch.dict(os.environ, {"KINGOGPT_SERVER_API_KEY": "synthetic-secret"}):
            result = json.loads(allowed.run_command("python -c 'import os; print(os.getenv(\"KINGOGPT_SERVER_API_KEY\"))'"))
        self.assertEqual(result["output"].strip(), "None")

    def test_acceptance_invalidated_by_write(self):
        command = "python -c 'print(42)'"
        self.store.state["verify_command"] = command
        self.assertTrue(self.workspace.completion_blockers())
        self.workspace.exec_policy = "allow"
        self.workspace.run_command(command)
        self.assertEqual(self.workspace.completion_blockers(), [])
        self.workspace.write_file("new.txt", "changed")
        self.assertTrue(any("Acceptance" in msg for msg in self.workspace.completion_blockers()))

    def test_embedded_run_directory_excluded(self):
        nested = RunStore(self.root / "private-run", workspace=self.root, prompt="test")
        try:
            workspace = AgentWorkspace(self.root, nested)
            with self.assertRaises(ValueError):
                workspace.read_file("private-run/state.json")
            self.assertNotIn("private-run/", json.loads(workspace.list_directory()))
        finally:
            nested.close()


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name, "workspace")
        self.root.mkdir()
        self.store = RunStore(Path(self.temp.name, "run"), workspace=self.root, prompt="Write result.txt and verify.")
        self.workspace = AgentWorkspace(self.root, self.store, allow_write=True)

    async def asyncTearDown(self):
        self.store.close()
        self.temp.cleanup()

    def sequence(self, items):
        iterator = iter(items)
        async def complete(messages, tools):
            item = next(iterator)
            if isinstance(item, Exception):
                raise item
            return item
        return complete

    async def test_final_requires_real_readback(self):
        complete = self.sequence([
            call("write_file", path="result.txt", content="37"),
            {"role": "assistant", "content": "I verified everything."},
            call("read_file", path="result.txt"),
            {"role": "assistant", "content": "Verified actual file: 37"},
        ])
        answer, success = await AgentLoop(self.store, self.workspace, complete).run()
        self.assertTrue(success)
        self.assertIn("37", answer)
        self.assertTrue(any(x["kind"] == "completion_rejected" for x in self.store.state["events"]))

    async def test_pause_resume_keeps_tool_results(self):
        await AgentLoop(self.store, self.workspace, self.sequence([
            call("write_file", path="result.txt", content="42")]), max_turns=1).run()
        self.assertEqual(self.store.state["status"], "paused")
        async def complete(messages, tools):
            self.assertTrue(any(msg["role"] == "tool" and "bytes_written" in msg["content"] for msg in messages))
            return call("read_file", path="result.txt")
        await AgentLoop(self.store, self.workspace, complete, max_turns=1).run()
        _, success = await AgentLoop(self.store, self.workspace, self.sequence([
            {"role": "assistant", "content": "Verified 42"}])).run()
        self.assertTrue(success)

    async def test_interrupted_call_never_replayed(self):
        message = call("write_file", path="result.txt", content="old proposal")
        self.store.state["messages"].append(message)
        self.store.state["pending"] = message["tool_calls"]
        (self.root / "result.txt").write_text("user recovered value")
        async def complete(messages, tools):
            self.assertIn("interrupted_execution", messages[-1]["content"])
            return {"role": "assistant", "content": "Paused action was not repeated."}
        await AgentLoop(self.store, self.workspace, complete).run()
        self.assertEqual((self.root / "result.txt").read_text(), "user recovered value")

    async def test_unavailable_tool_is_not_executed(self):
        self.workspace.allow_write = False
        await AgentLoop(self.store, self.workspace, self.sequence([
            call("write_file", path="result.txt", content="must not write")]), max_turns=1).run()
        self.assertFalse((self.root / "result.txt").exists())
        self.assertIn("not enabled", self.store.state["messages"][-1]["content"])

    async def test_repeated_operation_stops(self):
        async def complete(*_):
            return call("list_directory", path=".")
        with self.assertRaisesRegex(RuntimeError, "Repeated identical"):
            await AgentLoop(self.store, self.workspace, complete).run()
        self.assertEqual(self.store.state["status"], "interrupted")

    async def test_transient_api_failure_retries_without_side_effects(self):
        from openai import APIConnectionError
        try:
            import httpx2 as httpx
        except ImportError:
            import httpx
        error = APIConnectionError(request=httpx.Request("POST", "http://local/v1/chat/completions"))
        answer, success = await AgentLoop(self.store, self.workspace, self.sequence([
            error, {"role": "assistant", "content": "Recovered"}]), retry_delay=0).run()
        self.assertTrue(success)
        self.assertEqual(answer, "Recovered")
        self.assertEqual(self.store.state["turns"], 1)
        self.assertEqual(self.store.state["writes"], {})

    async def test_context_keeps_matching_calls_and_results(self):
        for number in range(20):
            message = call("read_file", path=f"file{number}.txt")
            self.store.state["messages"].extend([message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "x" * 4000}])
        loop = AgentLoop(self.store, self.workspace, None)
        visible = loop.model_messages()
        calls = {call["id"] for message in visible for call in message.get("tool_calls", [])}
        self.assertTrue(all(msg["tool_call_id"] in calls for msg in visible if msg["role"] == "tool"))
        self.assertLess(len(json.dumps(visible)), 48000)
        self.assertEqual(len(self.store.state["messages"]), 41)


if __name__ == "__main__":
    unittest.main()
