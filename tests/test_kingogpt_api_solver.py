import argparse
import unittest
from unittest import mock

import kingogpt_api_solver as solver
import kingogpt_token_capture


class KingoGPTApiSolverPromptTests(unittest.TestCase):
    def test_build_request_prompt_is_raw_user_prompt(self):
        prompt = solver.build_request_prompt("hello", None)

        self.assertEqual(prompt, "USER:\nhello")
        self.assertNotIn("KingoClaw", prompt)
        self.assertNotIn("JSON Action Contract", prompt)
        self.assertNotIn('{"call"', prompt)
        self.assertNotIn("memory_search", prompt)

    def test_build_request_prompt_keeps_user_system_prefix_only(self):
        prompt = solver.build_request_prompt("hello", "be brief")

        self.assertEqual(prompt, "SYSTEM:\nbe brief\n\nUSER:\nhello")
        self.assertNotIn("Reply with", prompt)
        self.assertNotIn('{"reply"', prompt)

    def test_build_payload_uses_prompt_without_contract_wrapping(self):
        args = argparse.Namespace(scenario_id="scenario-1")
        user = {"id": 123, "loginId": "user1"}

        payload = solver.build_payload(user, "USER:\nhello", args)

        self.assertEqual(payload["queries"], {"type": "text", "text": "USER:\nhello"})
        self.assertEqual(payload["users_id"], 123)
        self.assertEqual(payload["scenarios_id"], "scenario-1")
        self.assertEqual(payload["llms"]["reply_style_prompt"], "normal")
        self.assertNotIn("instruction", payload)

    def test_build_payload_sends_system_prompt_as_instruction(self):
        args = argparse.Namespace(scenario_id="scenario-1")
        user = {"id": 123, "loginId": "user1"}

        payload = solver.build_payload(user, "USER:\nhello", args, instruction="be brief")

        self.assertEqual(payload["queries"], {"type": "text", "text": "USER:\nhello"})
        self.assertEqual(payload["instruction"], "be brief")
        self.assertEqual(payload["llms"]["reply_style_prompt"], "be brief")

    def test_build_payload_does_not_send_oversized_reply_style_prompt(self):
        args = argparse.Namespace(scenario_id="scenario-1")
        user = {"id": 123, "loginId": "user1"}
        instruction = "x" * (solver.MAX_REPLY_STYLE_PROMPT_CHARS + 1)

        payload = solver.build_payload(user, "USER:\nhello", args, instruction=instruction)

        self.assertEqual(payload["instruction"], instruction)
        self.assertEqual(payload["llms"]["reply_style_prompt"], "normal")


class KingoGPTTokenRefreshTests(unittest.TestCase):
    def test_corrupt_token_cache_error_triggers_auto_refresh(self):
        self.assertTrue(
            solver.should_auto_refresh_token(
                RuntimeError("Failed to parse token cache file: state/kingogpt_token_cache.json")
            )
        )

    def test_resolve_credentials_reads_config_file(self):
        args = argparse.Namespace(
            config_file="kingogpt_config.json",
            login_id=None,
            password=None,
        )
        with mock.patch.object(
            kingogpt_token_capture,
            "candidate_config_files",
            return_value=["kingogpt_config.json"],
        ):
            with mock.patch.object(
                kingogpt_token_capture,
                "load_json_file",
                return_value={"id": "user1", "password": "pw1"},
            ):
                self.assertEqual(
                    kingogpt_token_capture.resolve_credentials(args),
                    ("user1", "pw1"),
                )

    def test_refresh_token_cache_passes_capture_defaults(self):
        args = argparse.Namespace(
            no_auto_refresh_token=False,
            token_cache="cache.json",
            token_config="config.json",
            profile_dir="profile",
            token_refresh_timeout=123,
        )
        expected = {"access_token": "token"}
        captured = {}

        async def fake_refresh(capture_args):
            captured["args"] = capture_args
            return expected

        with mock.patch.object(kingogpt_token_capture, "refresh_token_cache", fake_refresh):
            result = solver.refresh_token_cache(args)

        self.assertEqual(result, expected)
        capture_args = captured["args"]
        self.assertIsNone(capture_args.chrome_path)


if __name__ == "__main__":
    unittest.main()
