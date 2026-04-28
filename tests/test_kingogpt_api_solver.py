import argparse
import unittest

import kingogpt_api_solver as solver


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


if __name__ == "__main__":
    unittest.main()
