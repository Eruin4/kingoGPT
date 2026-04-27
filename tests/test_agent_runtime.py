import unittest

from internal_agent.agent.loop import Agent
from internal_agent.agent.parser import extract_json, validate_action


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class ParserTests(unittest.TestCase):
    def test_extracts_pure_json(self):
        self.assertEqual(
            extract_json('{"action":"final","args":{"answer":"ok"}}')["action"],
            "final",
        )

    def test_extracts_fenced_json(self):
        obj = extract_json('```json\n{"action":"final","args":{"answer":"ok"}}\n```')
        self.assertEqual(obj["args"]["answer"], "ok")

    def test_extracts_json_surrounded_by_prose(self):
        obj = extract_json('Here is the action: {"action":"search_docs","args":{"query":"x"}} thanks')
        self.assertEqual(obj["action"], "search_docs")

    def test_validate_action_adds_missing_args(self):
        obj = validate_action({"action": "search_docs"})
        self.assertEqual(obj["args"], {})

    def test_validate_action_rejects_final_without_answer(self):
        with self.assertRaises(ValueError):
            validate_action({"action": "final", "args": {}})

    def test_validate_action_accepts_legacy_reply(self):
        obj = validate_action({"reply": "ok"})
        self.assertEqual(obj, {"action": "final", "args": {"answer": "ok"}})

    def test_validate_action_rejects_unknown_action(self):
        with self.assertRaises(ValueError):
            validate_action({"action": "shell", "args": {}})


class AgentLoopTests(unittest.TestCase):
    def test_immediate_final(self):
        llm = FakeLLM(['{"action":"final","args":{"answer":"done"}}'])
        agent = Agent(llm)
        self.assertEqual(agent.run("task"), "done")

    def test_tool_then_final(self):
        llm = FakeLLM(
            [
                '{"action":"search_docs","args":{"query":"deploy"}}',
                '{"action":"final","args":{"answer":"summarized"}}',
            ]
        )
        agent = Agent(llm)
        self.assertEqual(agent.run("task"), "summarized")
        self.assertIn("[mock search result] query=deploy", llm.prompts[-1])

    def test_repairs_invalid_json(self):
        llm = FakeLLM(
            [
                "not json",
                '{"action":"final","args":{"answer":"fixed"}}',
            ]
        )
        agent = Agent(llm)
        self.assertEqual(agent.run("task"), "fixed")
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("previous output was invalid", llm.prompts[1].lower())

    def test_max_steps(self):
        llm = FakeLLM(['{"action":"search_docs","args":{"query":"again"}}'])
        agent = Agent(llm, max_steps=2)
        self.assertEqual(agent.run("task"), "Stopped: max_steps reached without final answer.")

    def test_text_fallback_when_repair_fails(self):
        llm = FakeLLM(["plain answer", "still not json"])
        agent = Agent(llm)
        self.assertEqual(agent.run("task"), "plain answer")


class OpenAICompatTests(unittest.TestCase):
    def test_chat_completion_endpoint_with_mock_agent(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        class FakeAgent:
            def __init__(self):
                self.task = None

            def run(self, task):
                self.task = task
                return "api ok"

        fake_agent = FakeAgent()
        original_agent = compat._agent
        compat._agent = fake_agent
        try:
            client = TestClient(compat.app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "internal-azure-web-agent",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["choices"][0]["message"]["content"], "api ok")
            self.assertEqual(fake_agent.task, "hello")
        finally:
            compat._agent = original_agent

    def test_health_endpoint(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        client = TestClient(compat.app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
