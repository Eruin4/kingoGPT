import json
import unittest

from fastapi.testclient import TestClient

from kingogpt.openai_server import ServerSettings, create_app


class FakeProvider:
    def __init__(self, answer="hello from kingogpt", chunks=None):
        self.answer = answer
        self.chunks = chunks
        self.calls = []

    def complete(self, prompt, instruction=None, on_chunk=None):
        self.calls.append({"prompt": prompt, "instruction": instruction})
        if on_chunk is not None:
            for chunk in self.chunks or [self.answer]:
                on_chunk(chunk)
        return self.answer


def client_for(provider, *, api_key=None):
    settings = ServerSettings(api_key=api_key, model_id="kingogpt-test")
    return TestClient(create_app(settings=settings, provider=provider))


class ModelsAndHealthTests(unittest.TestCase):
    def test_health_does_not_require_api_key(self):
        with client_for(FakeProvider(), api_key="secret") as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model": "kingogpt-test"})

    def test_models_uses_openai_list_shape(self):
        with client_for(FakeProvider()) as client:
            response = client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        self.assertEqual(response.json()["data"][0]["id"], "kingogpt-test")

    def test_api_key_is_enforced(self):
        with client_for(FakeProvider(), api_key="secret") as client:
            denied = client.get("/v1/models")
            allowed = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(denied.json()["error"]["code"], "invalid_api_key")
        self.assertEqual(allowed.status_code, 200)


class ChatCompletionTests(unittest.TestCase):
    def test_basic_chat_completion(self):
        provider = FakeProvider("answer")
        with client_for(provider) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-test",
                    "messages": [
                        {"role": "system", "content": "be brief"},
                        {"role": "user", "content": "hello"},
                    ],
                },
            )
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "answer")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertIn("SYSTEM\nbe brief", provider.calls[0]["prompt"])
        self.assertIn("USER\nhello", provider.calls[0]["prompt"])

    def test_unknown_model_returns_openai_error(self):
        with client_for(FakeProvider()) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "wrong", "messages": [{"role": "user", "content": "hi"}]},
            )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")

    def test_function_tool_call(self):
        provider = FakeProvider(
            json.dumps(
                {"type": "tool_call", "name": "weather", "arguments": {"city": "Seoul"}}
            )
        )
        tool = {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        with client_for(provider) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-test",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": [tool],
                },
            )
        choice = response.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "weather")
        self.assertEqual(
            json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]),
            {"city": "Seoul"},
        )
        self.assertIsNone(provider.calls[0]["instruction"])
        self.assertIn("Parameters JSON schema", provider.calls[0]["prompt"])

    def test_tool_result_follow_up_is_rendered(self):
        provider = FakeProvider("final answer")
        with client_for(provider) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-test",
                    "messages": [
                        {"role": "user", "content": "weather?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city":"Seoul"}'},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                    ],
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("TOOL RESULT call_1\nsunny", provider.calls[0]["prompt"])

    def test_streaming_chat_completion(self):
        provider = FakeProvider("hello", chunks=["hel", "lo"])
        with client_for(provider) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as response:
                lines = [line for line in response.iter_lines() if line]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(lines[-1], "data: [DONE]")
        payloads = [json.loads(line[6:]) for line in lines[:-1]]
        content = "".join(
            item["choices"][0]["delta"].get("content", "")
            for item in payloads
            if item.get("choices")
        )
        self.assertEqual(content, "hello")
        self.assertIn("usage", payloads[-1])


class ResponsesAPITests(unittest.TestCase):
    def test_string_input_returns_output_text(self):
        provider = FakeProvider("response answer")
        with client_for(provider) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "kingogpt-test", "input": "hello"},
            )
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output"][0]["type"], "message")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "response answer")
        self.assertEqual(payload["usage"]["total_tokens"], payload["usage"]["input_tokens"] + payload["usage"]["output_tokens"])

    def test_flat_function_tool_returns_function_call(self):
        provider = FakeProvider(
            json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"q": "x"}})
        )
        with client_for(provider) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "kingogpt-test",
                    "input": "look it up",
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "description": "Lookup a value",
                            "parameters": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                                "required": ["q"],
                            },
                        }
                    ],
                },
            )
        item = response.json()["output"][0]
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["name"], "lookup")
        self.assertEqual(json.loads(item["arguments"]), {"q": "x"})

    def test_function_call_output_follow_up(self):
        provider = FakeProvider("done")
        with client_for(provider) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "kingogpt-test",
                    "input": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": '{"q":"x"}',
                        },
                        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
                    ],
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("TOOL RESULT call_1\nresult", provider.calls[0]["prompt"])

    def test_streaming_response_events(self):
        provider = FakeProvider("hello", chunks=["he", "llo"])
        with client_for(provider) as client:
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "kingogpt-test", "input": "hi", "stream": True},
            ) as response:
                lines = [line for line in response.iter_lines() if line]
        self.assertEqual(response.status_code, 200)
        event_names = [line[7:] for line in lines if line.startswith("event: ")]
        self.assertEqual(event_names[0], "response.created")
        self.assertIn("response.output_text.delta", event_names)
        self.assertEqual(event_names[-1], "response.completed")
        data = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
        deltas = [item["delta"] for item in data if item["type"] == "response.output_text.delta"]
        self.assertEqual("".join(deltas), "hello")


if __name__ == "__main__":
    unittest.main()
