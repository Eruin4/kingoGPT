import json
import unittest
from unittest import mock

from internal_agent.standalone.agent.loop import Agent
from internal_agent.standalone.agent.parser import extract_json, validate_action


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
    def test_kingogpt_tool_call_json_maps_to_openai_tool_call(self):
        from kingogpt.tool_adapter import convert_kingogpt_json_to_openai_message

        message = convert_kingogpt_json_to_openai_message(
            '{"type":"tool_call","name":"read","arguments":{"path":"README.md"}}'
        )

        self.assertIsNone(message["content"])
        tool_call = message["tool_calls"][0]
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "read")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {"path": "README.md"},
        )

    def test_kingogpt_legacy_call_json_maps_to_openai_tool_call(self):
        from kingogpt.tool_adapter import convert_kingogpt_json_to_openai_message

        message = convert_kingogpt_json_to_openai_message(
            '{"call":"exec","args":{"command":"pwd"}}'
        )

        tool_call = message["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "exec")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {"command": "pwd"},
        )

    def test_kingogpt_final_json_maps_to_assistant_content(self):
        from kingogpt.tool_adapter import convert_kingogpt_json_to_openai_message

        message = convert_kingogpt_json_to_openai_message(
            '{"type":"final","content":"done"}'
        )

        self.assertEqual(message, {"role": "assistant", "content": "done"})

    def test_kingogpt_fenced_tool_call_json_maps_to_openai_tool_call(self):
        from kingogpt.tool_adapter import convert_kingogpt_json_to_openai_message

        message = convert_kingogpt_json_to_openai_message(
            '```json\n{"type":"tool_call","name":"read","arguments":{"path":"README.md"}}\n```'
        )

        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read")

    def test_sanitize_tool_calls_strips_non_openai_fields(self):
        from kingogpt.tool_adapter import sanitize_openai_tool_calls

        sanitized = sanitize_openai_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "call_id": "extra",
                    "response_item_id": "extra",
                    "function": {
                        "name": "read",
                        "arguments": {"path": "README.md"},
                        "extra": "ignored",
                    },
                }
            ]
        )

        self.assertEqual(
            sanitized,
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ],
        )

    def test_raw_chat_completion_maps_model_json_tool_call_to_openai_tool_call(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        class FakeLLM:
            def __init__(self):
                self.system_prompt = None

            def complete(self, prompt, *, system_prompt=None):
                self.system_prompt = system_prompt
                return '{"type":"tool_call","name":"read","arguments":{"path":"README.md"}}'

        fake_llm = FakeLLM()
        original_llm = compat._raw_llm
        compat._raw_llm = fake_llm
        try:
            client = TestClient(compat.app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-web",
                    "messages": [{"role": "user", "content": "read the file"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            choice = payload["choices"][0]
            self.assertEqual(choice["finish_reason"], "tool_calls")
            self.assertIsNone(choice["message"]["content"])
            self.assertEqual(
                choice["message"]["tool_calls"][0]["function"]["name"],
                "read",
            )
            self.assertIn("AVAILABLE TOOLS", fake_llm.system_prompt)
        finally:
            compat._raw_llm = original_llm

    def test_raw_chat_completion_endpoint_with_mock_llm(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        class FakeLLM:
            def __init__(self):
                self.prompt = None
                self.system_prompt = None

            def complete(self, prompt, *, system_prompt=None):
                self.prompt = prompt
                self.system_prompt = system_prompt
                return "api ok"

        class FakeAgent:
            def __init__(self):
                self.calls = 0

            def run(self, task):
                self.calls += 1
                raise AssertionError("raw endpoint should not call agent")

        fake_llm = FakeLLM()
        fake_agent = FakeAgent()
        original_llm = compat._raw_llm
        original_agent = compat._agent
        compat._raw_llm = fake_llm
        compat._agent = fake_agent
        try:
            client = TestClient(compat.app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-web",
                    "messages": [
                        {"role": "system", "content": "be brief"},
                        {"role": "user", "content": "hello"},
                    ],
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["choices"][0]["message"]["content"], "api ok")
            self.assertEqual(
                fake_llm.prompt,
                "USER:\nhello\n\nASSISTANT:",
            )
            self.assertEqual(fake_llm.system_prompt, "be brief")
            self.assertEqual(fake_agent.calls, 0)
        finally:
            compat._raw_llm = original_llm
            compat._agent = original_agent

    def test_raw_chat_completion_trims_old_history(self):
        import internal_agent.server.openai_compat as compat

        messages = [{"role": "system", "content": "stay brief"}]
        for index in range(20):
            messages.append({"role": "user", "content": f"message {index}"})

        with mock.patch.dict(
            compat.os.environ,
            {
                "KINGOGPT_MAX_HISTORY_MESSAGES": "3",
                "KINGOGPT_MAX_PROMPT_CHARS": "1000",
            },
        ):
            prompt, system_prompt = compat.messages_to_prompt_and_system(messages)

        self.assertEqual(system_prompt, "stay brief")
        self.assertNotIn("message 16", prompt)
        self.assertIn("message 17", prompt)
        self.assertIn("message 18", prompt)
        self.assertIn("message 19", prompt)

    def test_agent_chat_completion_endpoint_with_mock_agent(self):
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
                return "agent ok"

        fake_agent = FakeAgent()
        original_agent = compat._agent
        compat._agent = fake_agent
        try:
            client = TestClient(compat.app)
            response = client.post(
                "/v1/agent/chat/completions",
                json={
                    "model": "kingogpt-web",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["choices"][0]["message"]["content"], "agent ok")
            self.assertEqual(fake_agent.task, "USER:\nhello\n\nASSISTANT:")
        finally:
            compat._agent = original_agent


class AzureWebLLMTests(unittest.TestCase):
    def test_system_prompt_is_embedded_in_query_not_instruction(self):
        import internal_agent.llm.azure_web_adapter as adapter

        captured = {}

        def fake_chat_via_api(
            token,
            user,
            prompt,
            args,
            *,
            instruction=None,
            chat_room_id=None,
            chat_thread_id=None,
        ):
            captured["prompt"] = prompt
            captured["instruction"] = instruction
            captured["chat_thread_id"] = chat_thread_id
            return "ok", 14, 99

        llm = adapter.AzureWebLLM(auto_delete_thread=False)
        with mock.patch.object(
            adapter.kingogpt,
            "load_or_refresh_token",
            return_value=({}, "token", {}, {"id": 1}),
        ), mock.patch.object(
            adapter.kingogpt,
            "read_session_prompt_state",
            return_value=None,
        ), mock.patch.object(
            adapter.kingogpt,
            "chat_via_api",
            side_effect=fake_chat_via_api,
        ), mock.patch.object(
            adapter.kingogpt,
            "write_session_prompt_state",
        ), mock.patch.object(
            adapter.kingogpt,
            "write_token_cache",
        ):
            self.assertEqual(
                llm.complete("USER:\nhello\n\nASSISTANT:", system_prompt="be brief"),
                "ok",
            )

        self.assertEqual(
            captured["prompt"],
            "SYSTEM:\nbe brief\n\nUSER:\nhello\n\nASSISTANT:",
        )
        self.assertIsNone(captured["instruction"])

    def test_system_prompt_is_sent_again_when_thread_is_not_reused(self):
        import internal_agent.llm.azure_web_adapter as adapter

        captured = {}

        def fake_chat_via_api(
            token,
            user,
            prompt,
            args,
            *,
            instruction=None,
            chat_room_id=None,
            chat_thread_id=None,
        ):
            captured["prompt"] = prompt
            captured["instruction"] = instruction
            captured["chat_thread_id"] = chat_thread_id
            return "ok", 14, 99

        llm = adapter.AzureWebLLM(auto_delete_thread=False, reuse_thread=False)
        with mock.patch.object(
            adapter.kingogpt,
            "load_or_refresh_token",
            return_value=({"session_prompt_state": {}}, "token", {}, {"id": 1}),
        ), mock.patch.object(
            adapter.kingogpt,
            "read_session_prompt_state",
            return_value={"chatRoomId": 14, "chatThreadId": 99},
        ), mock.patch.object(
            adapter.kingogpt,
            "chat_via_api",
            side_effect=fake_chat_via_api,
        ), mock.patch.object(
            adapter.kingogpt,
            "write_session_prompt_state",
        ), mock.patch.object(
            adapter.kingogpt,
            "write_token_cache",
        ):
            llm.complete("USER:\nhello\n\nASSISTANT:", system_prompt="be brief")

        self.assertEqual(
            captured["prompt"],
            "SYSTEM:\nbe brief\n\nUSER:\nhello\n\nASSISTANT:",
        )
        self.assertIsNone(captured["instruction"])
        self.assertIsNone(captured["chat_thread_id"])

    def test_models_endpoint(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        client = TestClient(compat.app)
        response = client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "kingogpt-web")

    def test_retrieve_model_endpoint(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        client = TestClient(compat.app)
        response = client.get("/v1/models/kingogpt-web")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["id"], "kingogpt-web")
        self.assertEqual(payload["object"], "model")

    def test_retrieve_unknown_model_returns_openai_style_error(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        client = TestClient(compat.app)
        response = client.get("/v1/models/unknown-model")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"]["error"]["code"],
            "model_not_found",
        )

    def test_responses_endpoint_with_mock_llm(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        class FakeLLM:
            def __init__(self):
                self.prompt = None
                self.system_prompt = None

            def complete(self, prompt, *, system_prompt=None):
                self.prompt = prompt
                self.system_prompt = system_prompt
                return "responses ok"

        fake_llm = FakeLLM()
        original_llm = compat._raw_llm
        compat._raw_llm = fake_llm
        try:
            client = TestClient(compat.app)
            response = client.post(
                "/v1/responses",
                json={
                    "model": "kingogpt-web",
                    "instructions": "be brief",
                    "input": "hello",
                    "tools": [{"type": "web_search"}],
                    "tool_choice": "auto",
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["object"], "response")
            self.assertEqual(payload["output_text"], "responses ok")
            self.assertEqual(fake_llm.prompt, "USER:\nhello\n\nASSISTANT:")
            self.assertEqual(fake_llm.system_prompt, "be brief")
        finally:
            compat._raw_llm = original_llm

    def test_responses_endpoint_accepts_input_text_blocks(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        class FakeLLM:
            def __init__(self):
                self.prompt = None

            def complete(self, prompt, *, system_prompt=None):
                self.prompt = prompt
                return "block ok"

        fake_llm = FakeLLM()
        original_llm = compat._raw_llm
        compat._raw_llm = fake_llm
        try:
            client = TestClient(compat.app)
            response = client.post(
                "/v1/responses",
                json={
                    "model": "kingogpt-web",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "hello from hermes"},
                            ],
                        },
                    ],
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["output_text"], "block ok")
            self.assertIn("hello from hermes", fake_llm.prompt)
            self.assertEqual(payload["usage"]["output_tokens"], 2)
        finally:
            compat._raw_llm = original_llm

    def test_raw_chat_completion_streaming_smoke(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        class FakeLLM:
            def complete(self, prompt, *, system_prompt=None):
                return "stream ok"

        original_llm = compat._raw_llm
        compat._raw_llm = FakeLLM()
        try:
            client = TestClient(compat.app)
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-web",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ) as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                body = "".join(response.iter_text())

            data_lines = [
                line.removeprefix("data: ")
                for line in body.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            chunks = [json.loads(line) for line in data_lines]
            self.assertEqual(
                chunks[0]["choices"][0]["delta"],
                {"role": "assistant"},
            )
            self.assertEqual(
                chunks[1]["choices"][0]["delta"],
                {"content": "stream ok"},
            )
            self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "stop")
            self.assertIn("data: [DONE]", body)
        finally:
            compat._raw_llm = original_llm

    def test_raw_chat_completion_streaming_can_include_usage(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        class FakeLLM:
            def complete(self, prompt, *, system_prompt=None):
                return "stream usage ok"

        original_llm = compat._raw_llm
        compat._raw_llm = FakeLLM()
        try:
            client = TestClient(compat.app)
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-web",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ) as response:
                self.assertEqual(response.status_code, 200)
                body = "".join(response.iter_text())

            data_lines = [
                line.removeprefix("data: ")
                for line in body.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            chunks = [json.loads(line) for line in data_lines]
            usage_chunks = [chunk for chunk in chunks if chunk.get("usage")]
            self.assertEqual(usage_chunks[0]["choices"], [])
            self.assertGreater(usage_chunks[0]["usage"]["total_tokens"], 0)
        finally:
            compat._raw_llm = original_llm

    def test_raw_chat_completion_streams_tool_call(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        client = TestClient(compat.app)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "kingogpt-web",
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": "Use your available tools to inspect the current working directory.",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_files",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            body = "".join(response.iter_text())

        data_lines = [
            line.removeprefix("data: ")
            for line in body.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        chunks = [json.loads(line) for line in data_lines]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        tool_call_delta = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tool_call_delta["function"]["name"], "search_files")

    def test_raw_chat_completion_accepts_extra_openai_fields(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        class FakeLLM:
            def complete(self, prompt, *, system_prompt=None):
                return "extra ok"

        original_llm = compat._raw_llm
        compat._raw_llm = FakeLLM()
        try:
            client = TestClient(compat.app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kingogpt-web",
                    "messages": [{"role": "user", "content": "hello"}],
                    "top_p": 0.9,
                    "stop": ["END"],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "presence_penalty": 0.1,
                    "frequency_penalty": 0.2,
                    "user": "hermes-local",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["choices"][0]["message"]["content"],
                "extra ok",
            )
        finally:
            compat._raw_llm = original_llm

    def test_raw_chat_completion_returns_tool_call_for_directory_inspection(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        client = TestClient(compat.app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kingogpt-web",
                "messages": [
                    {
                        "role": "user",
                        "content": "Use your available tools to inspect the current working directory.",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_files",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        tool_call = choice["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "search_files")
        self.assertIn('"target": "files"', tool_call["function"]["arguments"])

    def test_raw_chat_completion_finishes_tool_smoke_after_tool_result(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        import internal_agent.server.openai_compat as compat

        client = TestClient(compat.app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "kingogpt-web",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Use your available tools to inspect the current working "
                            "directory, then answer with TOOL_SMOKE_OK."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_kingogpt_1",
                                "type": "function",
                                "function": {
                                    "name": "search_files",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_kingogpt_1",
                        "content": "scripts/hermes_tool_smoke.py\nDockerfile",
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "TOOL_SMOKE_OK hermes_tool_smoke.py",
        )

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
        self.assertEqual(
            response.json()["service"],
            "kingogpt-openai-compatible",
        )


if __name__ == "__main__":
    unittest.main()
