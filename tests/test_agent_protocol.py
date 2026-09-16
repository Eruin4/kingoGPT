import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from kingogpt.agent import Workspace
from kingogpt.openai_server import ServerSettings, create_app
from kingogpt.tool_protocol import ToolProtocolError, decision_prompt, parse_decision


TOOL = {"type": "function", "function": {
    "name": "lookup", "description": "Look up a record.",
    "parameters": {"type": "object", "properties": {
        "record": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"], "additionalProperties": False},
    }, "required": ["record"], "additionalProperties": False},
}}


class SequenceProvider:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.prompts = []

    def complete(self, prompt, instruction=None, on_chunk=None):
        self.prompts.append(prompt)
        answer = next(self.answers)
        if on_chunk:
            on_chunk(answer)
        return answer


class DecisionTests(unittest.TestCase):
    def test_tagged_decision_preserves_arguments_and_maps_name(self):
        raw = '다음 작업입니다.\n<decision>{"operation":"op_1","parameters":{"record":{"id":19}}}</decision>'
        call = parse_decision(raw, [TOOL])["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "lookup")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"record": {"id": 19}})

    def test_prose_examples_are_not_executed(self):
        for raw in [
            'You could try {"operation":"op_1","parameters":{"record":{"id":19}}}',
            'Use lookup(record=19)',
            '<decision>{"operation":"finish","answer":"a"}</decision><decision>{"operation":"finish","answer":"b"}</decision>',
        ]:
            with self.subTest(raw=raw), self.assertRaises(ToolProtocolError):
                parse_decision(raw, [TOOL])

    def test_invalid_arguments_are_not_coerced(self):
        for parameters in [{}, {"record": {"id": "19"}}, {"record": {"id": 19, "extra": 1}}, [], None, "bad json"]:
            with self.subTest(parameters=parameters), self.assertRaises(ToolProtocolError):
                parse_decision(json.dumps({"operation": "op_1", "parameters": parameters}), [TOOL])

    def test_unknown_operation_and_function_rejected(self):
        for obj in [{"operation": "op_99", "parameters": {}}, {"type": "tool_call", "name": "shell", "arguments": {}}]:
            with self.assertRaises(ToolProtocolError):
                parse_decision(json.dumps(obj), [TOOL])

    def test_tags_in_answer_are_literal_data(self):
        raw = json.dumps({"operation": "finish", "answer": "Use <decision>example</decision> in a document."})
        self.assertEqual(parse_decision(raw, [TOOL])["content"], "Use <decision>example</decision> in a document.")

    def test_non_json_numbers_are_rejected(self):
        with self.assertRaises(ToolProtocolError):
            parse_decision('{"operation":"op_1","parameters":{"record":{"id":NaN}}}', [TOOL])

    def test_full_schema_and_assistant_calls_survive_history(self):
        prompt = decision_prompt([
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": "Looking it up", "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": '{"record":{"id":19}}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "unique result"},
        ], [TOOL], None)
        self.assertIn('"additionalProperties": false', prompt)
        self.assertIn('operations_requested', prompt)
        self.assertIn('unique result', prompt)
        self.assertIn('call_1', prompt)
        self.assertIn('application function lookup', prompt)


class APIDecisionTests(unittest.TestCase):
    def request(self, answers, **overrides):
        provider = SequenceProvider(answers)
        body = {"model": "kingogpt", "messages": [{"role": "user", "content": "look up record 19"}], "tools": [TOOL], **overrides}
        with TestClient(create_app(settings=ServerSettings(tool_attempts=2), provider=provider)) as client:
            response = client.post("/v1/chat/completions", json=body)
        return response, provider

    def test_invalid_decision_retries_then_returns_real_valid_call(self):
        response, provider = self.request([
            '{"operation":"op_1","parameters":{"record":{"id":"19"}}}',
            '{"operation":"op_1","parameters":{"record":{"id":19}}}',
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(provider.prompts), 2)
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("Previous response (quoted data)", provider.prompts[1])

    def test_final_metadata_removed_without_rewriting_embedded_comments(self):
        answer = "Example: <!-- tools: keep --> inside text.\nDone.\n<!-- tools: none -->"
        response, _ = self.request([json.dumps({"operation": "finish", "answer": answer})])
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "Example: <!-- tools: keep --> inside text.\nDone.")

    def test_failed_decisions_fail_closed_even_in_auto_mode(self):
        response, provider = self.request(["I cannot use that tool.", '"failed to execute @list_sort"'])
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "tool_call_failed")
        self.assertEqual(len(provider.prompts), 2)

    def test_required_and_named_tool_choice_do_not_accept_final_answer(self):
        for choice in ["required", {"type": "function", "function": {"name": "lookup"}}]:
            with self.subTest(choice=choice):
                response, _ = self.request(['{"operation":"finish","answer":"done"}'] * 2, tool_choice=choice)
                self.assertEqual(response.status_code, 502)

    def test_none_does_not_decode_answer_as_a_call(self):
        response, _ = self.request(['{"operation":"op_1","parameters":{"record":{"id":19}}}'], tool_choice="none")
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "stop")

    def test_parallel_disabled_rejects_multiple_calls(self):
        raw = json.dumps({"tool_calls": [{"function": {"name": "lookup", "arguments": {"record": {"id": 19}}}}] * 2})
        response, _ = self.request([raw, raw], parallel_tool_calls=False)
        self.assertEqual(response.status_code, 502)

    def test_external_schema_reference_is_rejected_before_model_call(self):
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"$ref": "https://example.invalid/schema"}}}
        response, provider = self.request([], tools=[tool])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(provider.prompts, [])

    def test_stream_error_preserves_tool_failure_code(self):
        response, _ = self.request(["invalid"] * 2, stream=True)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertEqual(events[-1]["error"]["code"], "tool_call_failed")

    def test_metadata_cleaning_does_not_rewrite_function_arguments(self):
        tool = {"type": "function", "function": {"name": "write", "parameters": {
            "type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"],
        }}}
        content = '<!-- tools: none -->\n<div>keep this source unchanged</div>'
        raw = json.dumps({"operation": "op_1", "parameters": {"content": content}}) + '\n<!-- tools: none -->'
        response, _ = self.request([raw], tools=[tool])
        arguments = json.loads(response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["content"], content)

    def test_previous_response_id_is_not_silently_ignored(self):
        provider = SequenceProvider([])
        with TestClient(create_app(provider=provider)) as client:
            response = client.post("/v1/responses", json={"model": "kingogpt", "input": "continue", "previous_response_id": "resp_old"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(provider.prompts, [])


class WorkspaceTests(unittest.TestCase):
    def test_read_write_and_permissions(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root, allow_write=True)
            workspace.write_file("nested/a.txt", "안녕하세요")
            self.assertEqual(workspace.read_file("nested/a.txt"), "안녕하세요")
            with self.assertRaises(PermissionError):
                Workspace(root).write_file("nested/a.txt", "changed")
            self.assertEqual(workspace.read_file("nested/a.txt"), "안녕하세요")

    def test_escape_symlink_and_state_are_denied(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            workspace = Workspace(root, allow_write=True)
            Path(root, "escape").symlink_to(outside, target_is_directory=True)
            for path in ["../out.txt", outside + "/out.txt", "escape/out.txt", "state/kingogpt_config.json", ".env", ".git/config"]:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    workspace.write_file(path, "must not write")
            self.assertEqual(list(Path(outside).iterdir()), [])
