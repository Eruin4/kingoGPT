"""Tests for kingogpt.tool_adapter — the critical parsing and heuristic layer."""

import json
import unittest

from kingogpt.tool_adapter import (
    _content_to_text,
    _extract_json_object,
    convert_kingogpt_json_to_openai_message,
    example_argument_value,
    example_arguments,
    infer_tool_call_from_failed_response,
    render_available_tools,
    render_tool_contract,
    sanitize_openai_tool_calls,
)

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "target": {"type": "string"},
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
]


class ConvertKingogptJsonTests(unittest.TestCase):
    """Test convert_kingogpt_json_to_openai_message with various model outputs."""

    def test_final_type_returns_content(self):
        raw = json.dumps({"type": "final", "content": "42"})
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "42")

    def test_tool_call_type_returns_tool_calls(self):
        raw = json.dumps({"type": "tool_call", "name": "terminal", "arguments": {"command": "pwd"}})
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertEqual(msg["role"], "assistant")
        self.assertIsNotNone(msg.get("tool_calls"))
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "terminal")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args["command"], "pwd")

    def test_function_type_alternative(self):
        raw = json.dumps({"type": "function", "name": "read_file", "arguments": {"path": "foo.py"}})
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertIsNotNone(msg.get("tool_calls"))
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "read_file")

    def test_call_alternative(self):
        raw = json.dumps({"call": "terminal", "args": {"command": "ls"}})
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertIsNotNone(msg.get("tool_calls"))
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "terminal")

    def test_function_dict_alternative(self):
        raw = json.dumps({"function": {"name": "terminal", "arguments": {"command": "ls"}}})
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertIsNotNone(msg.get("tool_calls"))

    def test_tool_key_alternative(self):
        raw = json.dumps({"tool": "terminal", "arguments": {"command": "ls"}})
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertIsNotNone(msg.get("tool_calls"))

    def test_reply_key_returns_content(self):
        raw = json.dumps({"reply": "Hello, world!"})
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertEqual(msg["content"], "Hello, world!")

    def test_plain_text_returns_as_content(self):
        raw = "This is just a plain text response."
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], raw)

    def test_code_fenced_json(self):
        raw = '```json\n{"type": "final", "content": "done"}\n```'
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertEqual(msg["content"], "done")

    def test_missing_tool_name_returns_error_content(self):
        raw = json.dumps({"type": "tool_call", "name": "", "arguments": {}})
        msg = convert_kingogpt_json_to_openai_message(raw)
        self.assertIn("error", msg.get("content", "").lower())
        self.assertIsNone(msg.get("tool_calls"))


class ExtractJsonObjectTests(unittest.TestCase):
    """Test _extract_json_object with various malformed inputs."""

    def test_clean_json(self):
        result = _extract_json_object('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_json_with_surrounding_text(self):
        result = _extract_json_object('some text {"key": "value"} more text')
        self.assertEqual(result, {"key": "value"})

    def test_code_fenced_json(self):
        result = _extract_json_object('```json\n{"key": "value"}\n```')
        self.assertEqual(result, {"key": "value"})

    def test_no_json(self):
        result = _extract_json_object("no json here")
        self.assertIsNone(result)

    def test_array_returns_none(self):
        result = _extract_json_object("[1, 2, 3]")
        self.assertIsNone(result)

    def test_empty_string(self):
        result = _extract_json_object("")
        self.assertIsNone(result)

    def test_nested_json(self):
        result = _extract_json_object('prefix {"a": {"b": 1}} suffix')
        self.assertIsNotNone(result)
        self.assertEqual(result["a"]["b"], 1)


class InferToolCallTests(unittest.TestCase):
    """Test infer_tool_call_from_failed_response heuristics."""

    def test_bash_code_block_extracts_command(self):
        raw = 'Here is how to do it:\n```bash\nls -la /home\n```'
        result = infer_tool_call_from_failed_response(raw, SAMPLE_TOOLS, "list files")
        self.assertIsNotNone(result)
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "terminal")
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["command"], "ls -la /home")

    def test_os_listdir_maps_to_ls(self):
        raw = '```python\nimport os\nos.listdir(".")\n```'
        result = infer_tool_call_from_failed_response(raw, SAMPLE_TOOLS, "list files")
        self.assertIsNotNone(result)
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "terminal")

    def test_user_asks_to_list_files(self):
        raw = "I'll show you the files in the directory."
        result = infer_tool_call_from_failed_response(raw, SAMPLE_TOOLS, "What files are in this directory?")
        self.assertIsNotNone(result)
        func_name = result["tool_calls"][0]["function"]["name"]
        self.assertIn(func_name, ("search_files", "terminal"))

    def test_korean_file_listing(self):
        raw = "파일을 확인해보겠습니다."
        result = infer_tool_call_from_failed_response(raw, SAMPLE_TOOLS, "파일 목록을 보여줘")
        self.assertIsNotNone(result)

    def test_user_asks_to_read_file(self):
        raw = "Let me read that file for you."
        result = infer_tool_call_from_failed_response(raw, SAMPLE_TOOLS, "read foo.py")
        self.assertIsNotNone(result)
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "read_file")
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["path"], "foo.py")

    def test_no_tools_returns_none(self):
        result = infer_tool_call_from_failed_response("anything", [], "hello")
        self.assertIsNone(result)

    def test_no_match_returns_none(self):
        raw = "The answer to your question is 42."
        result = infer_tool_call_from_failed_response(raw, SAMPLE_TOOLS, "what is the meaning of life?")
        self.assertIsNone(result)


class RenderToolContractTests(unittest.TestCase):
    """Test render_tool_contract and render_available_tools output."""

    def test_contract_includes_strict_instructions(self):
        contract = render_tool_contract(SAMPLE_TOOLS)
        self.assertIn("CRITICAL INSTRUCTION", contract)
        self.assertIn("tool_call", contract)

    def test_contract_lists_all_tools(self):
        contract = render_tool_contract(SAMPLE_TOOLS)
        self.assertIn("terminal", contract)
        self.assertIn("read_file", contract)
        self.assertIn("search_files", contract)

    def test_render_available_tools_format(self):
        rendered = render_available_tools(SAMPLE_TOOLS)
        self.assertIn("- terminal:", rendered)
        self.assertIn("- read_file:", rendered)
        self.assertIn("Respond with ONLY a JSON object", rendered)

    def test_render_available_tools_empty(self):
        rendered = render_available_tools([])
        self.assertIn("Respond with ONLY a JSON object", rendered)

    def test_render_available_tools_skips_non_function(self):
        tools = [{"type": "not_function"}]
        rendered = render_available_tools(tools)
        # Should only contain the footer instruction, not any tool entries
        lines = [line for line in rendered.splitlines() if line.startswith("- ")]
        self.assertEqual(len(lines), 0)


class SanitizeOpenAIToolCallsTests(unittest.TestCase):
    """Test sanitize_openai_tool_calls normalization."""

    def test_assigns_ids(self):
        tool_calls = [{"function": {"name": "terminal", "arguments": '{"command":"ls"}'}}]
        sanitized = sanitize_openai_tool_calls(tool_calls)
        self.assertEqual(len(sanitized), 1)
        self.assertTrue(sanitized[0]["id"])
        self.assertEqual(sanitized[0]["type"], "function")

    def test_dict_arguments_serialized(self):
        tool_calls = [{"function": {"name": "terminal", "arguments": {"command": "ls"}}}]
        sanitized = sanitize_openai_tool_calls(tool_calls)
        args_str = sanitized[0]["function"]["arguments"]
        self.assertIsInstance(args_str, str)
        self.assertEqual(json.loads(args_str), {"command": "ls"})

    def test_none_arguments_become_empty_object(self):
        tool_calls = [{"function": {"name": "terminal", "arguments": None}}]
        sanitized = sanitize_openai_tool_calls(tool_calls)
        self.assertEqual(sanitized[0]["function"]["arguments"], "{}")

    def test_preserves_existing_id(self):
        tool_calls = [{"id": "my_id", "function": {"name": "terminal", "arguments": "{}"}}]
        sanitized = sanitize_openai_tool_calls(tool_calls)
        self.assertEqual(sanitized[0]["id"], "my_id")


class ContentToTextTests(unittest.TestCase):
    """Test _content_to_text with various content shapes."""

    def test_string(self):
        self.assertEqual(_content_to_text("hello"), "hello")

    def test_none(self):
        self.assertEqual(_content_to_text(None), "")

    def test_dict_with_text(self):
        self.assertEqual(_content_to_text({"text": "hello"}), "hello")

    def test_dict_with_input_text(self):
        self.assertEqual(_content_to_text({"input_text": "hello"}), "hello")

    def test_list_of_strings(self):
        self.assertEqual(_content_to_text(["a", "b"]), "a\nb")

    def test_list_of_dicts(self):
        result = _content_to_text([{"text": "a"}, {"text": "b"}])
        self.assertEqual(result, "a\nb")

    def test_image_type(self):
        result = _content_to_text([{"type": "input_image"}])
        self.assertEqual(result, "[image]")


class ExampleArgumentsTests(unittest.TestCase):
    """Test example_arguments and example_argument_value."""

    def test_command_argument(self):
        self.assertEqual(example_argument_value("command", {}), "pwd")

    def test_path_argument(self):
        self.assertEqual(example_argument_value("path", {}), ".")

    def test_boolean_type(self):
        self.assertEqual(example_argument_value("flag", {"type": "boolean"}), False)

    def test_integer_type(self):
        self.assertEqual(example_argument_value("count", {"type": "integer"}), 1)

    def test_timeout_integer(self):
        self.assertEqual(example_argument_value("timeout", {"type": "integer"}), 30)

    def test_enum_picks_first(self):
        self.assertEqual(example_argument_value("mode", {"enum": ["fast", "slow"]}), "fast")

    def test_example_arguments_uses_required(self):
        params = {
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        }
        result = example_arguments(params)
        self.assertIn("a", result)

    def test_example_arguments_empty(self):
        result = example_arguments({})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
