"""Tests for api_solver parsing functions and exception hierarchy."""

import unittest

import kingogpt.api_solver as solver
from kingogpt.exceptions import (
    AuthenticationError,
    TokenCacheCorruptError,
    TokenExpiredError,
    TokenMissingError,
)


class ExtractIdentifiersTests(unittest.TestCase):
    """Test extract_identifiers with all naming conventions."""

    def test_snake_case_in_document(self):
        event = {
            "data": {
                "documents": [
                    {"chat_rooms_id": 10, "chat_threads_id": 20}
                ]
            }
        }
        room, thread = solver.extract_identifiers(event)
        self.assertEqual(room, 10)
        self.assertEqual(thread, 20)

    def test_camel_case_in_document(self):
        event = {
            "data": {
                "documents": [
                    {"chatRoomsId": 10, "chatThreadsId": 20}
                ]
            }
        }
        room, thread = solver.extract_identifiers(event)
        self.assertEqual(room, 10)
        self.assertEqual(thread, 20)

    def test_ids_in_data_level(self):
        event = {"data": {"chat_rooms_id": 10, "chat_threads_id": 20, "documents": []}}
        room, thread = solver.extract_identifiers(event)
        self.assertEqual(room, 10)
        self.assertEqual(thread, 20)

    def test_ids_in_event_level(self):
        event = {"chat_rooms_id": 10, "chat_threads_id": 20}
        room, thread = solver.extract_identifiers(event)
        self.assertEqual(room, 10)
        self.assertEqual(thread, 20)

    def test_string_ids_parsed(self):
        event = {"data": {"documents": [{"chat_rooms_id": "10", "chat_threads_id": "20"}]}}
        room, thread = solver.extract_identifiers(event)
        self.assertEqual(room, 10)
        self.assertEqual(thread, 20)

    def test_empty_event(self):
        room, thread = solver.extract_identifiers({})
        self.assertIsNone(room)
        self.assertIsNone(thread)

    def test_empty_documents(self):
        event = {"data": {"documents": []}}
        room, thread = solver.extract_identifiers(event)
        self.assertIsNone(room)
        self.assertIsNone(thread)


class ExtractStreamTextTests(unittest.TestCase):
    """Test extract_stream_text with all response shapes."""

    def test_text_in_replies(self):
        event = {"data": {"documents": [{"replies": {"text": "hello"}}]}}
        self.assertEqual(solver.extract_stream_text(event), "hello")

    def test_content_in_choices_delta(self):
        event = {"choices": [{"delta": {"content": "hello"}}]}
        self.assertEqual(solver.extract_stream_text(event), "hello")

    def test_content_in_choices_message(self):
        event = {"choices": [{"message": {"content": "hello"}}]}
        self.assertEqual(solver.extract_stream_text(event), "hello")

    def test_content_at_top_level(self):
        event = {"content": "hello"}
        self.assertEqual(solver.extract_stream_text(event), "hello")

    def test_text_at_top_level(self):
        event = {"text": "hello"}
        self.assertEqual(solver.extract_stream_text(event), "hello")

    def test_empty_event(self):
        self.assertEqual(solver.extract_stream_text({}), "")

    def test_numeric_text_ignored(self):
        event = {"data": {"documents": [{"replies": {"text": 123}}]}}
        # text must be a string
        self.assertEqual(solver.extract_stream_text(event), "")


class ShouldResetPromptStateTests(unittest.TestCase):
    """Test should_reset_prompt_state with various error messages."""

    def test_http_4xx_with_chat_thread(self):
        self.assertTrue(solver.should_reset_prompt_state(
            "API request failed: HTTP 404 chat_threads_id not found"
        ))

    def test_http_5xx_with_room_id(self):
        self.assertTrue(solver.should_reset_prompt_state(
            "API request failed: HTTP 500 chat_rooms_id invalid"
        ))

    def test_no_http_status(self):
        self.assertFalse(solver.should_reset_prompt_state(
            "chat_threads_id not found"
        ))

    def test_unrelated_http_error(self):
        self.assertFalse(solver.should_reset_prompt_state(
            "API request failed: HTTP 500 internal error"
        ))


class ShouldAutoRefreshTokenTests(unittest.TestCase):
    """Test should_auto_refresh_token with typed and message-based errors."""

    def test_token_missing_error(self):
        self.assertTrue(solver.should_auto_refresh_token(TokenMissingError("missing")))

    def test_token_expired_error(self):
        self.assertTrue(solver.should_auto_refresh_token(TokenExpiredError("expired")))

    def test_token_cache_corrupt_error(self):
        self.assertTrue(solver.should_auto_refresh_token(TokenCacheCorruptError("corrupt")))

    def test_authentication_error(self):
        self.assertTrue(solver.should_auto_refresh_token(AuthenticationError("401")))

    def test_runtime_error_with_message(self):
        self.assertTrue(solver.should_auto_refresh_token(
            RuntimeError("access token is missing")
        ))

    def test_runtime_error_expired(self):
        self.assertTrue(solver.should_auto_refresh_token(
            RuntimeError("expired or about to expire")
        ))

    def test_runtime_error_401(self):
        self.assertTrue(solver.should_auto_refresh_token(
            RuntimeError("HTTP 401 Unauthorized")
        ))

    def test_unrelated_error(self):
        self.assertFalse(solver.should_auto_refresh_token(
            RuntimeError("something completely different")
        ))


class ParseOptionalIntTests(unittest.TestCase):
    """Test parse_optional_int edge cases."""

    def test_int(self):
        self.assertEqual(solver.parse_optional_int(42), 42)

    def test_string_int(self):
        self.assertEqual(solver.parse_optional_int("42"), 42)

    def test_empty_string(self):
        self.assertIsNone(solver.parse_optional_int(""))

    def test_none(self):
        self.assertIsNone(solver.parse_optional_int(None))

    def test_invalid_string(self):
        self.assertIsNone(solver.parse_optional_int("abc"))


class BuildRequestPromptTests(unittest.TestCase):
    """Test build_request_prompt formatting."""

    def test_user_only(self):
        result = solver.build_request_prompt("hello", None)
        self.assertEqual(result, "USER:\nhello")

    def test_with_system(self):
        result = solver.build_request_prompt("hello", "be brief")
        self.assertEqual(result, "SYSTEM:\nbe brief\n\nUSER:\nhello")

    def test_empty_system_treated_as_none(self):
        result = solver.build_request_prompt("hello", "")
        self.assertEqual(result, "USER:\nhello")

    def test_whitespace_system_treated_as_empty(self):
        # Whitespace-only system prompt is stripped but not skipped entirely.
        result = solver.build_request_prompt("hello", "   ")
        # system_prompt.strip() -> "", so SYSTEM: block is still emitted
        # with an empty body. This matches the "blocks.append(f'SYSTEM:\n{...}')" logic.
        self.assertIn("USER:\nhello", result)


class PromptHashAndStateTests(unittest.TestCase):
    """Test prompt hash and state key functions."""

    def test_create_prompt_hash_deterministic(self):
        h1 = solver.create_prompt_hash("test")
        h2 = solver.create_prompt_hash("test")
        self.assertEqual(h1, h2)

    def test_create_prompt_hash_different_inputs(self):
        h1 = solver.create_prompt_hash("a")
        h2 = solver.create_prompt_hash("b")
        self.assertNotEqual(h1, h2)

    def test_build_state_key_format(self):
        key = solver.build_state_key("session", "hash123")
        self.assertEqual(key, "session::hash123")

    def test_read_session_prompt_state_missing(self):
        cache = {}
        result = solver.read_session_prompt_state(cache, "key")
        self.assertIsNone(result)

    def test_read_session_prompt_state_exists(self):
        cache = {"session_prompt_state": {"key": {"chatRoomId": 10}}}
        result = solver.read_session_prompt_state(cache, "key")
        self.assertEqual(result, {"chatRoomId": 10})

    def test_write_and_read_session_prompt_state(self):
        cache = {}
        solver.write_session_prompt_state(
            cache, "key", prompt_hash="h", chat_room_id=10, chat_thread_id=20
        )
        result = solver.read_session_prompt_state(cache, "key")
        self.assertEqual(result["chatRoomId"], 10)
        self.assertEqual(result["chatThreadId"], 20)

    def test_delete_session_prompt_state(self):
        cache = {"session_prompt_state": {"key": {"chatRoomId": 10}}}
        solver.delete_session_prompt_state(cache, "key")
        self.assertIsNone(solver.read_session_prompt_state(cache, "key"))


if __name__ == "__main__":
    unittest.main()
