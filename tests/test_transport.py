import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kingogpt import api_solver, shared
from kingogpt.exceptions import AuthenticationError, BackendError, TokenCacheCorruptError
from kingogpt.openai_server import KingoGPTProvider, ServerSettings
from kingogpt.sse import iter_events


def frame(text=None, **fields):
    doc = {**fields}
    if text is not None:
        doc["replies"] = {"text": text}
    return ["data: " + json.dumps({"code": "200", "data": {"documents": [doc]}}, ensure_ascii=False), ""]


class StreamTests(unittest.TestCase):
    def invoke(self, lines, callback=None, status=200, content_type="text/event-stream"):
        response = Mock(status_code=status, headers={"Content-Type": content_type})
        response.iter_lines.return_value = iter(lines)
        self.response = response
        with patch.object(api_solver.requests, "post", return_value=response):
            return api_solver.chat_via_api("synthetic-token", {"id": 1}, "test",
                                          argparse.Namespace(scenario_id="test", request_timeout=2),
                                          on_chunk=callback, verbose=False)

    def test_korean_deltas_status_and_native_completion(self):
        chunks = []
        result = self.invoke(frame(chat_threads_id=8) + frame("IGNORE", event="check") +
                             frame("안녕 ") + frame("세상", is_sse_finished=True, finish_reason="stop"), chunks.append)
        self.assertEqual(result, ("안녕 세상", None, 8))
        self.assertEqual(chunks, ["안녕 ", "세상"])
        self.assertEqual(self.response.encoding, "utf-8")
        self.response.close.assert_called_once()

    def test_partial_reply_and_length_stop_are_not_success(self):
        for lines in [frame("partial"), frame("partial", is_sse_finished=True, finish_reason="length")]:
            with self.assertRaises(BackendError):
                self.invoke(lines)
            self.response.close.assert_called_once()

    def test_errors_close_response_and_are_not_answers(self):
        for code, error in [("00001", AuthenticationError), ("500", BackendError)]:
            with self.assertRaises(error):
                self.invoke(["data: " + json.dumps({"code": code, "text": "not an answer"}), ""])
            self.response.close.assert_called_once()
        with self.assertRaises(AuthenticationError):
            self.invoke([], status=401)
        self.response.close.assert_called_once()
        with self.assertRaises(BackendError):
            self.invoke([], content_type="text/html")
        self.response.close.assert_called_once()

    def test_multiline_and_comments(self):
        events = list(iter_events([b":heartbeat", b"event: message", b'data: {"text":',
                                   'data: "한글"}'.encode(), b"", b"data: [DONE]"]))
        self.assertEqual(json.loads(events[0][1]), {"text": "한글"})
        self.assertEqual(events[1][1], "[DONE]")

    def test_bad_json_and_oversized_events_fail(self):
        with self.assertRaises(BackendError):
            self.invoke(["data: invalid", ""])
        with self.assertRaises(BackendError):
            list(iter_events(["data: abcdef"], max_event_chars=4))


class CacheTests(unittest.TestCase):
    def test_refresh_uses_web_contract_and_rejects_invalid_tokens(self):
        from unittest.mock import MagicMock
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.json.return_value = {"accessToken": "new", "refreshToken": "rotated"}
        cache = {"access_token": "old", "refresh_token": "refresh", "chat_room_id": 14}
        with patch.object(shared.requests, "post", return_value=response) as post, \
             patch.object(shared, "decode_jwt_payload", return_value={"exp": 9999999999}):
            updated = shared.refresh_cached_tokens(cache)
        self.assertEqual(post.call_args.kwargs["json"], {"accessToken": "old", "refreshToken": "refresh"})
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(updated["refresh_token"], "rotated")
        self.assertEqual(updated["chat_room_id"], 14)
        self.assertEqual(cache["access_token"], "old")
        response.json.return_value = {"accessToken": "new"}
        with patch.object(shared.requests, "post", return_value=response):
            with self.assertRaises(AuthenticationError):
                shared.refresh_cached_tokens(cache)

    def test_atomic_private_cache_preserves_original_on_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "cache.json")
            shared.write_token_cache(str(path), {"access_token": "old"})
            with patch.object(shared.os, "replace", side_effect=OSError("disk failed")):
                with self.assertRaises(OSError):
                    shared.write_token_cache(str(path), {"access_token": "new"})
            self.assertEqual(shared.load_token_cache(str(path))["access_token"], "old")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(root).iterdir()), [path])
            path.write_text("[]")
            with self.assertRaises(TokenCacheCorruptError):
                shared.load_token_cache(str(path))

    def test_http_refresh_avoids_browser_and_preserves_session_state(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root, "cache.json"))
            cache = {"access_token": "old", "refresh_token": "refresh", "session_prompt_state": {"keep": {}}}
            shared.write_token_cache(path, cache)
            updated = {**cache, "access_token": "new"}
            with patch.object(api_solver, "refresh_cached_tokens", return_value=updated), \
                 patch("kingogpt.token_capture.refresh_token_cache") as browser:
                result = api_solver.refresh_token_cache(argparse.Namespace(no_auto_refresh_token=False, token_cache=path))
            browser.assert_not_called()
            self.assertEqual(result, updated)
            self.assertEqual(shared.load_token_cache(path), updated)


class ProviderTests(unittest.TestCase):
    def test_profile_reused_but_threads_are_not(self):
        provider = KingoGPTProvider(ServerSettings())
        state = ({}, "token", {}, {"id": 1})
        with patch.object(api_solver, "load_or_refresh_token", return_value=state) as load, \
             patch.object(api_solver, "ensure_token_is_fresh"), \
             patch.object(api_solver, "chat_via_api", return_value=("ok", 14, 99)) as chat:
            self.assertEqual(provider.complete("one"), "ok")
            self.assertEqual(provider.complete("two"), "ok")
        load.assert_called_once()
        self.assertTrue(all(call.kwargs["chat_thread_id"] is None for call in chat.call_args_list))

    def test_no_retry_after_emitting_partial_text(self):
        provider = KingoGPTProvider(ServerSettings())
        def fail(*args, **kwargs):
            kwargs["on_chunk"]("partial")
            raise AuthenticationError("expired")
        with patch.object(provider, "_credentials", return_value=({}, "token", {}, {"id": 1})) as auth, \
             patch.object(api_solver, "chat_via_api", side_effect=fail) as chat:
            with self.assertRaises(AuthenticationError):
                provider.complete("one")
        auth.assert_called_once()
        chat.assert_called_once()
