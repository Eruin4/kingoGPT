import unittest

try:
    from openai import AsyncOpenAI
    try:
        import httpx2
    except ImportError:
        import httpx as httpx2
except ImportError:  # OpenAI is a client-side integration-test dependency.
    httpx2 = None
    AsyncOpenAI = None

from kingogpt.openai_server import ServerSettings, create_app


class FakeProvider:
    def __init__(self, answer, chunks=None):
        self.answer = answer
        self.chunks = chunks or [answer]

    def complete(self, prompt, instruction=None, on_chunk=None):
        if on_chunk:
            for chunk in self.chunks:
                on_chunk(chunk)
        return self.answer


@unittest.skipIf(AsyncOpenAI is None, "OpenAI SDK is not installed")
class OpenAISDKCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, provider):
        app = create_app(
            settings=ServerSettings(model_id="kingogpt-test", api_key="test-key"),
            provider=provider,
        )
        http_client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app))
        client = AsyncOpenAI(
            base_url="http://test/v1",
            api_key="test-key",
            http_client=http_client,
        )
        self.addAsyncCleanup(client.close)
        return client

    async def test_chat_completion_parses_in_openai_sdk(self):
        client = await self.make_client(FakeProvider("sdk answer"))
        completion = await client.chat.completions.create(
            model="kingogpt-test",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(completion.object, "chat.completion")
        self.assertEqual(completion.choices[0].message.content, "sdk answer")

    async def test_chat_stream_parses_in_openai_sdk(self):
        client = await self.make_client(FakeProvider("hello", ["hel", "lo"]))
        stream = await client.chat.completions.create(
            model="kingogpt-test",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        text = ""
        async for chunk in stream:
            text += chunk.choices[0].delta.content or ""
        self.assertEqual(text, "hello")

    async def test_responses_output_text_parses_in_openai_sdk(self):
        client = await self.make_client(FakeProvider("responses answer"))
        response = await client.responses.create(model="kingogpt-test", input="hello")
        self.assertEqual(response.object, "response")
        self.assertEqual(response.output_text, "responses answer")


if __name__ == "__main__":
    unittest.main()
