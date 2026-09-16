"""Exercise the actual SDK tool loop and stream assembly against the gateway."""
import json
import tempfile
import unittest

try:
    from openai import AsyncOpenAI
    try:
        import httpx2 as sdk_httpx
    except ImportError:
        import httpx as sdk_httpx
    from agents import Agent, OpenAIChatCompletionsModel, OpenAIResponsesModel, Runner, RunConfig
except ImportError:
    AsyncOpenAI = None

from kingogpt.agent import Workspace, workspace_functions
from kingogpt.openai_server import create_app
from test_agent_protocol import SequenceProvider


@unittest.skipIf(AsyncOpenAI is None, "Install .[test] to run Agents SDK integration tests")
class SDKAgentTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, model_class, stream=False):
        provider = SequenceProvider([
            '<decision>{"operation":"op_2","parameters":{"path":"input.txt"}}</decision>',
            '<decision>{"operation":"op_3","parameters":{"path":"output.txt","content":"42"}}</decision>',
            '<decision>{"operation":"op_2","parameters":{"path":"output.txt"}}</decision>',
            '<decision>{"operation":"finish","answer":"Verified: 42"}</decision>',
        ])
        app = create_app(provider=provider)
        async with AsyncOpenAI(base_url="http://test/v1", api_key="local", max_retries=0,
                http_client=sdk_httpx.AsyncClient(transport=sdk_httpx.ASGITransport(app=app))) as client:
            with tempfile.TemporaryDirectory() as root:
                workspace = Workspace(root, allow_write=True)
                workspace.write_file("input.txt", "17 + 25")
                agent = Agent(name="test", model=model_class(model="kingogpt", openai_client=client), tools=workspace_functions(workspace))
                kwargs = dict(max_turns=6, run_config=RunConfig(tracing_disabled=True))
                if stream:
                    result = Runner.run_streamed(agent, "Read input.txt, compute, write output.txt and verify.", **kwargs)
                    async for _ in result.stream_events():
                        pass
                else:
                    result = await Runner.run(agent, "Read input.txt, compute, write output.txt and verify.", **kwargs)
                self.assertEqual(result.final_output, "Verified: 42")
                self.assertEqual(workspace.read_file("output.txt"), "42")
                calls = [item.raw_item.name for item in result.new_items if item.type == "tool_call_item"]
                self.assertEqual(calls, ["read_file", "write_file", "read_file"])
                self.assertIn("17 + 25", provider.prompts[1])
                self.assertIn("bytes_written", provider.prompts[2])
                self.assertEqual(len(provider.prompts), 4)

    async def test_chat_agent_loop(self):
        await self.exercise(OpenAIChatCompletionsModel)

    async def test_responses_agent_loop(self):
        await self.exercise(OpenAIResponsesModel)

    async def test_chat_streamed_agent_loop(self):
        await self.exercise(OpenAIChatCompletionsModel, stream=True)

    async def test_responses_streamed_agent_loop(self):
        await self.exercise(OpenAIResponsesModel, stream=True)
