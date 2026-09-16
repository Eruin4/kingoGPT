"""Verify a real multi-step agent using only disposable synthetic files."""
import argparse
import asyncio
import json
import os
import secrets
import tempfile

from agents import Agent, OpenAIChatCompletionsModel, OpenAIResponsesModel, Runner, RunConfig, RunHooks
from openai import AsyncOpenAI

from kingogpt.agent import Workspace, workspace_functions


class Progress(RunHooks):
    async def on_tool_start(self, context, agent, tool):
        print(f"  {tool.name}", flush=True)


async def probe(client, model_class, *, stream=False):
    with tempfile.TemporaryDirectory(prefix="kingogpt-agent-probe-") as root:
        workspace = Workspace(root, allow_write=True)
        nonce = secrets.token_hex(8)
        workspace.write_file("input.txt", f"reference={nonce}\nvalues=17,25\n")
        agent = Agent(
            name="KingoGPT probe",
            instructions="Use actual file operations. Read before writing, and read back after writing to verify. Never invent results.",
            model=model_class(model="kingogpt", openai_client=client),
            tools=workspace_functions(workspace),
        )
        prompt = (
            "Read input.txt. Add its values. Write result.txt with the reference unchanged and the sum. "
            "Read result.txt back to verify. Report the reference and sum."
        )
        options = dict(max_turns=12, run_config=RunConfig(tracing_disabled=True), hooks=Progress())
        if stream:
            result = Runner.run_streamed(agent, prompt, **options)
            async for _ in result.stream_events():
                pass
        else:
            result = await Runner.run(agent, prompt, **options)
        output = workspace.read_file("result.txt")
        calls = [item.raw_item for item in result.new_items if item.type == "tool_call_item"]
        wrote = False
        verified = False
        read_input = False
        for call in calls:
            arguments = json.loads(call.arguments)
            path = arguments.get("path", "").removeprefix("./")
            if call.name == "read_file" and path == "input.txt":
                read_input = True
            if call.name == "write_file" and path == "result.txt":
                assert read_input, "Wrote output before reading input"
                wrote = True
            if wrote and call.name == "read_file" and path == "result.txt":
                verified = True
        assert wrote and verified, "No write followed by read-back verification"
        assert nonce in output and "42" in output, "Output did not use the actual input"
        assert nonce in str(result.final_output) and "42" in str(result.final_output), "Final answer did not use the actual results"
        return {"api": model_class.__name__, "stream": stream, "tool_calls": len(calls), "status": "passed"}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("KINGOGPT_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("KINGOGPT_SERVER_API_KEY", "local"))
    parser.add_argument("--api", choices=["chat", "responses", "both"], default="both")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    async with AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=360, max_retries=0) as client:
        models = []
        if args.api in ("chat", "both"):
            models.append(OpenAIChatCompletionsModel)
        if args.api in ("responses", "both"):
            models.append(OpenAIResponsesModel)
        for model_class in models:
            print(json.dumps(await probe(client, model_class, stream=args.stream)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
