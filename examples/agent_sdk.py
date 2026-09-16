"""Run with: python examples/agent_sdk.py (requires a running local gateway)."""
import asyncio
import os

from agents import Agent, OpenAIChatCompletionsModel, Runner, RunConfig, function_tool
from openai import AsyncOpenAI


@function_tool
def add_numbers(numbers: list[int]) -> int:
    """Add a list of integers and return the sum."""
    return sum(numbers)


async def main():
    async with AsyncOpenAI(
        base_url=os.getenv("KINGOGPT_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("KINGOGPT_SERVER_API_KEY", "local"),
        max_retries=0,
        timeout=360,
    ) as client:
        agent = Agent(
            name="KingoGPT",
            instructions="Calculate using add_numbers and answer from its actual result.",
            model=OpenAIChatCompletionsModel(model="kingogpt", openai_client=client),
            tools=[add_numbers],
        )
        result = await Runner.run(
            agent, "17, 25, 8을 더해줘.", max_turns=6,
            run_config=RunConfig(tracing_disabled=True),
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
