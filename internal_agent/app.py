import argparse
import json
import sys

from internal_agent.agent.loop import Agent
from internal_agent.config import DEFAULT_MAX_STEPS
from internal_agent.llm.azure_web_adapter import AzureWebLLM


class StaticLLM:
    def __init__(self, response: str):
        self.response = response

    def complete(self, prompt: str) -> str:
        return self.response


def configure_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the internal KingoGPT agent CLI.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--reuse-thread",
        action="store_true",
        help="Reuse the previous KingoGPT chat thread for adapter calls.",
    )
    parser.add_argument(
        "--keep-thread",
        action="store_true",
        help="Do not delete the KingoGPT chat thread after each adapter call.",
    )
    parser.add_argument(
        "--echo-llm",
        action="store_true",
        help="Print raw KingoGPT streaming chunks while the adapter runs.",
    )
    parser.add_argument(
        "--fake-response",
        default=None,
        help="Return this raw LLM response for local CLI smoke tests.",
    )
    parser.add_argument(
        "--fake-final-answer",
        default=None,
        help="Return a valid final JSON action with this answer for local CLI smoke tests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_output()
    args = parse_args(argv)
    fake_response = args.fake_response
    if args.fake_final_answer is not None:
        fake_response = json.dumps(
            {"action": "final", "args": {"answer": args.fake_final_answer}},
            ensure_ascii=False,
        )

    llm = (
        StaticLLM(fake_response)
        if fake_response is not None
        else AzureWebLLM(
            reuse_thread=args.reuse_thread,
            auto_delete_thread=not args.keep_thread,
            echo=args.echo_llm,
        )
    )
    agent = Agent(llm, max_steps=args.max_steps)

    while True:
        try:
            task = input("\nTask> ").strip()
        except EOFError:
            print()
            return 0

        if task.lower() in {"exit", "quit"}:
            return 0
        if not task:
            continue

        answer = agent.run(task)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    raise SystemExit(main())
