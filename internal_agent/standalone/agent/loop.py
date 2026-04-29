from collections.abc import Callable
from typing import Any

from internal_agent.config import DEFAULT_MAX_STEPS
from internal_agent.standalone.agent.parser import extract_json, validate_action
from internal_agent.standalone.agent.prompts import build_agent_prompt, build_repair_prompt
from internal_agent.standalone.agent.state import make_history_entry
from internal_agent.standalone.tools.registry import execute_tool


ToolExecutor = Callable[[str, dict[str, Any]], str]


class Agent:
    def __init__(
        self,
        llm,
        max_steps: int = DEFAULT_MAX_STEPS,
        tool_executor: ToolExecutor = execute_tool,
        allow_text_fallback: bool = True,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.tool_executor = tool_executor
        self.allow_text_fallback = allow_text_fallback

    def _parse_or_repair(
        self,
        raw: str,
        *,
        task: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            return validate_action(extract_json(raw))
        except Exception as exc:
            repair_prompt = build_repair_prompt(raw, str(exc), task=task, history=history)
            repaired = self.llm.complete(repair_prompt)
            try:
                return validate_action(extract_json(repaired))
            except Exception:
                fallback_text = raw.strip() or repaired.strip()
                if self.allow_text_fallback and fallback_text:
                    return {"action": "final", "args": {"answer": fallback_text}}
                raise

    def run(self, task: str) -> str:
        history: list[dict[str, Any]] = []

        for step in range(self.max_steps):
            prompt = build_agent_prompt(task, history)
            raw = self.llm.complete(prompt)
            action_obj = self._parse_or_repair(raw, task=task, history=history)

            action = action_obj["action"]
            args = action_obj.get("args", {})

            if action == "final":
                answer = args.get("answer", "")
                return answer if isinstance(answer, str) else str(answer)

            try:
                observation = self.tool_executor(action, args)
            except Exception as exc:
                observation = f"TOOL_ERROR: {type(exc).__name__}: {exc}"

            history.append(
                make_history_entry(
                    step=step + 1,
                    action=action,
                    args=args,
                    observation=str(observation),
                )
            )

        return "Stopped: max_steps reached without final answer."
