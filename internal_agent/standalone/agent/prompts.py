import json


PLANNER_PROMPT = """
Return one JSON object only.
Use one of these forms:
{"action":"search_docs","args":{"query":"..."}}
{"action":"run_python","args":{"code":"..."}}
{"action":"final","args":{"answer":"..."}}
Do not add markdown or explanation.
""".strip()


def build_agent_prompt(task: str, history: list[dict]) -> str:
    return f"""
{PLANNER_PROMPT}

Task:
{task}

History:
{json.dumps(history, ensure_ascii=False, indent=2)}

Next JSON:
""".strip()


def build_repair_prompt(
    bad_output: str,
    error: str,
    *,
    task: str | None = None,
    history: list[dict] | None = None,
) -> str:
    context = ""
    if task is not None:
        context += f"\nOriginal user task:\n{task}\n"
    if history is not None:
        context += "\nPrevious valid tool steps:\n"
        context += json.dumps(history, ensure_ascii=False, indent=2)
        context += "\n"

    return f"""
The previous output was invalid.

Error:
{error}
{context}

Bad output:
{bad_output}

Return one corrected JSON object only.
For final answers use: {{"action":"final","args":{{"answer":"..."}}}}
""".strip()
