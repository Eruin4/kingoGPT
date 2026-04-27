from internal_agent.config import DEFAULT_OBSERVATION_LIMIT


def make_history_entry(
    *,
    step: int,
    action: str,
    args: dict,
    observation: str,
    limit: int = DEFAULT_OBSERVATION_LIMIT,
) -> dict:
    return {
        "step": step,
        "action": action,
        "args": args,
        "observation": observation[:limit],
    }

