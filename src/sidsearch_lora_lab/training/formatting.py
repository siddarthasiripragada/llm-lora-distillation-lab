from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = "You are a SidSearch planner. Return exactly one JSON object that follows the SidSearch schema."


def format_chat_example(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": json.dumps(row["expected_output"], sort_keys=True)},
    ]


def format_plain_text(row: dict[str, Any]) -> str:
    messages = format_chat_example(row)
    return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)

