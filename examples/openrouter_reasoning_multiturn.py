import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from llm_async.models import Tool
from llm_async.models.message import Message, message_to_dict
from llm_async.providers import OpenRouterProvider

load_dotenv(".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "z-ai/glm-4.7-flash"


def require(key: str, value: Any) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return str(value)


def get_utc_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


TOOLS = [
    Tool(
        name="get_utc_time",
        description="Returns the current UTC date and time.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
]

TOOLS_MAP = {"get_utc_time": get_utc_time}


def print_turn(label: str, content: str, original: dict[str, Any] | None = None) -> None:
    print(f"\n=== {label} ===")
    print(f"Content: {content}")
    if original and original.get("reasoning"):
        print(f"Reasoning: {original['reasoning'][:120]}...")


async def main() -> None:
    provider = OpenRouterProvider(api_key=require("OPENROUTER_API_KEY", OPENROUTER_API_KEY))
    messages: list[Any] = [Message(role="user", content="What is the current UTC time?")]

    response = await provider.acomplete(model=MODEL, messages=messages, tools=TOOLS)
    main_response = response.main_response
    if not main_response:
        raise RuntimeError("No response returned")

    if main_response.tool_calls:
        tool_call = main_response.tool_calls[0]
        tool_result = await provider.execute_tool(tool_call, TOOLS_MAP)

        messages = [message_to_dict(m) if isinstance(m, Message) else m for m in messages]
        messages.extend([main_response.original or {}, tool_result])

        response = await provider.acomplete(model=MODEL, messages=messages)
        main_response = response.main_response
        if not main_response:
            raise RuntimeError("No follow-up response returned")

    print_turn("Turn 1 — current UTC time", main_response.content, main_response.original)

    messages = [message_to_dict(m) if isinstance(m, Message) else m for m in messages]
    messages.append({"role": "assistant", "content": main_response.content})
    messages.append({"role": "user", "content": "Which vegetables are good to sow right now?"})

    response = await provider.acomplete(model=MODEL, messages=messages)
    main_response = response.main_response
    if not main_response:
        raise RuntimeError("No response for second turn")

    print_turn("Turn 2 — vegetables to sow", main_response.content, main_response.original)

    print("\n=== Full original payload (turn 2) ===")
    print(json.dumps(main_response.original or {}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
