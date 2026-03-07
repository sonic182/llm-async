import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv

from llm_async.models.message import Message
from llm_async.providers import OpenRouterProvider

load_dotenv(".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "z-ai/glm-4.7-flash"


def require(key: str, value: Any) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return str(value)


async def main() -> None:
    provider = OpenRouterProvider(api_key=require("OPENROUTER_API_KEY", OPENROUTER_API_KEY))
    response = await provider.acomplete(
        model=MODEL,
        messages=[
            Message(
                role="user",
                content=(
                    "Return a JSON object with a single key named answer and value 'Pong.'. "
                    "Do not add markdown."
                ),
            )
        ],
    )

    if response.main_response is None:
        print("No main response returned.")
        return

    print("Normalized content:")
    print(response.main_response.content)

    print("\nOriginal message payload:")
    print(json.dumps(response.main_response.original or {}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
