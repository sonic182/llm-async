import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from llm_async.models.message import Message
from llm_async.providers import OpenAIResponsesProvider

load_dotenv(".env")

X_AI_API_KEY = os.environ.get("X_AI_API_KEY")
X_AI_MODEL = os.environ.get("X_AI_MODEL", "grok-4-1-fast-reasoning")


def require(key: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


async def main() -> None:
    provider = OpenAIResponsesProvider(
        api_key=require("X_AI_API_KEY", X_AI_API_KEY),
        base_url="https://api.x.ai/v1",
    )

    response = await provider.acomplete(
        model=X_AI_MODEL,
        messages=[Message("user", "Give me one short tip for writing better async Python code.")],
    )

    if not response.main_response:
        raise RuntimeError("No response returned")

    print(response.main_response.content)


if __name__ == "__main__":
    asyncio.run(main())
