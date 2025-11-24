import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from llm_async.models.message import Message
from llm_async.providers import OpenAIResponsesProvider

# Requires llm-async and python-dotenv installed in your environment.
load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_RESPONSES_MODEL = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-5-nano")


def require(key: str, value: Any) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return str(value)


async def main() -> None:
    print("\n--- OpenAI Responses: streaming ---")
    provider = OpenAIResponsesProvider(
        api_key=require("OPENAI_API_KEY", OPENAI_API_KEY), http2=True
    )

    # Ask for a structured story; stream text deltas as they arrive
    response = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=[Message("user", "Return a short story about a unicorn and a dragon.")],
        reasoning={"effort": "low"},
        stream=True,
    )

    collected = []
    async for chunk in response.stream_content():
        collected.append(chunk)
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
