import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv

from llm_async.models.message import Message
from llm_async.models.response_schema import ResponseSchema
from llm_async.providers import OpenAIResponsesProvider

# Requires llm-async and python-dotenv installed in your environment.
load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_RESPONSES_MODEL = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-5-nano")


def require(key: str, value: Any) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return str(value)


async def basic_non_streaming() -> None:
    print("\n--- OpenAI Responses: basic (non-streaming) ---")
    provider = OpenAIResponsesProvider(api_key=require("OPENAI_API_KEY", OPENAI_API_KEY))
    resp = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=[Message("user", "Tell me a three sentence bedtime story about a unicorn.")],
    )
    print(resp.main_response.content if resp.main_response else "<no content>")


async def streaming_with_structured_outputs() -> None:
    print("\n--- OpenAI Responses: streaming with structured outputs ---")
    provider = OpenAIResponsesProvider(api_key=require("OPENAI_API_KEY", OPENAI_API_KEY))

    # JSON schema for a short story with title and body
    response_schema = ResponseSchema(
        schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title", "body"],
            "additionalProperties": False,
        }
    )

    # Ask for a structured story; stream text deltas as they arrive
    response = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=[Message("user", "Return a short JSON story about a unicorn and a dragon.")],
        stream=True,
        response_schema=response_schema,
    )

    # Stream raw deltas (text). Caller can accumulate and json.loads when complete.
    collected = []
    async for chunk in response.stream_content():
        collected.append(chunk)
        print(chunk, end="", flush=True)

    print("\n--- accumulated JSON ---")
    try:
        obj = json.loads("".join(collected))
        print(json.dumps(obj, indent=2))
    except json.JSONDecodeError:
        print("<stream did not form valid JSON; ensure model supports strict schema>")


async def main() -> None:
    await basic_non_streaming()
    await streaming_with_structured_outputs()


if __name__ == "__main__":
    asyncio.run(main())
