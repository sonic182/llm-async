import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from llm_async.models.message import Message
from llm_async.models.response_schema import ResponseSchema
from llm_async.providers import (
    GoogleProvider,
    OpenAIProvider,
    OpenAIResponsesProvider,
    OpenRouterProvider,
)

load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_RESPONSES_MODEL = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4.1-mini")

PROMPT = "Return a JSON object with today's forecast, including a summary and probability of rain."


@dataclass
class ForecastSchema:
    location: str = field(metadata={"description": "City or region for the forecast"})
    summary: str = field(metadata={"description": "Short natural language forecast"})
    rain_probability: float = field(
        metadata={
            "description": "Chance of precipitation as a float between 0 and 1",
            "minimum": 0,
            "maximum": 1,
        }
    )


SHARED_SCHEMA = ResponseSchema.from_dataclass(ForecastSchema)


def require(key: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


def pretty_print(name: str, payload: dict[str, Any]) -> None:
    print(f"\n{name} structured output:")
    print(json.dumps(payload, indent=2))


async def gather_structured_outputs() -> None:
    messages = [Message(role="user", content=PROMPT)]

    openai_provider = OpenAIProvider(api_key=require("OPENAI_API_KEY", OPENAI_API_KEY))
    openrouter_provider = OpenRouterProvider(
        api_key=require("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    )
    google_provider = GoogleProvider(api_key=require("GEMINI_API_KEY", GEMINI_API_KEY))
    responses_provider = OpenAIResponsesProvider(api_key=require("OPENAI_API_KEY", OPENAI_API_KEY))

    providers = [
        ("OpenAI", openai_provider, OPENAI_MODEL),
        ("OpenRouter", openrouter_provider, OPENROUTER_MODEL),
        ("Google", google_provider, GEMINI_MODEL),
        ("OpenAI Responses", responses_provider, OPENAI_RESPONSES_MODEL),
    ]

    for name, provider, model in providers:
        response = await provider.acomplete(
            model=model,
            messages=messages,
            response_schema=SHARED_SCHEMA,
        )
        content = response.main_response.content if response.main_response else ""
        try:
            pretty_print(name, json.loads(content))
        except json.JSONDecodeError:
            print(f"\n{name} response was not valid JSON. Raw content:\n{content}")


if __name__ == "__main__":
    asyncio.run(gather_structured_outputs())
