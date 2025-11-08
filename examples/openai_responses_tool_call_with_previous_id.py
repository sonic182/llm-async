import asyncio
import os
import random
import uuid

from dotenv import load_dotenv

from llm_async.models import Tool
from llm_async.models.message import Message
from llm_async.providers import OpenAIResponsesProvider

load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_RESPONSES_MODEL = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4.1")


def require(key: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name"},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name"},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)


def get_weather(city: str) -> str:
    weather_statuses = ["sunny", "rainy", "cloudy", "snowy", "windy", "foggy"]
    status = random.choice(weather_statuses)
    return f"The weather in {city} is currently {status}."


TOOLS_MAP = {"get_weather": get_weather}


async def main() -> None:
    provider = OpenAIResponsesProvider(api_key=require("OPENAI_API_KEY", OPENAI_API_KEY))

    session_id = uuid.uuid4().hex

    initial_message = Message(
        "user", "What is the weather like in Madrid? Use the get_weather tool to find out."
    )

    response = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=[initial_message],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        prompt_cache_key=session_id,
    )

    main_response = response.main_response
    if not main_response:
        raise RuntimeError("No main response returned")

    if not main_response.tool_calls:
        raise RuntimeError("Model did not request any tool calls")

    tool_call = main_response.tool_calls[0]
    tool_result = await provider.execute_tool(tool_call, TOOLS_MAP)

    response_payload = response.original
    if not isinstance(response_payload, dict) or "id" not in response_payload:
        raise RuntimeError("Response payload missing id for continuing conversation")

    response_id = response_payload["id"]

    final = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=[tool_result],
        tools=[WEATHER_TOOL],
        previous_response_id=response_id,
        prompt_cache_key=session_id,
    )

    if not final.main_response:
        raise RuntimeError("No final response returned")

    print("Final answer:", final.main_response.content)
    print(f"Session ID (prompt_cache_key): {session_id}")


if __name__ == "__main__":
    asyncio.run(main())
