import asyncio
import os

from dotenv import load_dotenv

from llm_async.models import Tool
from llm_async.models.message import Message, message_to_dict, normalize_messages
from llm_async.providers import OpenAIResponsesProvider

# Requires llm-async and python-dotenv installed in your environment.
load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_RESPONSES_MODEL = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4.1")


def require(key: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


# Define a calculator tool
CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False,
    },
    input_schema={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False,
    },
)


def calculator(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        return a / b
    return 0


TOOLS_MAP = {"calculator": calculator}


async def main() -> None:
    provider = OpenAIResponsesProvider(api_key=require("OPENAI_API_KEY", OPENAI_API_KEY))

    messages = [Message("user", "What is 15 + 27? Use the calculator tool (force use it).")]

    response = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=list(messages),
        tools=[CALCULATOR_TOOL],
        tool_choice="required",
    )

    main_response = response.main_response
    if not main_response:
        raise RuntimeError("No main response returned")

    if not main_response.tool_calls:
        raise RuntimeError("Model did not request any tool calls")

    tool_call = main_response.tool_calls[0]
    tool_result = await provider.execute_tool(tool_call, TOOLS_MAP)

    messages_with_tool = [message_to_dict(msg) for msg in messages]
    if main_response.original:
        messages_with_tool.append(main_response.original)
    messages_with_tool.append(tool_result)

    final = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=messages_with_tool,
        tools=[CALCULATOR_TOOL],
    )

    if not final.main_response:
        raise RuntimeError("No final response returned")

    print("Final answer:", final.main_response.content)


if __name__ == "__main__":
    asyncio.run(main())
