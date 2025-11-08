import asyncio
import os

from dotenv import load_dotenv

from llm_async.models import Tool
from llm_async.models.message import Message, message_to_dict
from llm_async.providers import (
    BaseProvider,
    ClaudeProvider,
    GoogleProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

# Requires llm_async and python-dotenv installed in your environment.
load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


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
    },
    input_schema={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
    },
)


MESSAGES = [Message("user", "What is 15 + 27?")]


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


def resolve_api_keys() -> dict[str, str]:
    keys = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
        "CLAUDE_API_KEY": CLAUDE_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required API keys: {', '.join(missing)}")
    return {name: value or "" for name, value in keys.items()}


async def run_provider(provider_name: str, provider: BaseProvider, model_name: str) -> None:
    response = await provider.acomplete(
        model=model_name,
        messages=list(MESSAGES),
        tools=[CALCULATOR_TOOL],
    )

    main_response = response.main_response
    if not main_response or not main_response.tool_calls:
        raise RuntimeError(f"{provider_name}: No tool calls returned from the model")

    tool_call = main_response.tool_calls[0]
    tool_res = await provider.execute_tool(tool_call, TOOLS_MAP)

    original_message = main_response.original
    if original_message is None:
        raise RuntimeError(f"{provider_name}: Provider response missing original message data")

    messages_with_tool = [message_to_dict(message) for message in MESSAGES]
    messages_with_tool.extend([original_message, tool_res])

    follow_up = await provider.acomplete(model=model_name, messages=messages_with_tool)
    if not follow_up.main_response:
        raise RuntimeError(f"{provider_name}: No completion returned from provider")

    print(f"{provider_name}: {follow_up.main_response.content}")


async def main() -> None:
    keys = resolve_api_keys()

    providers = [
        ("OpenAI", OpenAIProvider(api_key=keys["OPENAI_API_KEY"]), OPENAI_MODEL),
        (
            "OpenRouter",
            OpenRouterProvider(api_key=keys["OPENROUTER_API_KEY"]),
            OPENROUTER_MODEL,
        ),
        ("Claude", ClaudeProvider(api_key=keys["CLAUDE_API_KEY"]), CLAUDE_MODEL),
        ("Google", GoogleProvider(api_key=keys["GEMINI_API_KEY"]), GEMINI_MODEL),
    ]

    for provider_name, provider, model_name in providers:
        await run_provider(provider_name, provider, model_name)


if __name__ == "__main__":
    asyncio.run(main())
