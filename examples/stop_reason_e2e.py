import asyncio
import os

from dotenv import load_dotenv

from llm_async.models import Tool
from llm_async.models.message import Message
from llm_async.providers import ClaudeProvider, GoogleProvider, OpenAIProvider, OpenRouterProvider
from llm_async.providers.openai_responses import OpenAIResponsesProvider

load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENAI_RESPONSES_MODEL = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_CALC_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["operation", "a", "b"],
    "additionalProperties": False,
}

CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Perform basic arithmetic operations",
    parameters=_CALC_SCHEMA,
    input_schema=_CALC_SCHEMA,
)


def check(label: str, stop_reason: str | None, expected: str) -> None:
    status = "OK" if stop_reason == expected else "FAIL"
    print(f"  [{status}] {label}: stop_reason={stop_reason!r}  (expected {expected!r})")


async def test_openai_chat() -> None:
    print("\n=== OpenAI Chat Completions ===")
    provider = OpenAIProvider(api_key=OPENAI_API_KEY)

    resp = await provider.acomplete(
        model=OPENAI_MODEL,
        messages=[Message("user", "Say hello in one word.")],
    )
    check("normal stop", resp.main_response.stop_reason, "stop")

    resp = await provider.acomplete(
        model=OPENAI_MODEL,
        messages=[Message("user", "What is 7 * 6?")],
        tools=[CALCULATOR_TOOL],
    )
    check("tool_calls", resp.main_response.stop_reason, "tool_calls")


async def test_openai_responses() -> None:
    print("\n=== OpenAI Responses API ===")
    provider = OpenAIResponsesProvider(api_key=OPENAI_API_KEY)

    resp = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=[Message("user", "Say hello in one word.")],
    )
    check("normal stop", resp.main_response.stop_reason, "stop")

    resp = await provider.acomplete(
        model=OPENAI_RESPONSES_MODEL,
        messages=[Message("user", "What is 7 * 6?")],
        tools=[CALCULATOR_TOOL],
    )
    check("tool_calls", resp.main_response.stop_reason, "tool_calls")


async def test_openrouter() -> None:
    print("\n=== OpenRouter (Chat Completions) ===")
    provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)

    resp = await provider.acomplete(
        model=OPENROUTER_MODEL,
        messages=[Message("user", "Say hello in one word.")],
    )
    check("normal stop", resp.main_response.stop_reason, "stop")

    resp = await provider.acomplete(
        model=OPENROUTER_MODEL,
        messages=[Message("user", "What is 7 * 6?")],
        tools=[CALCULATOR_TOOL],
    )
    check("tool_calls", resp.main_response.stop_reason, "tool_calls")


async def test_claude() -> None:
    print("\n=== Claude ===")
    provider = ClaudeProvider(api_key=CLAUDE_API_KEY)

    resp = await provider.acomplete(
        model=CLAUDE_MODEL,
        messages=[Message("user", "Say hello in one word.")],
    )
    check("normal stop", resp.main_response.stop_reason, "end_turn")

    resp = await provider.acomplete(
        model=CLAUDE_MODEL,
        messages=[Message("user", "What is 7 * 6?")],
        tools=[CALCULATOR_TOOL],
    )
    check("tool_calls", resp.main_response.stop_reason, "tool_use")


async def test_gemini() -> None:
    print("\n=== Gemini ===")
    provider = GoogleProvider(api_key=GEMINI_API_KEY)

    resp = await provider.acomplete(
        model=GEMINI_MODEL,
        messages=[Message("user", "Say hello in one word.")],
    )
    check("normal stop", resp.main_response.stop_reason, "STOP")

    resp = await provider.acomplete(
        model=GEMINI_MODEL,
        messages=[Message("user", "What is 7 * 6?")],
        tools=[CALCULATOR_TOOL],
    )
    check("tool_calls", resp.main_response.stop_reason, "STOP")


async def main() -> None:
    await test_openai_chat()
    await test_openai_responses()
    await test_openrouter()
    await test_claude()
    await test_gemini()
    print()


if __name__ == "__main__":
    asyncio.run(main())
