import asyncio
import os

from dotenv import load_dotenv

from llm_async import Agent, tool
from llm_async.providers import OpenAIProvider, OpenAIResponsesProvider, OpenRouterProvider

load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_RESPONSES_MODEL = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4.1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def require(key: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations: add, subtract, multiply, or divide."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        return a / b
    return 0


@tool
def word_count(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


MESSAGES = [
    {
        "role": "user",
        "content": (
            "Do two things: "
            "1) What is 15 + 27? Use the calculator. "
            "2) How many words are in 'the quick brown fox'? Use word_count."
        ),
    }
]


async def run_agent(label: str, agent: Agent) -> None:
    print(f"\n--- {label} ---")
    response = await agent.acomplete(messages=list(MESSAGES))
    print(response.main_response.content)  # type: ignore[union-attr]


async def main() -> None:
    openai_key = require("OPENAI_API_KEY", OPENAI_API_KEY)
    openrouter_key = require("OPENROUTER_API_KEY", OPENROUTER_API_KEY)

    agents = [
        (
            "OpenAI",
            Agent(
                provider=OpenAIProvider(api_key=openai_key),
                model=OPENAI_MODEL,
                tools=[calculator, word_count],
                parallel_tools=True,
            ),
        ),
        (
            "OpenAI Responses",
            Agent(
                provider=OpenAIResponsesProvider(api_key=openai_key),
                model=OPENAI_RESPONSES_MODEL,
                tools=[calculator, word_count],
                parallel_tools=True,
            ),
        ),
        (
            "OpenRouter",
            Agent(
                provider=OpenRouterProvider(api_key=openrouter_key),
                model=OPENROUTER_MODEL,
                tools=[calculator, word_count],
                parallel_tools=True,
            ),
        ),
    ]

    for label, agent in agents:
        await run_agent(label, agent)


if __name__ == "__main__":
    asyncio.run(main())
