import asyncio
import os

from dotenv import load_dotenv

from llm_async.models.message import Message
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
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

MESSAGES = [Message(role="user", content="Write a short poem about the ocean.")]


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


async def stream_provider(provider_name: str, provider: BaseProvider, model_name: str) -> None:
    print(f"\nStreaming response from {provider_name} ({model_name}):")
    print("-" * 50)

    response = await provider.acomplete(
        model=model_name,
        messages=list(MESSAGES),
        stream=True,
    )

    async for chunk in response.stream_content():
        print(chunk, end="", flush=True)

    print("\n" + "-" * 50)
    print(f"{provider_name} stream completed!\n")


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
        await stream_provider(provider_name, provider, model_name)


if __name__ == "__main__":
    asyncio.run(main())
