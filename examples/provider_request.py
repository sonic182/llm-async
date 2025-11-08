import asyncio
import os

from dotenv import load_dotenv

from llm_async.providers import OpenAIResponsesProvider

load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def require(key: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


async def main() -> None:
    provider = OpenAIResponsesProvider(api_key=require("OPENAI_API_KEY", OPENAI_API_KEY))

    print("Fetching available models from OpenAI Responses API...\n")

    response = await provider.request("GET", "/models")

    print("Available models (first 10):")
    for model in response.get("data", [])[:10]:
        print(f"  - {model.get('id')}")

    total = len(response.get("data", []))
    print(f"\n... and {total - 10} more models")
    print(f"Total models available: {total}")


if __name__ == "__main__":
    asyncio.run(main())
