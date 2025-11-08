# llm_async — Async multi‑provider LLM client for Python

High-performance, async-first LLM client for OpenAI, Claude, Google Gemini, and OpenRouter. Built on top of aiosonic for fast, low-latency HTTP and true asyncio streaming across providers.

[![PyPI - Version](https://img.shields.io/pypi/v/llm_async.svg)](https://pypi.org/project/llm_async/)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm_async.svg)](https://pypi.org/project/llm_async/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)
[![Coverage](https://img.shields.io/badge/coverage-%E2%80%94-blue.svg)](#)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Table of Contents

- [Features](#features)
  - [Supported Providers & Features](#supported-providers--features)
- [Installation](#installation)
  - [Using Poetry (Recommended)](#using-poetry-recommended)
  - [Using pip](#using-pip)
- [Quickstart](#quickstart)
- [Usage](#usage)
  - [Basic Chat Completion](#basic-chat-completion)
    - [OpenAI](#openai)
    - [OpenRouter](#openrouter)
    - [Google Gemini](#google-gemini)
  - [Custom Base URL](#custom-base-url)
  - [Tool Usage](#tool-usage)
  - [Structured Outputs](#structured-outputs)
- [API Reference](#api-reference)
  - [OpenAIProvider](#openaiprovider)
  - [OpenRouterProvider](#openrouterprovider)
  - [GoogleProvider](#googleprovider)
- [Development](#development)
  - [Setup](#setup)
  - [Running Tests](#running-tests)
  - [Building](#building)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## Features

### Supported Providers & Features

| Feature | OpenAI | Claude | Google Gemini | OpenRouter |
|---------|--------|--------|---------------|-----------|
| Chat Completions | ✅ | ✅ | ✅ | ✅ |
| Tool Calling | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Structured Outputs | ✅ | ❌ | ✅ | ✅ |

Notes:
- Structured Outputs: Supported by OpenAI, Google Gemini, and OpenRouter; not supported by Claude.
- See [Examples](#examples) for tool-call round-trips and streaming demos.

- **Async-first**: Built with asyncio for high-performance, non-blocking operations.
- **Provider Support**: Supports OpenAI, Anthropic Claude, Google Gemini, and OpenRouter for chat completions.
- **Tool Calling**: Tool execution with unified tool definitions across providers.
- **Structured Outputs**: Enforce JSON schema validation on responses (OpenAI, Google, OpenRouter).
- **Extensible**: Easy to add new providers by inheriting from `BaseProvider`.
- **Tested**: Comprehensive test suite with high coverage.

#### Performance
- Built on top of [aiosonic](https://github.com/sonic182/aiosonic) for fast, low-overhead async HTTP requests and streaming.
- True asyncio end-to-end: concurrent requests across providers with minimal overhead.
- Designed for fast tool-call round-trips and low-latency streaming.

## Installation

### Using Poetry (Recommended)

```bash
poetry add llm_async
```

### Using pip

```bash
pip install llm-async
```

## Quickstart

Minimal async example with streaming using OpenAI-compatible interface:

```python
import asyncio
from llm_async import OpenAIProvider

async def main():
    provider = OpenAIProvider(api_key="YOUR_OPENAI_API_KEY")
    # Stream tokens as they arrive
    async for chunk in await provider.acomplete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Give me 3 ideas for a CLI tool."}],
        stream=True,
    ):
        print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

## Usage

### Basic Chat Completion

#### OpenAI
```python
import asyncio
from llm_async import OpenAIProvider

async def main():
    # Initialize the provider with your API key
    provider = OpenAIProvider(api_key="your-openai-api-key")

    # Perform a chat completion
    response = await provider.acomplete(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )

    print(response.main_response.content)  # Output: The assistant's response

# Run the async function
asyncio.run(main())
```

#### OpenRouter
```python
import asyncio
import os
from llm_async import OpenRouterProvider

async def main():
    # Initialize the provider with your API key
    provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    # Perform a chat completion
    response = await provider.acomplete(
        model="openrouter/auto",  # Let OpenRouter choose the best model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        http_referer="https://github.com/your-username/your-app",  # Optional
        x_title="My AI App"  # Optional
    )

    print(response.main_response.content)  # Output: The assistant's response

# Run the async function
asyncio.run(main())
```

#### Google Gemini
```python
import asyncio
from llm_async.providers.google import GoogleProvider

async def main():
    # Initialize the provider with your API key
    provider = GoogleProvider(api_key="your-google-gemini-api-key")

    # Perform a chat completion
    response = await provider.acomplete(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )

    print(response.main_response.content)  # Output: The assistant's response

# Run the async function
asyncio.run(main())
```

### Custom Base URL

```python
provider = OpenAIProvider(
    api_key="your-api-key",
    base_url="https://custom-openai-endpoint.com/v1"
)
```

### Tool Usage

```python
import asyncio
import os
from llm_async.models import Tool
from llm_async.providers import OpenAIProvider

# Define a calculator tool
calculator_tool = Tool(
    name="calculator",
    description="Perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"]
    },
    input_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"]
    }
)

def calculator(operation: str, a: float, b: float) -> float:
    """Calculator function that can be called by the LLM."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    return 0

async def main():
    # Initialize provider
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Tool executor mapping
    tools_map = {"calculator": calculator}
    
    # Initial user message
    messages = [{"role": "user", "content": "What is 15 + 27?"}]
    
    # First turn: Ask the LLM to perform a calculation
    response = await provider.acomplete(
        model="gpt-4o-mini",
        messages=messages,
        tools=[calculator_tool]
    )
    
    # Execute the tool call
    tool_call = response.main_response.tool_calls[0]
    tool_result = await provider.execute_tool(tool_call, tools_map)
    
    # Second turn: Send the tool result back to the LLM
    messages_with_tool = messages + [response.main_response.original] + [tool_result]
    
    final_response = await provider.acomplete(
        model="gpt-4o-mini",
        messages=messages_with_tool
    )
    
    print(final_response.main_response.content)  # Output: The final answer

asyncio.run(main())
```

## Recipes
- Streaming across providers: see `examples/stream_all_providers.py`
- Tool-call round-trip (calculator): see `examples/tool_call_all_providers.py`
- Structured outputs (JSON schema): see section below and examples

### Examples

The `examples` directory contains runnable scripts for local testing against all supported providers:

- `examples/tool_call_all_providers.py` shows how to execute the same calculator tool call round-trip with OpenAI, OpenRouter, Claude, and Google using shared message/tool definitions.
- `examples/stream_all_providers.py` streams completions from the same provider list so you can compare chunking formats and latency.

Both scripts expect a `.env` file with `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `CLAUDE_API_KEY`, and `GEMINI_API_KEY` (plus optional per-provider model overrides). Run them via Poetry, e.g. `poetry run python examples/tool_call_all_providers.py`.

### Structured Outputs

Enforce JSON schema validation on model responses for consistent, type-safe outputs.

```python
import asyncio
import json
from llm_async import OpenAIProvider
from llm_async.providers.google import GoogleProvider

# Define response schema
response_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False
}

async def main():
    # OpenAI example
    openai_provider = OpenAIProvider(api_key="your-openai-key")
    response = await openai_provider.acomplete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_schema=response_schema
    )
    result = json.loads(response.main_response.content)
    print(f"OpenAI: {result}")

    # Google Gemini example
    google_provider = GoogleProvider(api_key="your-google-key")
    response = await google_provider.acomplete(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_schema=response_schema
    )
    result = json.loads(response.main_response.content)
    print(f"Gemini: {result}")

asyncio.run(main())
```

**Supported Providers**: OpenAI, Google Gemini, OpenRouter. Claude does not support structured outputs.

## Why llm_async?
- Async-first performance (aiosonic-based) vs. sync or heavier HTTP stacks.
- Unified provider interface: same message/tool/streaming patterns across OpenAI, Claude, Gemini, OpenRouter.
- Structured outputs (OpenAI, Google, OpenRouter) with JSON schema validation.
- Tool-call round-trip helpers for consistent multi-turn execution.
- Minimal surface area: easy to extend with new providers via BaseProvider.

## API Reference

### OpenAIProvider

- `__init__(api_key: str, base_url: str = "https://api.openai.com/v1")`
- `acomplete(model: str, messages: list[dict], stream: bool = False, **kwargs) -> Response | AsyncIterator[StreamChunk]`

  Performs a chat completion. When `stream=True` the method returns an async iterator that yields StreamChunk objects as they arrive from the provider.

### OpenRouterProvider

- `__init__(api_key: str, base_url: str = "https://openrouter.ai/api/v1")`
- `acomplete(model: str, messages: list[dict], stream: bool = False, **kwargs) -> Response | AsyncIterator[StreamChunk]`

  Performs a chat completion using OpenRouter's unified API. Supports the same OpenAI-compatible interface with additional optional headers:
  - `http_referer`: Your application's URL (recommended)
  - `x_title`: Your application's name (recommended)

  OpenRouter provides access to hundreds of AI models from various providers through a single API.

### GoogleProvider

- `__init__(api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/")`
- `acomplete(model: str, messages: list[dict], stream: bool = False, **kwargs) -> Response | AsyncIterator[StreamChunk]`

  Performs a chat completion using Google's Gemini API. Supports structured outputs and uses camelCase for API keys (e.g., `generationConfig`).

**Streaming**
- **Usage**: `async for chunk in await provider.acomplete(..., stream=True):` print or process `chunk` in real time.

**Example output**

```
--- OpenAI streaming response ---
1. Peel and slice potatoes.
2. Par-cook potatoes briefly.
3. Whisk eggs with salt and pepper.
4. Sauté onions until translucent (optional).
5. Combine potatoes and eggs in a pan and cook until set.
6. Fold and serve.
--- Claude streaming response ---
1. Prepare potatoes by peeling and slicing.
2. Fry or boil until tender.
3. Beat eggs and season.
4. Mix potatoes with eggs and cook gently.
5. Serve warm.
```

## Development

### Setup

```bash
git clone https://github.com/sonic182/llm-async.git
cd llm_async
poetry install
```

### Running Tests

```bash
poetry run pytest
```

### Building

```bash
poetry build
```

## Roadmap

- Support for additional providers (e.g., Grok, Anthropic direct API)
- More advanced tool features
- Response caching and retry mechanisms

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/sonic182/llm-async).

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- sonic182
