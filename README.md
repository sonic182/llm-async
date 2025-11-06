# llm_async

An async-first Python library for interacting with Large Language Model (LLM) providers.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Using Poetry (Recommended)](#using-poetry-recommended)
  - [Using pip](#using-pip)
- [Usage](#usage)
  - [Basic Chat Completion](#basic-chat-completion)
    - [OpenAI](#openai)
    - [OpenRouter](#openrouter)
    - [Google Gemini](#google-gemini)
  - [Custom Base URL](#custom-base-url)
  - [Tool Usage](#tool-usage)
  - [Pub/Sub Events for Tool Execution](#pubsub-events-for-tool-execution)
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

- **Async-first**: Built with asyncio for high-performance, non-blocking operations.
- **Provider Support**: Supports OpenAI, Anthropic Claude, Google Gemini, and OpenRouter for chat completions.
- **Tool Calling**: Automatic tool execution with unified tool definitions across providers.
- **Structured Outputs**: Enforce JSON schema validation on responses (OpenAI, Google, OpenRouter).
- **Extensible**: Easy to add new providers by inheriting from `BaseProvider`.
- **Tested**: Comprehensive test suite with high coverage.

## Installation

### Using Poetry (Recommended)

```bash
poetry add llm_async
```

### Using pip

```bash
pip install git+https://github.com/sonic182/llm_async.git
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
from llm_async.models import Tool
from llm_async.providers import OpenAIProvider, ClaudeProvider

# Define a calculator tool that works with both providers
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
    elif operation == "multiply":
        return a * b
    # ... other operations

async def main():
    # Initialize providers
    openai = OpenAIProvider(api_key="your-openai-key")
    claude = ClaudeProvider(api_key="your-anthropic-key")
    
    # Tool executor
    tool_executor = {"calculator": calculator}
    
    # Use the same tool with OpenAI
    response = await openai.acomplete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 15 + 27?"}],
        tools=[calculator_tool],
        tool_executor=tool_executor
    )
    print(f"OpenAI: {response}")
    
    # Use the same tool with Claude
    response = await claude.acomplete(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "What is 15 + 27?"}],
        tools=[calculator_tool],
        tool_executor=tool_executor
    )
    print(f"Claude: {response}")

asyncio.run(main())
```

### Pub/Sub Events for Tool Execution

llm_async supports real-time event emission during tool execution via a pub/sub system. This allows you to monitor tool progress, handle errors, and build interactive UIs for agentic workflows.

Events are emitted for each tool call with topics like `tools.{provider}.{tool_name}.{status}` where status is `start`, `complete`, or `error`.

#### Basic Usage

```python
import asyncio
from llm_async import OpenAIProvider
from llm_async.pubsub import LocalQueueBackend, PubSub
from llm_async.models import Tool

# Define your tool (same as above)
calculator_tool = Tool(...)  # See Tool Usage example

def calculator(operation: str, a: float, b: float) -> float:
    # Implementation
    pass

async def event_monitor(pubsub: PubSub):
    """Monitor tool execution events."""
    print("📡 Monitoring tool events...")
    async for event in pubsub.subscribe("tools.*"):
        topic = event.topic
        payload = event.payload
        
        if "start" in topic:
            print(f"⏱️  STARTED: {payload.get('tool_name')} with args {payload.get('args')}")
        elif "complete" in topic:
            print(f"✅ COMPLETED: {payload.get('tool_name')} -> {payload.get('result')}")
        elif "error" in topic:
            print(f"❌ ERROR: {payload.get('tool_name')} - {payload.get('error')}")

async def main():
    # Setup pub/sub
    backend = LocalQueueBackend()
    pubsub = PubSub(backend)
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(event_monitor(pubsub))
    
    try:
        provider = OpenAIProvider(api_key="your-openai-key")
        
        response = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 15 + 27"}],
            tools=[calculator_tool],
            tool_executor={"calculator": calculator},
            pubsub=pubsub  # Enable event emission
        )
        
        print(f"\n🤖 Final Response: {response.main_response.content}")
    finally:
        await asyncio.sleep(0.2)  # Allow final events
        await pubsub.close()
        monitor_task.cancel()

asyncio.run(main())
```

#### Event Payloads

- **Start**: `{"call_id": str, "tool_name": str, "args": dict}`
- **Complete**: `{"call_id": str, "tool_name": str, "result": str}`
- **Error**: `{"call_id": str, "tool_name": str, "error": str}`

#### Backend Options

- **LocalQueueBackend**: In-memory asyncio queues (default, for single-process)
- Future backends: Redis, RabbitMQ (extensible via `PubSubBackend`)




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
git clone https://github.com/sonic182/llm_async.git
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
