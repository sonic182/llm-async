from unittest.mock import AsyncMock, patch

import pytest

from llm_async.models import Response, Tool
from llm_async.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
async def test_openrouter_init() -> None:
    provider = OpenRouterProvider(api_key="test_key")
    assert provider.api_key == "test_key"
    assert provider.base_url == "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_openrouter_acomplete_non_stream_success() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_client.post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test_key")
        result = await provider.acomplete(
            model="openrouter/model", messages=[{"role": "user", "content": "Hi"}]
        )
        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.content == "Hello!"
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_openrouter_acomplete_non_stream_error() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_client.post.side_effect = Exception("API Error")

        provider = OpenRouterProvider(api_key="test_key")
        with pytest.raises(Exception):  # noqa: B017
            await provider.acomplete(
                model="openrouter/model",
                messages=[{"role": "user", "content": "Hi"}],
            )


@pytest.mark.asyncio
async def test_openrouter_acomplete_stream_success() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_read_chunks():
            yield b'data: {"choices": [{"delta": {"content": "H"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": "i"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = fake_read_chunks
        mock_client.post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test_key")
        result = await provider.acomplete(
            model="openrouter/model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        # result should be a Response object with stream_generator
        assert result.stream
        assert result.stream_generator is not None
        collected = [chunk.content async for chunk in result.stream_generator]
        assert collected == ["H", "i"]
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_openrouter_acomplete_stream_error() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_read_chunks():
            yield b'data: {"choices": [{"delta": {"content": "H"}}]}\n\n'
            raise Exception("Stream connection error")

        mock_response.read_chunks = fake_read_chunks
        mock_client.post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test_key")
        result = await provider.acomplete(
            model="openrouter/model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result.stream
        assert result.stream_generator is not None

        # Test that stream error is handled gracefully
        collected = []
        try:
            async for chunk in result.stream_generator:
                collected.append(chunk.content)
        except Exception:
            pass  # Expected to fail due to mock error

        assert collected == ["H"]  # Should have collected the first chunk
        mock_client.post.assert_called_once()


def test_openrouter_tool_formatting() -> None:
    provider = OpenRouterProvider(api_key="test_key")
    tool = Tool(
        name="get_weather",
        description="Get current weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
    )
    formatted = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ]
    # OpenRouter uses the same tool formatting as OpenAI
    assert provider._format_tools([tool]) == formatted


@pytest.mark.asyncio
async def test_openrouter_acomplete_with_tools() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "The weather is sunny.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_client.post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test_key")
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        )
        result = await provider.acomplete(
            model="openrouter/model",
            messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            tools=[tool],
            tool_choice="auto",
        )
        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.content == "The weather is sunny."
        assert result.main_response.tool_calls is not None
        assert len(result.main_response.tool_calls) == 1
        tc = result.main_response.tool_calls[0]
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function == {"name": "get_weather", "arguments": '{"location": "NYC"}'}
        # Check payload
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert "tools" in payload
        assert "tool_choice" in payload


@pytest.mark.asyncio
async def test_openrouter_acomplete_with_multiple_tools() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll help you with multiple operations.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            },
                            {
                                "id": "call_124",
                                "type": "function",
                                "function": {
                                    "name": "calculate_total",
                                    "arguments": '{"items": [10, 20, 30]}',
                                },
                            },
                        ],
                    }
                }
            ]
        }
        mock_client.post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test_key")
        tools = [
            Tool(
                name="get_weather",
                description="Get current weather",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            ),
            Tool(
                name="calculate_total",
                description="Calculate total of items",
                parameters={
                    "type": "object",
                    "properties": {"items": {"type": "array", "items": {"type": "number"}}},
                },
            ),
        ]
        result = await provider.acomplete(
            model="openrouter/model",
            messages=[{"role": "user", "content": "Get weather and calculate total"}],
            tools=tools,
            tool_choice="auto",
        )
        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.tool_calls is not None
        assert len(result.main_response.tool_calls) == 2

        # Check payload
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert len(payload["tools"]) == 2


@pytest.mark.asyncio
async def test_openrouter_custom_headers() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_client.post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test_key")
        await provider.acomplete(
            model="openrouter/model",
            messages=[{"role": "user", "content": "Hi"}],
            http_referer="https://example.com",
            x_title="My App",
        )

        # Check that custom headers were added
        call_args = mock_client.post.call_args
        headers = call_args[1]["headers"]
        assert headers["HTTP-Referer"] == "https://example.com"
        assert headers["X-Title"] == "My App"


@pytest.mark.asyncio
async def test_openrouter_acomplete_with_response_schema() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"location": "London"}'}}]
        }
        mock_client.post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test_key")
        schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
        await provider.acomplete(
            model="openrouter/model",
            messages=[{"role": "user", "content": "Where?"}],
            response_schema=schema,
        )
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }
