from unittest.mock import AsyncMock, patch

import pytest

from llm_async.models import Response, Tool
from llm_async.providers.claude import ClaudeProvider


@pytest.mark.asyncio
async def test_claude_init() -> None:
    provider = ClaudeProvider(api_key="test_key")
    assert provider.api_key == "test_key"
    assert provider.base_url == "https://api.anthropic.com/v1"


@pytest.mark.asyncio
async def test_claude_acomplete_non_stream_success() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": [{"text": "Hello!"}]}
        mock_client.post.return_value = mock_response

        provider = ClaudeProvider(api_key="test_key")
        result = await provider.acomplete(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1024,
        )
        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.content == "Hello!"
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_claude_acomplete_non_stream_error() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_client.post.side_effect = Exception("API Error")

        provider = ClaudeProvider(api_key="test_key")
        with pytest.raises(Exception):  # noqa: B017
            await provider.acomplete(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1024,
            )


@pytest.mark.asyncio
async def test_claude_acomplete_stream_success() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_read_chunks():
            yield b'data: {"delta": {"text": "H"}}\n\n'
            yield b'data: {"delta": {"text": "i"}}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = fake_read_chunks
        mock_client.post.return_value = mock_response

        provider = ClaudeProvider(api_key="test_key")
        result = await provider.acomplete(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_tokens=1024,
        )

        assert result.stream
        assert result.stream_generator is not None
        collected = [chunk.content async for chunk in result.stream_generator]
        assert collected == ["H", "i"]
        mock_client.post.assert_called_once()


def test_claude_tool_formatting() -> None:
    provider = ClaudeProvider(api_key="test_key")
    tool = Tool(
        name="get_weather",
        description="Get current weather",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
    )
    formatted = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }
    ]
    assert provider._format_tools([tool]) == formatted


@pytest.mark.asyncio
async def test_claude_acomplete_with_tools() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": "The weather is sunny."},
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"location": "NYC"},
                },
            ]
        }
        mock_client.post.return_value = mock_response

        provider = ClaudeProvider(api_key="test_key")
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        )
        result = await provider.acomplete(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "get_weather"},
            max_tokens=1024,
        )
        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.content == "The weather is sunny."
        assert result.main_response.tool_calls is not None
        assert len(result.main_response.tool_calls) == 1
        tc = result.main_response.tool_calls[0]
        assert tc.id == "call_123"
        assert tc.type == "tool_use"
        assert tc.name == "get_weather"
        assert tc.input == {"location": "NYC"}
        # Check payload
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert "tools" in payload
        assert "tool_choice" in payload


@pytest.mark.asyncio
async def test_claude_response_schema_not_supported() -> None:
    provider = ClaudeProvider(api_key="test_key")
    schema = {"type": "object", "properties": {"test": {"type": "string"}}}
    with pytest.raises(NotImplementedError, match="does not support structured outputs"):
        await provider.acomplete(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Test"}],
            response_schema=schema,
        )
