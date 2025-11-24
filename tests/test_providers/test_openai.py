from unittest.mock import AsyncMock, patch

import pytest

from llm_async.models import Message, Response, Tool
from llm_async.models.response_schema import ResponseSchema
from llm_async.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_init() -> None:
    provider = OpenAIProvider(api_key="test_key")
    assert provider.api_key == "test_key"
    assert provider.base_url == "https://api.openai.com/v1"


def test_openai_serializes_message_instances() -> None:
    provider = OpenAIProvider(api_key="test_key")
    message = Message(role="user", content="Hi")
    serialized = provider._serialize_messages([message])
    assert serialized == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_openai_acomplete_non_stream_success() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_client.post.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        result = await provider.acomplete(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}]
        )
        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.content == "Hello!"
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_openai_acomplete_non_stream_error() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_client.post.side_effect = Exception("API Error")

        provider = OpenAIProvider(api_key="test_key")
        with pytest.raises(Exception):  # noqa: B017
            await provider.acomplete(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )


@pytest.mark.asyncio
async def test_openai_acomplete_stream_success() -> None:
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

        provider = OpenAIProvider(api_key="test_key")
        result = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        # result should be a Response object with stream_generator
        assert result.stream
        assert result.stream_generator is not None
        collected = [chunk.content async for chunk in result.stream_generator]
        assert collected == ["H", "i"]
        mock_client.post.assert_called_once()


def test_openai_tool_formatting() -> None:
    provider = OpenAIProvider(api_key="test_key")
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
    assert provider._format_tools([tool]) == formatted


@pytest.mark.asyncio
async def test_openai_acomplete_with_tools() -> None:
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

        provider = OpenAIProvider(api_key="test_key")
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        )
        result = await provider.acomplete(
            model="gpt-4o-mini",
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
async def test_openai_acomplete_with_response_schema() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"location": "London"}'}}]
        }
        mock_client.post.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        schema = ResponseSchema(
            schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            }
        )
        await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Where?"}],
            response_schema=schema,
        )
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["response_format"] == schema.for_openai()
