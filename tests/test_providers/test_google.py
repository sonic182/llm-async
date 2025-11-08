from unittest.mock import AsyncMock, patch

import pytest

from llm_async.models import Message, Response
from llm_async.providers.google import GoogleProvider


@pytest.mark.asyncio
async def test_google_init() -> None:
    provider = GoogleProvider(api_key="test_key")
    assert provider.api_key == "test_key"
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta/models/"


def test_google_serializes_message_instances() -> None:
    provider = GoogleProvider(api_key="test_key")
    message = Message(role="user", content="Hello")
    serialized = provider._serialize_messages([message])
    assert serialized == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_google_acomplete_non_stream_success() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Hello!"}]}}]
        }
        mock_client.post.return_value = mock_response

        provider = GoogleProvider(api_key="test_key")
        result = await provider.acomplete(
            model="gemini-2.5-flash", messages=[{"role": "user", "content": "Hi"}]
        )
        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.content == "Hello!"
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_google_acomplete_with_response_schema() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"location": "London"}'}]}}]
        }
        mock_client.post.return_value = mock_response

        provider = GoogleProvider(api_key="test_key")
        schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
        await provider.acomplete(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Where?"}],
            response_schema=schema,
        )
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["generationConfig"] == {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        }
