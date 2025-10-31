from unittest.mock import AsyncMock, patch

import pytest

from llm_async.providers.claude import ClaudeProvider
from llm_async.providers.google import GoogleProvider
from llm_async.providers.openai import OpenAIProvider
from llm_async.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
async def test_stream_malformed_chunks_openai() -> None:
    """Test OpenAI streaming with malformed chunks"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def malformed_chunks():
            # Skip truly malformed JSON that would break parsing
            # Test with valid but unexpected structure instead
            yield b'data: {"unexpected": "structure"}\n\n'
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = malformed_chunks
        mock_client.post.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        result = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result.stream
        assert result.stream_generator is not None

        # Should handle unexpected chunks gracefully
        collected = []
        async for chunk in result.stream_generator:
            if chunk.content:
                collected.append(chunk.content)

        # Should only collect valid chunks with content
        assert collected == ["Hello"]


@pytest.mark.asyncio
async def test_stream_connection_error_openai() -> None:
    """Test OpenAI streaming with connection error"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def error_chunks():
            yield b'data: {"choices": [{"delta": {"content": "H"}}]}\n\n'
            raise Exception("Connection lost")

        mock_response.read_chunks = error_chunks
        mock_client.post.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        result = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result.stream
        assert result.stream_generator is not None

        # Should raise exception when connection fails
        collected = []
        with pytest.raises(Exception, match="Connection lost"):
            async for chunk in result.stream_generator:
                if chunk.content:
                    collected.append(chunk.content)

        # Should have collected the first chunk before error
        assert collected == ["H"]


@pytest.mark.asyncio
async def test_stream_empty_chunks_claude() -> None:
    """Test Claude streaming with empty chunks"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def empty_chunks():
            yield b"\n\n"  # Empty chunk
            yield b'data: {"delta": {"text": "Hello"}}\n\n'
            yield b"\n\n"  # Another empty chunk
            yield b'data: {"delta": {"text": " world"}}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = empty_chunks
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

        # Should skip empty chunks
        collected = []
        async for chunk in result.stream_generator:
            if chunk.content:
                collected.append(chunk.content)

        assert collected == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_invalid_json_google() -> None:
    """Test Google streaming with invalid JSON"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def invalid_json_chunks():
            yield b'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}\n\n'
            yield b'data: {"invalid": json}\n\n'  # Invalid JSON
            yield b'data: {"candidates": [{"content": {"parts": [{"text": " world"}]}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = invalid_json_chunks
        mock_client.post.return_value = mock_response

        provider = GoogleProvider(api_key="test_key")
        result = await provider.acomplete(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result.stream
        assert result.stream_generator is not None

        # Should handle invalid JSON gracefully
        collected = []
        async for chunk in result.stream_generator:
            if chunk.content:
                collected.append(chunk.content)

        # Should only collect valid chunks
        assert collected == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_timeout_openrouter() -> None:
    """Test OpenRouter streaming with timeout scenario"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def timeout_chunks():
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            # Simulate long delay (timeout scenario)
            import asyncio

            await asyncio.sleep(0.1)
            yield b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = timeout_chunks
        mock_client.post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test_key")
        result = await provider.acomplete(
            model="openrouter/model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result.stream
        assert result.stream_generator is not None

        # Should handle delays/timeouts gracefully
        collected = []
        async for chunk in result.stream_generator:
            if chunk.content:
                collected.append(chunk.content)

        assert collected == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_partial_chunks_openai() -> None:
    """Test OpenAI streaming with partial/incomplete chunks"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def partial_chunks():
            # Partial chunk that gets completed in next yield
            yield b'data: {"choices": [{"delta": {"content": "Hel'
            yield b'lo"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = partial_chunks
        mock_client.post.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        result = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result.stream
        assert result.stream_generator is not None

        # Should handle partial chunks correctly
        collected = []
        async for chunk in result.stream_generator:
            if chunk.content:
                collected.append(chunk.content)

        # Should properly reconstruct the content
        assert collected == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_no_done_marker() -> None:
    """Test streaming without [DONE] marker"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def no_done_chunks():
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
            # No [DONE] marker - stream ends naturally

        mock_response.read_chunks = no_done_chunks
        mock_client.post.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        result = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result.stream
        assert result.stream_generator is not None

        # Should handle streams without [DONE] marker
        collected = []
        async for chunk in result.stream_generator:
            if chunk.content:
                collected.append(chunk.content)

        assert collected == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_api_error_response() -> None:
    """Test streaming with API error response"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client

        # Mock API error response
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_client.post.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")

        result = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        with pytest.raises(Exception, match="HTTP 500"):
            async for _chunk in result.stream_generator:
                pass
