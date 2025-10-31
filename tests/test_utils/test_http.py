from unittest.mock import AsyncMock, patch

import pytest

from llm_async.utils.http import post_json, stream_json
from llm_async.utils.retry import RetryConfig


@pytest.mark.asyncio
async def test_post_json() -> None:
    with patch("llm_async.utils.http.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_client.post.return_value = mock_response

        result = await post_json(
            mock_client,
            "https://test.com",
            {"data": "test"},
            {"Authorization": "Bearer test"},
        )
        assert result == {"key": "value"}
        mock_client.post.assert_called_once_with(
            "https://test.com",
            json={"data": "test"},
            headers={"Authorization": "Bearer test"},
        )


@pytest.mark.asyncio
async def test_stream_json() -> None:
    # Since it's a stub, just test it doesn't raise
    async for item in stream_json(None, "https://test.com", {}, {}):
        assert item == {}
        break  # only one yield


@pytest.mark.asyncio
async def test_post_json_retry_on_http_error() -> None:
    """Test that post_json retries on HTTP errors."""
    with patch("llm_async.utils.http.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client

        # First two attempts fail with 500, third succeeds
        mock_response_fail = AsyncMock()
        mock_response_fail.status_code = 500
        mock_response_fail.text.return_value = "Internal Server Error"

        mock_response_success = AsyncMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"key": "value"}

        mock_client.post.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success,
        ]

        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)

        result = await post_json(
            mock_client,
            "https://test.com",
            {"data": "test"},
            {"Authorization": "Bearer test"},
            retry_config=retry_config,
        )

        assert result == {"key": "value"}
        assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_post_json_retry_on_exception() -> None:
    """Test that post_json retries on network exceptions."""
    with patch("llm_async.utils.http.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client

        # First attempt raises ConnectionError, second succeeds
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}

        mock_client.post.side_effect = [ConnectionError("Connection failed"), mock_response]

        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)

        result = await post_json(
            mock_client,
            "https://test.com",
            {"data": "test"},
            {"Authorization": "Bearer test"},
            retry_config=retry_config,
        )

        assert result == {"key": "value"}
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_post_json_no_retry_on_client_error() -> None:
    """Test that post_json does not retry on 4xx errors."""
    with patch("llm_async.utils.http.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.text.return_value = "Bad Request"
        mock_client.post.return_value = mock_response

        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)

        with pytest.raises(Exception, match="HTTP 400"):
            await post_json(
                mock_client,
                "https://test.com",
                {"data": "test"},
                {"Authorization": "Bearer test"},
                retry_config=retry_config,
            )

        # Should only attempt once for client errors
        assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_post_json_exhausts_retries() -> None:
    """Test that post_json raises after exhausting all retry attempts."""
    with patch("llm_async.utils.http.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_client.post.return_value = mock_response

        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)

        with pytest.raises(Exception, match="HTTP 500"):
            await post_json(
                mock_client,
                "https://test.com",
                {"data": "test"},
                {"Authorization": "Bearer test"},
                retry_config=retry_config,
            )

        # Should attempt exactly max_attempts times
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_stream_json_retry_on_initial_connection() -> None:
    """Test that stream_json retries on initial connection errors."""
    with patch("llm_async.utils.http.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client

        # First attempt fails with 500, second succeeds with stream data
        mock_response_fail = AsyncMock()
        mock_response_fail.status_code = 500
        mock_response_fail.text.return_value = "Internal Server Error"

        mock_response_success = AsyncMock()
        mock_response_success.status_code = 200

        # Mock stream chunks - create an async generator
        async def mock_read_chunks():
            yield b'data: {"text": "hello"}\n\n'
            yield b"data: [DONE]\n\n"

        # Create the async generator and assign it to the mock
        mock_response_success.read_chunks = mock_read_chunks

        mock_client.post.side_effect = [mock_response_fail, mock_response_success]

        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)

        chunks = []
        async for chunk in stream_json(
            mock_client,
            "https://test.com",
            {"data": "test"},
            {"Authorization": "Bearer test"},
            retry_config=retry_config,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == {"text": "hello"}
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_stream_json_no_retry_mid_stream() -> None:
    """Test that stream_json does not retry once stream is established."""
    with patch("llm_async.utils.http.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 200

        # Stream starts successfully but then raises an exception mid-stream
        async def mock_read_chunks():
            yield b'data: {"text": "hello"}\n\n'
            raise ConnectionError("Stream interrupted")

        mock_response.read_chunks = mock_read_chunks
        mock_client.post.return_value = mock_response

        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)

        chunks = []
        with pytest.raises(ConnectionError):
            async for chunk in stream_json(
                mock_client,
                "https://test.com",
                {"data": "test"},
                {"Authorization": "Bearer test"},
                retry_config=retry_config,
            ):
                chunks.append(chunk)

        # Should only have one successful chunk before the stream fails
        assert len(chunks) == 1
        assert chunks[0] == {"text": "hello"}
        # Should only attempt connection once since stream was established
        assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_post_json_custom_retry_config() -> None:
    """Test that post_json accepts custom retry configuration objects."""
    with patch("llm_async.utils.http.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_client.post.return_value = mock_response

        # Use a simple object instead of RetryConfig
        class CustomRetryConfig:
            max_attempts = 2
            base_delay = 0.1
            backoff_factor = 2.0
            max_delay = 10.0
            retry_on_status = (429, 500)
            jitter = False

        result = await post_json(
            mock_client,
            "https://test.com",
            {"data": "test"},
            {"Authorization": "Bearer test"},
            retry_config=CustomRetryConfig(),
        )

        assert result == {"key": "value"}
        mock_client.post.assert_called_once()
