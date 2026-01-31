import asyncio
import json
import random
from collections.abc import AsyncIterator
from typing import Any

import aiosonic  # type: ignore[import-untyped]
from aiosonic import HeadersType  # type: ignore[import-untyped]

from ..models import StreamChunk
from .retry import DEFAULT_RETRY_EXCEPTIONS, RetryConfig, retry_http  # type: ignore


async def _post_json_core(
    client: aiosonic.HTTPClient,
    url: str,
    json_data: dict[str, Any],
    headers: HeadersType,
) -> dict[str, Any]:
    """Core POST JSON function without retry logic."""
    response = await client.post(url, json=json_data, headers=headers)
    if response.status_code >= 400:
        text = await response.text()
        raise Exception(f"HTTP {response.status_code}: {text}")
    return await response.json()  # type: ignore[no-any-return]


async def post_json(
    client: aiosonic.HTTPClient,
    url: str,
    json_data: dict[str, Any],
    headers: HeadersType,
    retry_config: object | None = None,
) -> dict[str, Any]:
    """POST JSON with optional retry logic.

    `retry_config` may be an instance of RetryConfig or a simple object with
    attributes matching RetryConfig fields.
    """
    # Convert simple object to RetryConfig if needed
    if retry_config is not None and not isinstance(retry_config, RetryConfig):
        config = RetryConfig()
        if hasattr(retry_config, "max_attempts"):
            config.max_attempts = getattr(retry_config, "max_attempts", 1)
        if hasattr(retry_config, "base_delay"):
            config.base_delay = getattr(retry_config, "base_delay", 1.0)
        if hasattr(retry_config, "backoff_factor"):
            config.backoff_factor = getattr(retry_config, "backoff_factor", 2.0)
        if hasattr(retry_config, "max_delay"):
            config.max_delay = getattr(retry_config, "max_delay", 60.0)
        if hasattr(retry_config, "retry_on_status"):
            config.retry_on_status = getattr(
                retry_config, "retry_on_status", (429, 500, 502, 503, 504)
            )
        if hasattr(retry_config, "retry_on_exceptions"):
            config.retry_on_exceptions = getattr(
                retry_config, "retry_on_exceptions", DEFAULT_RETRY_EXCEPTIONS
            )
        if hasattr(retry_config, "jitter"):
            config.jitter = getattr(retry_config, "jitter", True)
        retry_config = config

    # Apply retry decorator
    decorated_func = retry_http(retry_config)(_post_json_core)
    return await decorated_func(client, url, json_data, headers)


async def stream_json(
    client: aiosonic.HTTPClient,
    url: str,
    json_data: dict[str, Any],
    headers: HeadersType,
    retry_config: RetryConfig | None = RetryConfig(),
) -> AsyncIterator[dict[str, Any]]:
    """POST `json_data` to `url` and yield parsed JSON objects from SSE-style stream.

    `retry_config` behaves like in `post_json` and controls retries for the
    initial request only. Once the stream is established, this function does
    not attempt to reconnect.
    """
    # Backwards-compatibility for tests that call with client=None
    if client is None:
        yield {}
        return

    last_exc: Exception | None = None
    response = None

    # Handle case where retry_config might be None by creating default instance
    retry_config = retry_config or RetryConfig()

    # Determine expected exceptions
    retry_exceptions = DEFAULT_RETRY_EXCEPTIONS
    if retry_config.retry_on_exceptions:
        retry_exceptions = retry_config.retry_on_exceptions

    for attempt in range(retry_config.max_attempts):
        try:
            response = await client.post(url, json=json_data, headers=headers)
            if response.status_code >= 400:
                text = await response.text()
                err = Exception(f"HTTP {response.status_code}: {text}")
                if (
                    response.status_code in retry_config.retry_on_status
                    and attempt < retry_config.max_attempts - 1
                ):
                    last_exc = err
                    delay = retry_config.base_delay * (retry_config.backoff_factor**attempt)
                    delay = min(delay, retry_config.max_delay)
                    if retry_config.jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    await asyncio.sleep(delay)
                    continue
                raise err
            break
        except retry_exceptions as e:
            # expected transient error: retry
            last_exc = e
            if attempt == retry_config.max_attempts - 1:
                raise
            delay = retry_config.base_delay * (retry_config.backoff_factor**attempt)
            delay = min(delay, retry_config.max_delay)
            if retry_config.jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            await asyncio.sleep(delay)
        except Exception:
            # unexpected error, re-raise
            raise

    if response is None:
        if last_exc:
            raise last_exc
        raise Exception("stream_json failed to obtain a response")

    buffer = ""

    # aiosonic Response.read_chunks yields bytes-ish chunks asynchronously
    async for raw_chunk in response.read_chunks():
        if not raw_chunk:
            continue
        # raw_chunk may be bytes or str depending on client behavior
        if isinstance(raw_chunk, bytes):
            try:
                chunk_text = raw_chunk.decode("utf-8")
            except Exception:
                chunk_text = raw_chunk.decode("latin-1")
        else:
            chunk_text = str(raw_chunk)

        buffer += chunk_text

        # Process complete SSE events separated by double-newline
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        # Skip invalid JSON chunks
                        continue

    # Process any remaining buffered data after stream end
    if buffer:
        for line in buffer.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    # Skip invalid JSON chunks
                    continue


def parse_stream_chunk(chunk: dict[str, Any], provider: str) -> StreamChunk | None:
    """Extract textual content from a streaming chunk for a given provider.

    Returns a StreamChunk holding textual content and the original chunk, or None
    if the chunk contains no text.
    """
    if not isinstance(chunk, dict):
        return None

    if provider == "openai":
        # OpenAI streaming uses: { "choices": [{ "delta": { "content": "..." } }] }
        try:
            choices = chunk.get("choices")
            if not choices:
                return None
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            return StreamChunk(content, chunk)
        except Exception:
            return None
    elif provider == "claude":
        # Hypothetical Claude streaming format: { "delta": { "text": "..." } }
        try:
            content = chunk.get("delta", {}).get("text")
            return StreamChunk(content, chunk)
        except Exception:
            return None
    elif provider == "google":
        # Google streaming format: { "candidates": [{ "content": { "parts": [{ "text": "..." }] } }] }
        try:
            candidates = chunk.get("candidates")
            if not candidates:
                return None
            content_data = candidates[0].get("content", {})
            parts = content_data.get("parts", [])
            if parts:
                # Extract text from the first part
                text = parts[0].get("text", "")
                return StreamChunk(text, chunk)
            return None
        except Exception:
            return None
    else:
        raise ValueError(f"Unsupported provider for parsing stream chunk: {provider}")
