from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from llm_async.models.message import Message


@dataclass
class Response:
    original: dict[str, Any]
    provider_name: str
    main_response: Message | None = None
    stream: bool = False
    stream_generator: AsyncIterator[StreamChunk] | None = None

    async def stream_content(self) -> AsyncIterator[str]:
        """Convenience method to iterate over stream content"""
        if self.stream and self.stream_generator:
            async for chunk in self.stream_generator:
                if chunk.content:
                    yield chunk.content


@dataclass
class StreamChunk:
    """Represents a single chunk from a streaming response.

    - `content`: the textual content of the chunk (may be None)
    - `original`: the original parsed JSON chunk from the provider
    """

    content: str | None
    original: dict[str, Any]
