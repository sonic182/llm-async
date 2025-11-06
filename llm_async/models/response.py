from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    id: str
    type: str
    name: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    function: Optional[dict[str, Any]] = None


@dataclass
class MainResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    original_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    original: dict[str, Any]
    provider_name: str
    main_response: MainResponse
    stream: bool = False
    stream_generator: Optional[AsyncIterator["StreamChunk"]] = None

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

    content: Optional[str]
    original: dict[str, Any]
