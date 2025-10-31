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
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    original_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    original: dict[str, Any]
    provider_name: str
    stream: bool = False
    stream_generator: Optional[AsyncIterator["StreamChunk"]] = None
    main_response: Optional[MainResponse] = None

    def __post_init__(self):
        if not self.stream:
            self.main_response = self._extract_main_response(self.provider_name)

    def _extract_main_response(self, provider_name: str) -> MainResponse:
        # This method is kept for backward compatibility but should be refactored
        # to use provider instances directly in the future
        if provider_name in ["openai", "openrouter"]:
            message = self.original["choices"][0]["message"]
            tool_calls_data = message.get("tool_calls")
            tool_calls = None
            if tool_calls_data:
                tool_calls = [
                    ToolCall(id=tc["id"], type=tc["type"], function=tc.get("function"))
                    for tc in tool_calls_data
                ]
            return MainResponse(
                content=message.get("content"),
                tool_calls=tool_calls,
                original_data={"content": message.get("content"), "tool_calls": tool_calls_data},
            )
        elif provider_name == "claude":
            content_blocks = self.original["content"]
            text_content = ""
            tool_calls = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block["text"]
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block["id"],
                            type="tool_use",
                            name=block["name"],
                            input=block["input"],
                        )
                    )
                elif "text" in block:
                    text_content += block["text"]
            return MainResponse(
                content=text_content or None,
                tool_calls=tool_calls or None,
                original_data={
                    "content": text_content or None,
                    "tool_calls": (
                        [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "name": tc.name,
                                "input": tc.input,
                            }
                            for tc in tool_calls
                        ]
                        if tool_calls
                        else None
                    ),
                },
            )
        elif provider_name == "google":
            candidates = self.original.get("candidates", [])
            if not candidates:
                return MainResponse(
                    content=None,
                    tool_calls=None,
                    original_data={"content": None, "tool_calls": None},
                )

            candidate = candidates[0]
            content_data = candidate.get("content", {})
            parts = content_data.get("parts", [])

            text_content = ""
            tool_calls = []

            for part in parts:
                if "text" in part:
                    text_content += part["text"]
                elif "functionCall" in part:
                    func_call = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{len(tool_calls)}",
                            type="function",
                            function={
                                "name": func_call.get("name", ""),
                                "arguments": func_call.get("args", {}),
                            },
                        )
                    )

            return MainResponse(
                content=text_content or None,
                tool_calls=tool_calls or None,
                original_data={
                    "content": text_content or None,
                    "tool_calls": (
                        [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": tc.function,
                            }
                            for tc in tool_calls
                        ]
                        if tool_calls
                        else None
                    ),
                },
            )
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

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
