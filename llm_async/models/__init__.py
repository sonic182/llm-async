from dataclasses import dataclass
from typing import Any

from llm_async.models.event import Event
from llm_async.models.message import Message
from llm_async.models.response import Response, StreamChunk
from llm_async.models.response_schema import ResponseSchema
from llm_async.models.tool_call import ToolCall

__all__ = [
    "Tool",
    "Response",
    "StreamChunk",
    "Event",
    "Message",
    "ToolCall",
    "ResponseSchema",
]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None
