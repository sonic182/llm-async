from dataclasses import dataclass
from typing import Any, Union

from .event import Event
from .response import Response, StreamChunk

__all__ = ["Tool", "Response", "StreamChunk", "Event"]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Union[dict[str, Any], None] = None
    input_schema: Union[dict[str, Any], None] = None
