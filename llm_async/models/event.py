from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Event:
    """Represents a pub/sub event for tool execution."""

    topic: str
    payload: dict[str, Any]
    timestamp: datetime
