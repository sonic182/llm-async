from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    type: str
    name: str | None = None
    input: dict[str, Any] | None = None
    function: dict[str, Any] | None = None
