from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ToolCall:
    id: str
    type: str
    name: str | None = None
    input: dict[str, Any] | None = None
    function: dict[str, Any] | None = None

    def to_responses_api_message(self) -> dict[str, Any]:
        """Convert to OpenAI Responses API format message."""
        if self.type == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": self.id,
                "output": self.function.get("output", "")
                if isinstance(self.function, dict)
                else str(self.input or ""),
            }
        return {}

    @staticmethod
    def from_responses_api_function_call(
        fc_id: str, call_id: str, name: str, arguments: str
    ) -> ToolCall:
        """Create ToolCall from Responses API function call."""
        return ToolCall(
            id=fc_id,
            type="function",
            name=name,
            function={"name": name, "arguments": arguments},
            input={"call_id": call_id},
        )

    @staticmethod
    def function_call_output(call_id: str, output: str) -> ToolCall:
        """Create a function call output ToolCall for Responses API."""
        return ToolCall(
            id=call_id,
            type="function_call_output",
            name=None,
            input=None,
            function={"output": output},
        )
