from collections.abc import Callable, Mapping
from typing import Any

from llm_async.models import Message, Response, Tool
from llm_async.models.response_schema import ResponseSchema
from llm_async.models.tool_call import ToolCall
from llm_async.utils.http import post_json

from .base import BaseProvider

DEFAULT_MAX_TOKENS = 8192


class ClaudeProvider(BaseProvider):
    BASE_URL = "https://api.anthropic.com/v1"

    def _format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in tools
        ]

    def _parse_response(self, original: dict[str, Any]) -> Message:
        content_blocks = original.get("content", [])
        text_content = ""
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        type="tool_use",
                        name=block.get("name"),
                        input=block.get("input"),
                    )
                )
            elif "text" in block:
                text_content += block.get("text", "")
        message = {
            "role": original.get("role", "assistant"),
            "content": content_blocks,
        }
        return Message(
            role=message["role"],
            content=text_content or "",
            tool_calls=tool_calls or None,
            original=message,
        )

    async def execute_tool(
        self, tool_call: ToolCall, tools_map: dict[str, Callable[..., Any]]
    ) -> dict[str, Any]:
        if tool_call.type == "tool_use" and tool_call.name:
            func_name = tool_call.name
            args = tool_call.input
            tool_call_id = tool_call.id
        elif tool_call.type == "function" and tool_call.function:
            func_name = tool_call.function.get("name")
            args = tool_call.function.get("arguments")
            if isinstance(args, str):
                import json

                args = json.loads(args)
            elif not isinstance(args, dict):
                args = {}
            tool_call_id = tool_call.id
        else:
            raise Exception("no tool defined")

        if not func_name:
            raise Exception("no tool defined")

        if func_name not in tools_map:
            error_msg = f"Tool {func_name} not found in tools_map"
            raise ValueError(error_msg)

        if isinstance(args, dict):
            result = tools_map[func_name](**args)
        else:
            result = tools_map[func_name](args)

        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(result),
                }
            ],
        }

    async def _single_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_schema: ResponseSchema | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Response:
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": messages,
        }

        # Handle structured outputs
        if response_schema is not None:
            raise NotImplementedError(
                "Claude provider does not support structured outputs with response_schema"
            )

        # Handle system message
        if messages and messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]
            payload["system"] = system
            payload["messages"] = messages

        if tools:
            payload["tools"] = self._format_tools(tools)
        if tool_choice:
            payload["tool_choice"] = tool_choice

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        if stream:
            payload["stream"] = True
            return self._stream_response(f"{self.base_url}/messages", payload, headers, "claude")
        else:
            response = await post_json(
                self.client,
                f"{self.base_url}/messages",
                payload,
                headers,
                retry_config=self.retry_config,
            )
            main_response = self._parse_response(response)
            return Response(response, self.__class__.name(), main_response)
