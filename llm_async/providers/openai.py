from collections.abc import Callable, Mapping
from typing import Any

from aiosonic import HeadersType  # type: ignore[import-untyped]

from llm_async.models import Message, Response, Tool
from llm_async.models.response_schema import ResponseSchema
from llm_async.models.tool_call import ToolCall
from llm_async.utils.http import post_json

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    BASE_URL = "https://api.openai.com/v1"

    def _format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _parse_response(self, original: dict[str, Any]) -> Message:
        choice = original["choices"][0]
        message_payload = choice["message"]
        tool_calls_data = message_payload.get("tool_calls")
        tool_calls = None
        if tool_calls_data:
            tool_calls = [
                ToolCall(id=tc.get("id", ""), type=tc.get("type", ""), function=tc.get("function"))
                for tc in tool_calls_data
            ]
        content_value = message_payload.get("content")
        if content_value is None:
            content_value = ""
        role = message_payload.get("role", "assistant")
        return Message(
            role=role,
            content=content_value,
            tool_calls=tool_calls,
            original=message_payload,
        )

    async def execute_tool(
        self, tool_call: ToolCall, tools_map: dict[str, Callable[..., Any]]
    ) -> dict[str, Any]:
        if tool_call.type == "function" and tool_call.function:
            func_name = tool_call.function.get("name")
            args = tool_call.function.get("arguments")
            if isinstance(args, str):
                import json

                args = json.loads(args)
            elif not isinstance(args, dict):
                args = {}
            tool_call_id = tool_call.id
        elif tool_call.type == "tool_use" and tool_call.name:
            func_name = tool_call.name
            args = tool_call.input
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

        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}

    async def _single_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_schema: ResponseSchema | Mapping[str, Any] | None = None,
        headers: HeadersType | None = None,
        **kwargs: Any,
    ) -> Response:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        schema_obj = ResponseSchema.coerce(response_schema)
        if schema_obj:
            payload["response_format"] = schema_obj.for_openai()

        if tools:
            payload["tools"] = self._format_tools(tools)
        if tool_choice:
            payload["tool_choice"] = tool_choice

        final_headers = self._headers_for_request(headers)

        if stream:
            return self._stream_response(
                f"{self.base_url}/chat/completions",
                payload,
                final_headers,
                "openai",
            )

        response = await post_json(
            self.client,
            f"{self.base_url}/chat/completions",
            payload,
            final_headers,
            retry_config=self.retry_config,
        )
        main_response = self._parse_response(response)
        return Response(response, self.__class__.name(), main_response)
