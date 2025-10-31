from typing import Any, Callable, Optional, Union

from llm_async.utils.http import post_json
from llm_async.utils.retry import RetryConfig  # type: ignore

from ..models import Response, Tool
from ..models.response import MainResponse, ToolCall
from .base import BaseProvider

DEFAULT_MAX_TOKENS = 8192


class ClaudeProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        retry_config: Optional[RetryConfig] = None,
    ):
        super().__init__(api_key, base_url, retry_config)

    def _clean_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned_messages: list[dict[str, Any]] = []
        for message in messages:
            cleaned_message = message.copy()
            if message["role"] == "tool":
                if isinstance(message.get("content"), str):
                    cleaned_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.get("tool_call_id", ""),
                                "content": message["content"],
                            }
                        ],
                    }
                elif isinstance(message.get("content"), list):
                    cleaned_message["role"] = "user"
            elif message["role"] not in ["user", "assistant", "system"]:
                continue
            cleaned_messages.append(cleaned_message)
        return cleaned_messages

    def _format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in tools
        ]

    def _parse_response(self, original: dict[str, Any]) -> MainResponse:
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
        return MainResponse(
            content=text_content or None,
            tool_calls=tool_calls or None,
            original_data={
                "content": text_content or None,
                "tool_calls": (
                    [
                        {"id": tc.id, "type": tc.type, "name": tc.name, "input": tc.input}
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None
                ),
            },
        )

    def _create_assistant_message_with_tools(self, main_resp: MainResponse) -> dict[str, Any]:
        content_blocks = []
        if main_resp.content:
            content_blocks.append({"type": "text", "text": main_resp.content})
        if main_resp.tool_calls:
            for tc in main_resp.tool_calls:
                content_blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                )
        return {"role": "assistant", "content": content_blocks}

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        tool_executor: dict[str, Callable[..., Any]],
        pubsub: Union[Any, None] = None,
    ) -> list[dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
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
                continue

            if not func_name:
                continue

            if func_name not in tool_executor:
                error_msg = f"Tool {func_name} not found in tool_executor"
                if pubsub:
                    await pubsub.publish(
                        f"tools.{self.name()}.{func_name}.error",
                        {
                            "call_id": tool_call_id,
                            "tool_name": func_name,
                            "error": error_msg,
                        },
                    )
                raise ValueError(error_msg)

            try:
                if pubsub:
                    await pubsub.publish(
                        f"tools.{self.name()}.{func_name}.start",
                        {"call_id": tool_call_id, "tool_name": func_name, "args": args},
                    )

                if isinstance(args, dict):
                    result = tool_executor[func_name](**args)
                else:
                    result = tool_executor[func_name](args)

                if pubsub:
                    await pubsub.publish(
                        f"tools.{self.name()}.{func_name}.complete",
                        {"call_id": tool_call_id, "tool_name": func_name, "result": str(result)},
                    )

                results.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": str(result),
                            }
                        ],
                    }
                )
            except Exception as e:
                if pubsub:
                    await pubsub.publish(
                        f"tools.{self.name()}.{func_name}.error",
                        {
                            "call_id": tool_call_id,
                            "tool_name": func_name,
                            "error": str(e),
                        },
                    )
                raise
        return results

    async def _single_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: Union[list[Tool], None] = None,
        tool_choice: Union[str, dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Response:
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": messages,
        }

        # Handle structured outputs
        if "response_schema" in kwargs:
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
            return Response(response, self.__class__.name())
