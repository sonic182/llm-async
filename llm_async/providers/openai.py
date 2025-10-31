from typing import Any, Callable, Optional, Union

from llm_async.utils.http import post_json
from llm_async.utils.retry import RetryConfig  # type: ignore

from ..models import Response, Tool
from ..models.response import MainResponse, ToolCall
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        retry_config: Optional[RetryConfig] = None,
    ):
        super().__init__(api_key, base_url, retry_config)

    def _clean_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # OpenAI/OpenRouter support 'tool' role
        return messages

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

    def _parse_response(self, original: dict[str, Any]) -> MainResponse:
        message = original["choices"][0]["message"]
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

    def _create_assistant_message_with_tools(self, main_resp: MainResponse) -> dict[str, Any]:
        tool_calls_dict = (
            [
                {"id": tc.id, "type": tc.type, "function": tc.function}
                for tc in (main_resp.tool_calls or [])
            ]
            if main_resp.tool_calls
            else []
        )
        return {
            "role": "assistant",
            "content": main_resp.content or "",
            "tool_calls": tool_calls_dict,
        }

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
                    {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}
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
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        # Handle structured outputs
        if "response_schema" in payload:
            schema = payload.pop("response_schema")
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }

        if tools:
            payload["tools"] = self._format_tools(tools)
        if tool_choice:
            payload["tool_choice"] = tool_choice

        if stream:
            return self._stream_response(
                f"{self.base_url}/chat/completions",
                payload,
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                "openai",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = await post_json(
            self.client,
            f"{self.base_url}/chat/completions",
            payload,
            headers,
            retry_config=self.retry_config,
        )
        return Response(response, self.__class__.name())
