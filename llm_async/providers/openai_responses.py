from collections.abc import Callable, Mapping, Sequence
from typing import Any

from llm_async.models import Message, Response, Tool
from llm_async.models.response import StreamChunk
from llm_async.models.response_schema import ResponseSchema
from llm_async.models.tool_call import ToolCall
from llm_async.utils.http import post_json, stream_json
from llm_async.utils.retry import RetryConfig  # type: ignore

from .base import BaseProvider


def _normalize_responses_messages(
    messages: Sequence[Message | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, Message):
            msg_dict = {
                "role": message.role,
                "content": message.content,
            }
            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "name": tc.name,
                        "function": tc.function,
                    }
                    for tc in message.tool_calls
                    if tc.function
                ]
            normalized.append(msg_dict)
        elif isinstance(message, Mapping):
            normalized.append(dict(message))
        else:
            raise TypeError("Message entries must be Message or mapping")
    return normalized


class OpenAIResponsesProvider(BaseProvider):
    BASE_URL = "https://api.openai.com/v1"

    async def acomplete(
        self,
        model: str,
        messages: Sequence[Message | Mapping[str, Any]],
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_schema: ResponseSchema | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Response:
        normalized_messages = _normalize_responses_messages(messages)
        return await self._single_complete(
            model,
            normalized_messages,
            stream,
            tools,
            tool_choice,
            response_schema=response_schema,
            **kwargs,
        )

    def _messages_to_input(self, messages: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if (
            len(messages) == 1
            and messages[0].get("role") == "user"
            and isinstance(messages[0].get("content"), str)
        ):
            return messages[0]["content"]

        responses_messages = []
        for msg in messages:
            msg_type = msg.get("type")

            if msg_type == "function_call_output":
                responses_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("call_id", ""),
                        "output": msg.get("output", ""),
                    }
                )
            else:
                role = msg.get("role")
                if role == "user":
                    responses_messages.append(
                        {
                            "role": "user",
                            "content": msg.get("content", ""),
                        }
                    )
                elif role == "assistant":
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            fc_id = tc.get("id", "")
                            call_id = tc.get("id", "")
                            tc_input = tc.get("input", {})
                            if isinstance(tc_input, dict) and "call_id" in tc_input:
                                call_id = tc_input["call_id"]
                            responses_messages.append(
                                {
                                    "type": "function_call",
                                    "id": fc_id,
                                    "call_id": call_id,
                                    "name": tc.get("function", {}).get("name")
                                    if isinstance(tc.get("function"), dict)
                                    else tc.get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments")
                                    if isinstance(tc.get("function"), dict)
                                    else "",
                                }
                            )
                    else:
                        responses_messages.append(
                            {
                                "role": "assistant",
                                "content": msg.get("content", ""),
                            }
                        )

        return responses_messages if responses_messages else messages

    def _parse_response(self, original: dict[str, Any]) -> Message:
        text: str = ""
        tool_calls: list[ToolCall] | None = None

        if isinstance(original, dict):
            output_items = original.get("output")
            if isinstance(output_items, list) and output_items:
                parts: list[str] = []
                parsed_tool_calls: list[ToolCall] = []
                for item in output_items:
                    if not isinstance(item, dict):
                        continue

                    if item.get("type") == "function_call":
                        name = item.get("name")
                        arguments = item.get("arguments")
                        fc_id = item.get("id", "")
                        call_id = item.get("call_id", "")
                        parsed_tool_calls.append(
                            ToolCall.from_responses_api_function_call(
                                fc_id=fc_id or call_id,
                                call_id=call_id,
                                name=name or "",
                                arguments=arguments or "",
                            )
                        )
                        continue

                    content = item.get("content")
                    if not isinstance(content, list):
                        continue
                    for entry in content:
                        if not isinstance(entry, dict):
                            continue
                        entry_type = entry.get("type")
                        if entry_type in {"output_text", "text"} and isinstance(
                            entry.get("text"), str
                        ):
                            parts.append(entry.get("text") or "")
                            continue
                        if entry_type == "tool_call":
                            name = entry.get("name")
                            arguments = entry.get("arguments")
                            tool_call_id = entry.get("id", "")
                            parsed_tool_calls.append(
                                ToolCall(
                                    id=tool_call_id,
                                    type="function",
                                    name=name,
                                    function={"name": name, "arguments": arguments},
                                )
                            )
                if parts:
                    text = "".join(parts)
                if parsed_tool_calls:
                    tool_calls = parsed_tool_calls
            elif isinstance(original.get("output_text"), str):
                text = original.get("output_text", "")
            elif isinstance(original.get("choices"), list):
                # Fallback to Chat format if server returned chat-like payload
                choice = original["choices"][0]
                message_payload = choice.get("message", {})
                text = message_payload.get("content") or ""
                tcs = message_payload.get("tool_calls")
                if isinstance(tcs, list) and tcs:
                    tool_calls = [
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", ""),
                            name=tc.get("function", {}).get("name")
                            if isinstance(tc.get("function"), dict)
                            else None,
                            function=tc.get("function"),
                        )
                        for tc in tcs
                    ]

        message_payload: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            message_payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function,
                }
                for tc in tool_calls
                if tc.function
            ]

        return Message(
            role="assistant", content=text, tool_calls=tool_calls, original=message_payload
        )

    def _format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        # Responses API expects name at top-level, not nested under "function"
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": True,
            }
            for tool in tools
        ]

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
            call_id = (tool_call.input or {}).get("call_id") if tool_call.input else None
            tool_call_id = call_id or tool_call.id
        elif tool_call.type == "tool_use" and tool_call.name:
            func_name = tool_call.name
            args = tool_call.input or {}
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

        return ToolCall.function_call_output(
            call_id=tool_call_id, output=str(result)
        ).to_responses_api_message()

    def _stream_responses_request(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> Response:
        async def _gen():
            async for chunk in stream_json(
                self.client,
                url,
                payload,
                headers,
                retry_config=self.retry_config,
            ):
                if not isinstance(chunk, dict):
                    continue
                delta_text = self._extract_stream_text(chunk)
                if delta_text:
                    yield StreamChunk(delta_text, chunk)

        return Response({}, self.__class__.name(), stream=True, stream_generator=_gen())

    @staticmethod
    def _extract_stream_text(chunk: dict[str, Any]) -> str | None:
        chunk_type = chunk.get("type")
        if isinstance(chunk_type, str) and chunk_type.endswith(".delta"):
            delta = chunk.get("delta")
            if isinstance(delta, str) and delta:
                return delta
            if isinstance(delta, dict):
                text_value = delta.get("text")
                if isinstance(text_value, str) and text_value:
                    return text_value
                output_text = delta.get("output_text")
                if isinstance(output_text, list):
                    # Some Responses events wrap output_text array
                    return "".join(part for part in output_text if isinstance(part, str)) or None
            content = chunk.get("content")
            if isinstance(content, str) and content:
                return content
        return None

    @staticmethod
    def _normalize_tool_choice(
        tool_choice: str | dict[str, Any],
    ) -> str | dict[str, Any]:
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                if "name" in tool_choice and isinstance(tool_choice["name"], str):
                    return {"type": "function", "name": tool_choice["name"]}
                function_obj = tool_choice.get("function")
                if isinstance(function_obj, dict) and isinstance(function_obj.get("name"), str):
                    return {"type": "function", "name": function_obj["name"]}
        return tool_choice

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
        previous_response_id = kwargs.pop("previous_response_id", None)

        payload: dict[str, Any] = {
            "model": model,
            "stream": stream,
            **kwargs,
        }

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        payload["input"] = self._messages_to_input(messages)

        schema_obj = ResponseSchema.coerce(response_schema)
        if schema_obj:
            payload["text"] = {"format": schema_obj.for_openai_responses()}

        if tools:
            payload["tools"] = self._format_tools(tools)
        if tool_choice:
            payload["tool_choice"] = self._normalize_tool_choice(tool_choice)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if stream:
            return self._stream_responses_request(
                f"{self.base_url}/responses",
                payload,
                headers,
            )

        response = await post_json(
            self.client,
            f"{self.base_url}/responses",
            payload,
            headers,
            retry_config=self.retry_config,
        )
        main_response = self._parse_response(response)
        return Response(response, self.__class__.name(), main_response)
