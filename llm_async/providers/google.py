from typing import Any, Callable, Optional, Union

from llm_async.utils.http import post_json
from llm_async.utils.retry import RetryConfig  # type: ignore

from ..models import Response, Tool
from ..models.response import MainResponse, ToolCall
from .base import BaseProvider


class GoogleProvider(BaseProvider):
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        vertex_config: Optional[dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Google Provider supporting both Google AI API and Vertex AI.

        Args:
            api_key: API key for Google AI API (ignored if vertex_config provided)
            base_url: Base URL for API (auto-detected if vertex_config provided)
            vertex_config: Configuration for Vertex AI with keys:
                - project_id: Google Cloud project ID
                - location_id: Location/region (e.g., "us-central1", "global")
                - api_endpoint: Optional custom API endpoint
                - goth_token: Optional Bearer token for Vertex AI auth
        """
        self.vertex_config = vertex_config

        if vertex_config:
            if not base_url:
                base_url = self._build_vertex_url(vertex_config)
            self.bearer_token = vertex_config.get("goth_token", "")
        else:
            # Google AI API mode
            if not base_url:
                base_url = "https://generativelanguage.googleapis.com/v1beta/models/"

        super().__init__(api_key, base_url, retry_config)

    def _clean_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned_messages: list[dict[str, Any]] = []
        for message in messages:
            cleaned_message = message.copy()
            if message.get("role") == "assistant":
                cleaned_message["role"] = "model"
            elif message.get("role") == "system":
                # Google handles system messages separately
                continue
            elif message.get("role") == "tool":
                cleaned_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": message.get("tool_call_id", ""),
                            "name": message.get("name", ""),
                            "content": message.get("content", ""),
                        }
                    ],
                }
            cleaned_messages.append(cleaned_message)
        return cleaned_messages

    def _format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        formatted_tools = []
        for tool in tools:
            formatted_tool = {"name": tool.name, "description": tool.description or ""}
            if tool.parameters:
                formatted_tool["parameters"] = tool.parameters  # type: ignore
            elif tool.input_schema:
                formatted_tool["parameters"] = tool.input_schema  # type: ignore
            formatted_tools.append(formatted_tool)
        return [{"functionDeclarations": formatted_tools}]

    def _parse_response(self, original: dict[str, Any]) -> MainResponse:
        candidates = original.get("candidates", [])
        if not candidates:
            return MainResponse(
                content=None, tool_calls=None, original_data={"role": "model", "parts": []}
            )
        candidate = candidates[0]
        content_data = candidate.get("content", {})
        parts = content_data.get("parts", [])
        text_content = ""
        tool_calls: list[ToolCall] = []
        for part in parts:
            if "text" in part:
                text_content += part.get("text", "")
            elif "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=f"call_{len(tool_calls)}",
                        type="function",
                        function={
                            "name": func_call.get("name", ""),
                            "arguments": func_call.get("args", {}),
                        },
                    )
                )
        message = {
            "role": candidate.get("role", "model"),
            "parts": parts,
        }
        return MainResponse(
            content=text_content or None,
            tool_calls=tool_calls or None,
            original_data=message,
        )

    def _create_assistant_message_with_tools(self, main_resp: MainResponse) -> dict[str, Any]:
        if (
            main_resp.tool_calls
            and len(main_resp.tool_calls) > 0
            and main_resp.tool_calls[0].function
        ):
            func = main_resp.tool_calls[0].function
            name = str(func.get("name", ""))
            args = func.get("arguments")
            if not isinstance(args, dict):
                args = {}
            return {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]}
        else:
            return {"role": "model", "parts": [{"text": main_resp.content or ""}]}

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

        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": func_name,
                        "response": {"result": str(result)},
                    }
                }
            ],
        }

    async def _single_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: Union[list[Tool], None] = None,
        tool_choice: Union[str, dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Response:
        """Send completion request to Google's Gemini API."""
        payload = self._build_request_payload(messages, tools, **kwargs)
        headers = self._build_headers()

        # For Google AI API, the base URL already includes the model path
        # So we just need to append the action suffix
        suffix = "generateContent"
        if stream:
            suffix = "streamGenerateContent?alt=sse"

        url = f"{self.base_url}{model}:{suffix}"

        if stream:
            return self._stream_response(url, payload, headers, "google")

        response = await post_json(
            self.client, url, payload, headers, retry_config=self.retry_config
        )
        main_response = self._parse_response(response)
        return Response(response, self.__class__.name(), main_response)

    def _build_request_payload(
        self,
        messages: list[dict[str, Any]],
        tools: Union[list[Tool], None],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request payload for Google's Gemini API."""
        contents = self._format_messages(messages)

        # Handle structured outputs
        response_schema = kwargs.pop("response_schema", None)

        payload = {
            "contents": contents,
            **kwargs,
        }

        # Handle system message
        if messages and messages[0]["role"] == "system":
            system_content = messages[0]["content"]
            payload["system_instruction"] = {"parts": [{"text": system_content}]}
            # Remove system message from contents (contents already excludes system messages)
            # No need to modify contents further since _format_messages handles this

        # Handle tools/function calling
        if tools:
            payload["tools"] = self._format_tools(tools)

        if response_schema:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": self._remove_additional_properties(response_schema),
            }

        return payload

    def _format_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert standard message format to Google's format.

        Notes:
        - If a message already contains provider-formatted `parts`, pass them through.
        - Convert `tool_calls` to `functionCall` parts.
        - Convert tool result blocks to `functionResponse` with recommended structure.
        """
        formatted_messages = []

        for message in messages:
            if message.get("role") == "system":
                # System messages are handled separately in the request
                continue

            # Google uses 'model' instead of 'assistant'
            role = "user" if message.get("role") in ["user", "system"] else "model"

            # 1) If the message is already in Google's format, keep the parts as-is
            if "parts" in message and isinstance(message["parts"], list):
                formatted_messages.append({"role": role, "parts": message["parts"]})
                continue

            # Handle different content formats
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")

            if tool_calls:
                # Handle assistant messages with tool calls for Google
                parts: list[dict[str, Any]] = []
                for tc in tool_calls:
                    if tc.get("type") == "function" and tc.get("function"):
                        parts.append(
                            {
                                "functionCall": {
                                    "name": tc["function"].get("name", ""),
                                    "args": tc["function"].get("arguments", {}) or {},
                                }
                            }
                        )
                formatted_messages.append({"role": role, "parts": parts})
                continue

            if isinstance(content, list):
                # Handle complex content blocks (like tool results)
                parts = []
                for block in content:
                    if block.get("type") == "text":
                        parts.append({"text": block.get("text", "")})
                    elif block.get("type") == "tool_result":
                        # Convert tool result to function response for Google
                        func_name = block.get("name") or block.get("tool_call_id", "")
                        result_text = str(block.get("content", ""))
                        # Use the simple functionResponse shape expected by the client examples
                        parts.append(
                            {
                                "functionResponse": {
                                    "name": func_name,
                                    "response": {"result": result_text},
                                }
                            }
                        )
                formatted_messages.append({"role": role, "parts": parts})
                continue

            # Simple text content
            formatted_messages.append({"role": role, "parts": [{"text": str(content)}]})

        return formatted_messages

    def _build_headers(self) -> dict[str, str]:
        """Build headers based on authentication mode."""
        headers = {"Content-Type": "application/json"}

        if self.vertex_config:
            token = self.bearer_token
            if not token:
                raise ValueError("Bearer token required for Vertex AI authentication")
            headers["Authorization"] = f"Bearer {token}"
        else:
            headers["X-GOOG-API-KEY"] = self.api_key

        return headers

    def _build_vertex_url(self, vertex_config: dict[str, Any]) -> str:
        """Build the Vertex AI API URL."""
        project_id = vertex_config.get("project_id")
        location_id = vertex_config.get("location_id", "global")

        if not project_id:
            raise ValueError("project_id is required for Vertex AI configuration")

        # Handle custom endpoint
        if "api_endpoint" in vertex_config:
            return vertex_config["api_endpoint"]

        # Build standard Vertex AI endpoint
        if location_id == "global":
            endpoint = "aiplatform.googleapis.com"
        else:
            endpoint = f"{location_id}-aiplatform.googleapis.com"

        return f"https://{endpoint}/v1/projects/{project_id}/locations/{location_id}/publishers/google/models"

    def _remove_additional_properties(self, schema: Any) -> Any:
        """Recursively remove 'additionalProperties' from schema (Google API doesn't support it)."""
        if isinstance(schema, dict):
            # Remove additionalProperties and process nested values
            cleaned = {}
            for key, value in schema.items():
                if key not in ["additionalProperties", "additional_properties"]:
                    cleaned[key] = self._remove_additional_properties(value)
            return cleaned
        elif isinstance(schema, list):
            # Process list elements
            return [self._remove_additional_properties(item) for item in schema]
        else:
            # Return other types unchanged
            return schema
