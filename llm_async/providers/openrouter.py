from typing import Any, Optional, Union

from llm_async.utils.http import post_json
from llm_async.utils.retry import RetryConfig  # type: ignore

from ..models import Response, Tool
from .openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        retry_config: Optional[RetryConfig] = None,
    ):
        super().__init__(api_key, base_url, retry_config)

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

        # Add optional OpenRouter-specific headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if "http_referer" in kwargs:
            headers["HTTP-Referer"] = kwargs.pop("http_referer")
        if "x_title" in kwargs:
            headers["X-Title"] = kwargs.pop("x_title")

        if stream:
            return self._stream_response(
                f"{self.base_url}/chat/completions", payload, headers, "openai"
            )

        response = await post_json(
            self.client,
            f"{self.base_url}/chat/completions",
            payload,
            headers,
            retry_config=self.retry_config,
        )
        return Response(response, self.__class__.name())
