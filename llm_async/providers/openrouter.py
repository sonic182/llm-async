from typing import Any

from llm_async.models import Response, Tool
from llm_async.utils.http import post_json

from .openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    BASE_URL = "https://openrouter.ai/api/v1"

    async def _single_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
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
        main_response = self._parse_response(response)
        return Response(response, self.__class__.name(), main_response)
