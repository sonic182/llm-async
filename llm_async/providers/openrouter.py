from collections.abc import Mapping
from typing import Any

from aiosonic import HeadersType  # type: ignore[import-untyped]
from llm_async.models import Response, Tool
from llm_async.models.response_schema import ResponseSchema
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
        response_schema: ResponseSchema | Mapping[str, Any] | None = None,
        headers: HeadersType | None = None,
        **kwargs: Any,
    ) -> Response:
        request_kwargs = dict(kwargs)
        http_referer = request_kwargs.pop("http_referer", None)
        x_title = request_kwargs.pop("x_title", None)

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **request_kwargs,
        }

        schema_obj = ResponseSchema.coerce(response_schema)
        if schema_obj:
            payload["response_format"] = schema_obj.for_openai()

        if tools:
            payload["tools"] = self._format_tools(tools)
        if tool_choice:
            payload["tool_choice"] = tool_choice

        final_headers = self._headers_for_request(headers)
        if http_referer:
            final_headers["HTTP-Referer"] = http_referer
        if x_title:
            final_headers["X-Title"] = x_title

        if stream:
            return self._stream_response(
                f"{self.base_url}/chat/completions", payload, final_headers, "openai"
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
