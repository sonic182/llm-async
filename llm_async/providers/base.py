from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import aiosonic  # type: ignore[import-untyped]
from aiosonic import HeadersType  # type: ignore[import-untyped]

from llm_async.models import Message, Response, Tool
from llm_async.models.message import message_to_dict, normalize_messages, validate_messages
from llm_async.models.response_schema import ResponseSchema
from llm_async.models.tool_call import ToolCall
from llm_async.utils.http import parse_stream_chunk, post_json, stream_json
from llm_async.utils.retry import RetryConfig  # type: ignore


class BaseProvider:
    BASE_URL = None

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        retry_config: RetryConfig | None = None,
        client_kwargs: dict | None = None,
        http2: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self.client = aiosonic.HTTPClient(**(client_kwargs or {}))
        self.http2 = http2
        self.retry_config = retry_config

    @classmethod
    def name(cls) -> str:
        return cls.__name__.replace("Provider", "").lower()

    async def acomplete(
        self,
        model: str,
        messages: Sequence[Message | Mapping[str, Any]],
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_schema: ResponseSchema | Mapping[str, Any] | None = None,
        headers: HeadersType | None = None,
        **kwargs: Any,
    ) -> Response:
        normalized_messages = normalize_messages(messages)
        validate_messages(normalized_messages)
        serialized_messages = self._serialize_messages(normalized_messages)
        return await self._single_complete(
            model,
            serialized_messages,
            stream,
            tools,
            tool_choice,
            response_schema=response_schema,
            headers=headers,
            **kwargs,
        )

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
        raise NotImplementedError

    def _serialize_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        return [message_to_dict(message) for message in messages]

    def _format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Format tools for the current provider."""
        raise NotImplementedError

    def _parse_response(self, original: dict[str, Any]) -> Message:
        """Parse provider response into standard format."""
        raise NotImplementedError

    async def execute_tool(
        self, tool_call: ToolCall, tools_map: dict[str, Callable[..., Any]]
    ) -> dict[str, Any]:
        """Execute tools and return results in provider-specific format."""
        raise NotImplementedError

    def _default_headers(self) -> HeadersType:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _headers_for_request(self, headers: HeadersType | None = None) -> dict[str, str]:
        merged: dict[str, str] = dict(self._default_headers())
        if not headers:
            return merged
        if isinstance(headers, dict):
            merged.update(headers)
            return merged
        for key, value in headers:
            merged[key] = value
        return merged

    async def request(
        self,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
        path: str,
        json_data: dict[str, Any] | None = None,
        headers: HeadersType | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a request to the provider's API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: API endpoint path (e.g., "/v1/models")
            json_data: JSON body for POST/PUT/PATCH requests
            **kwargs: Additional headers to include

        Returns:
            The JSON response from the API
        """
        import json

        final_headers = self._headers_for_request(headers)
        for key, value in kwargs.items():
            final_headers[key] = str(value)
        url = f"{self.base_url}{path}"

        if method == "GET":
            response = await self.client.get(url, headers=final_headers)
        elif method == "POST":
            response = await self.client.post(url, headers=final_headers, json=json_data or {})
        elif method == "PUT":
            response = await self.client.put(url, headers=final_headers, json=json_data or {})
        elif method == "DELETE":
            response = await self.client.delete(url, headers=final_headers)
        elif method == "PATCH":
            response = await self.client.patch(url, headers=final_headers, json=json_data or {})
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        if isinstance(response, dict):
            return response

        text = await response.text()
        return json.loads(text)

    async def _http_post(
        self, endpoint: str, data: dict[str, Any], headers: HeadersType | None = None
    ) -> dict[str, Any]:
        final_headers = self._headers_for_request(headers)
        return await post_json(
            self.client,
            f"{self.base_url}{endpoint}",
            data,
            final_headers,
            retry_config=self.retry_config,
        )

    def _stream_response(
        self, url: str, payload: dict[str, Any], headers: HeadersType, parse_provider: str
    ):
        """Return a Response with a stream_generator using shared stream logic."""

        async def _gen():
            async for chunk in stream_json(
                self.client, url, payload, headers, retry_config=self.retry_config
            ):
                sc = parse_stream_chunk(chunk, parse_provider)
                if sc and sc.content is not None:
                    yield sc

        return Response({}, self.__class__.name(), stream=True, stream_generator=_gen())
