from typing import Any, Callable, Optional, Union

import aiosonic  # type: ignore[import-untyped]

from ..models import Response, Tool
from ..models.response import MainResponse, ToolCall
from ..utils.http import parse_stream_chunk, post_json, stream_json
from ..utils.retry import RetryConfig  # type: ignore


class BaseProvider:
    def __init__(
        self, api_key: str, base_url: str = "", retry_config: Optional[RetryConfig] = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.client = aiosonic.HTTPClient()
        self.retry_config = retry_config

    @classmethod
    def name(cls) -> str:
        return cls.__name__.replace("Provider", "").lower()

    async def acomplete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: Union[list[Tool], None] = None,
        tool_choice: Union[str, dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Response:
        return await self._single_complete(model, messages, stream, tools, tool_choice, **kwargs)

    async def _single_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: Union[list[Tool], None] = None,
        tool_choice: Union[str, dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Response:
        raise NotImplementedError

    def _clean_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean messages to be compatible with the current provider."""
        raise NotImplementedError

    def _format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Format tools for the current provider."""
        raise NotImplementedError

    def _parse_response(self, original: dict[str, Any]) -> MainResponse:
        """Parse provider response into standard format."""
        raise NotImplementedError

    def _create_assistant_message_with_tools(self, main_resp: MainResponse) -> dict[str, Any]:
        """Create an assistant message with tool calls in the format expected by this provider."""
        raise NotImplementedError

    async def execute_tool(
        self, tool_call: ToolCall, tools_map: dict[str, Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        """Execute tools and return results in provider-specific format."""
        raise NotImplementedError

    async def _http_post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return await post_json(
            self.client,
            f"{self.base_url}{endpoint}",
            data,
            headers,
            retry_config=self.retry_config,
        )

    def _stream_response(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], parse_provider: str
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
