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
        auto_execute_tools: bool = False,
        tool_executor: Union[dict[str, Callable[..., Any]], None] = None,
        max_tool_iterations: int = 10,
        pubsub: Union[Any, None] = None,
        **kwargs: Any,
    ) -> Response:
        if auto_execute_tools:
            if tool_executor is None:
                raise ValueError("tool_executor must be provided when auto_execute_tools=True")
            return await self._auto_execute_tools(
                model,
                messages,
                stream,
                tools,
                tool_choice,
                tool_executor,
                max_tool_iterations,
                pubsub,
                **kwargs,
            )
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

    async def _auto_execute_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        tools: Union[list[Tool], None],
        tool_choice: Union[str, dict[str, Any], None],
        tool_executor: dict[str, Callable[..., Any]],
        max_tool_iterations: int,
        pubsub: Union[Any, None] = None,
        **kwargs: Any,
    ) -> Response:
        if stream:
            raise NotImplementedError("Auto tool execution not supported with streaming")

        # Create a copy of messages to avoid modifying the original
        current_messages = messages.copy()

        for _ in range(max_tool_iterations + 1):
            # Clean messages for this provider
            provider_messages = self._clean_messages(current_messages)

            response = await self._single_complete(
                model, provider_messages, stream, tools, tool_choice, **kwargs
            )

            if response.stream:
                raise NotImplementedError("Auto tool execution not supported with streaming")

            main_resp = response.main_response
            if main_resp and main_resp.tool_calls:
                # Create assistant message with tool calls in provider-specific format
                assistant_message = self._create_assistant_message_with_tools(main_resp)
                current_messages.append(assistant_message)

                # Execute tools and append results
                tool_results = await self._execute_tools(
                    main_resp.tool_calls, tool_executor, pubsub
                )
                current_messages.extend(tool_results)
            else:
                # Final response
                return response
        raise RuntimeError(f"Exceeded maximum tool iterations ({max_tool_iterations})")

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

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        tool_executor: dict[str, Callable[..., Any]],
        pubsub: Union[Any, None] = None,
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
