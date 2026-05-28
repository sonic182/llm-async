import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from llm_async.models import Message, Response, Tool
from llm_async.providers.base import BaseProvider


class Agent:
    """Run provider completions with decorated tools until a final response is produced.

    The agent passes its tools to the provider, executes requested tool calls, appends
    tool results to the message list, and repeats until the model stops requesting
    tools or ``max_iter`` is reached.
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        tools: list[Callable[..., Any]],
        parallel_tools: bool = True,
        max_iter: int = 10,
    ) -> None:
        """Create an agent for a provider, model, and list of decorated tool functions."""
        self.provider = provider
        self.model = model
        self.parallel_tools = parallel_tools
        self.max_iter = max_iter
        self._tool_list: list[Tool] = [f.tool for f in tools]  # type: ignore[attr-defined]
        self._tools_map: dict[str, Callable[..., Any]] = {
            f.tool.name: f
            for f in tools  # type: ignore[attr-defined]
        }

    async def acomplete(
        self,
        messages: Sequence[Message | Mapping[str, Any]],
        **kwargs: Any,
    ) -> Response:
        """Complete a conversation, automatically executing any tool calls.

        Additional keyword arguments are forwarded to the provider's ``acomplete`` call.
        """
        msgs: list[Any] = list(messages)
        response: Response | None = None

        for _ in range(self.max_iter):
            response = await self.provider.acomplete(
                model=self.model,
                messages=msgs,
                tools=self._tool_list,
                **kwargs,
            )
            tool_calls = response.main_response.tool_calls  # type: ignore[union-attr]
            if not tool_calls:
                return response

            msgs.append(response.main_response.original)  # type: ignore[union-attr]

            if self.parallel_tools:
                results = list(
                    await asyncio.gather(
                        *[self.provider.execute_tool(tc, self._tools_map) for tc in tool_calls]
                    )
                )
            else:
                results = [
                    await self.provider.execute_tool(tc, self._tools_map) for tc in tool_calls
                ]

            msgs.extend(results)

        return response  # type: ignore[return-value]
