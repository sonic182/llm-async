from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_async import tool
from llm_async.agent import Agent
from llm_async.models import Message, Response
from llm_async.models.tool_call import ToolCall


def _make_response(tool_calls: list[ToolCall] | None, original: dict | None = None) -> Response:
    msg = Message(role="assistant", content="", tool_calls=tool_calls)
    msg.original = original or {"role": "assistant", "content": ""}
    resp = MagicMock(spec=Response)
    resp.main_response = msg
    return resp


def _make_tool_call(name: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, type="function", function={"name": name, "arguments": "{}"})


@tool
def dummy_tool(x: str) -> str:
    """A dummy tool."""
    return f"result:{x}"


@pytest.fixture
def provider() -> AsyncMock:
    p = AsyncMock()
    p.execute_tool = AsyncMock(return_value={"role": "tool", "content": "ok"})
    return p


@pytest.fixture
def agent(provider: AsyncMock) -> Agent:
    return Agent(provider=provider, model="gpt-4o-mini", tools=[dummy_tool])


@pytest.mark.asyncio
async def test_no_tool_calls(agent: Agent, provider: AsyncMock) -> None:
    provider.acomplete.return_value = _make_response(tool_calls=None)
    messages = [{"role": "user", "content": "hello"}]

    response = await agent.acomplete(messages)

    provider.acomplete.assert_called_once()
    provider.execute_tool.assert_not_called()
    assert response.main_response.tool_calls is None


@pytest.mark.asyncio
async def test_single_tool_call(agent: Agent, provider: AsyncMock) -> None:
    tc = _make_tool_call("dummy_tool")
    assistant_original = {"role": "assistant", "tool_calls": []}
    first = _make_response([tc], original=assistant_original)
    second = _make_response(tool_calls=None)
    provider.acomplete.side_effect = [first, second]

    messages = [{"role": "user", "content": "do something"}]
    response = await agent.acomplete(messages)

    assert provider.acomplete.call_count == 2
    assert provider.execute_tool.call_count == 1
    assert response.main_response.tool_calls is None

    second_call_messages = provider.acomplete.call_args_list[1][1]["messages"]
    assert assistant_original in second_call_messages
    assert {"role": "tool", "content": "ok"} in second_call_messages


@pytest.mark.asyncio
async def test_parallel_execution(provider: AsyncMock) -> None:
    agent = Agent(provider=provider, model="m", tools=[dummy_tool], parallel_tools=True)
    tc1 = _make_tool_call("dummy_tool", "c1")
    tc2 = _make_tool_call("dummy_tool", "c2")
    first = _make_response([tc1, tc2])
    second = _make_response(tool_calls=None)
    provider.acomplete.side_effect = [first, second]

    with patch(
        "llm_async.agent.asyncio.gather", wraps=__import__("asyncio").gather
    ) as mock_gather:
        await agent.acomplete([{"role": "user", "content": "go"}])
        mock_gather.assert_called_once()

    assert provider.execute_tool.call_count == 2


@pytest.mark.asyncio
async def test_sequential_execution(provider: AsyncMock) -> None:
    agent = Agent(provider=provider, model="m", tools=[dummy_tool], parallel_tools=False)
    tc1 = _make_tool_call("dummy_tool", "c1")
    tc2 = _make_tool_call("dummy_tool", "c2")
    first = _make_response([tc1, tc2])
    second = _make_response(tool_calls=None)
    provider.acomplete.side_effect = [first, second]

    with patch("llm_async.agent.asyncio.gather") as mock_gather:
        await agent.acomplete([{"role": "user", "content": "go"}])
        mock_gather.assert_not_called()

    assert provider.execute_tool.call_count == 2


@pytest.mark.asyncio
async def test_max_iter_safeguard(provider: AsyncMock) -> None:
    tc = _make_tool_call("dummy_tool")
    provider.acomplete.return_value = _make_response([tc])

    agent = Agent(provider=provider, model="m", tools=[dummy_tool], max_iter=3)
    await agent.acomplete([{"role": "user", "content": "loop"}])

    assert provider.acomplete.call_count == 3


@pytest.mark.asyncio
async def test_kwargs_forwarded(agent: Agent, provider: AsyncMock) -> None:
    provider.acomplete.return_value = _make_response(tool_calls=None)

    await agent.acomplete(
        [{"role": "user", "content": "hi"}],
        tool_choice="required",
        temperature=0.5,
    )

    call_kwargs = provider.acomplete.call_args[1]
    assert call_kwargs["tool_choice"] == "required"
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_tools_passed_to_provider(agent: Agent, provider: AsyncMock) -> None:
    provider.acomplete.return_value = _make_response(tool_calls=None)

    await agent.acomplete([{"role": "user", "content": "hi"}])

    call_kwargs = provider.acomplete.call_args[1]
    assert call_kwargs["tools"] == [dummy_tool.tool]
