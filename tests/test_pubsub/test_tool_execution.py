import asyncio

import pytest

from llmpy.models.response import ToolCall
from llmpy.providers.claude import ClaudeProvider
from llmpy.providers.openai import OpenAIProvider
from llmpy.pubsub import LocalQueueBackend, PubSub


@pytest.mark.asyncio
async def test_openai_execute_tools_with_pubsub():
    """Test that OpenAI provider emits pub/sub events during tool execution."""
    backend = LocalQueueBackend()
    pubsub = PubSub(backend)

    provider = OpenAIProvider(api_key="test-key")

    # Create a simple tool
    def simple_calculator(a: int, b: int) -> int:
        return a + b

    tool_executor = {"calculator": simple_calculator}

    # Create tool calls
    tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function={"name": "calculator", "arguments": '{"a": 5, "b": 3}'},
        )
    ]

    # Collect events
    received_events = []

    async def collect_events():
        async for event in pubsub.subscribe("tools.openai.*"):
            received_events.append(event)
            if len(received_events) >= 3:  # start, complete, and possibly error
                break

    asyncio.create_task(collect_events())
    await asyncio.sleep(0.1)

    # Execute tools
    results = await provider._execute_tools(tool_calls, tool_executor, pubsub)

    await asyncio.sleep(0.1)  # Wait for events to be processed
    await pubsub.close()

    # Verify results
    assert len(results) == 1
    assert results[0]["content"] == "8"

    # Verify events
    assert len(received_events) >= 2

    # Check start event
    start_events = [e for e in received_events if "start" in e.topic]
    assert len(start_events) >= 1
    assert start_events[0].payload["tool_name"] == "calculator"

    # Check complete event
    complete_events = [e for e in received_events if "complete" in e.topic]
    assert len(complete_events) >= 1
    assert complete_events[0].payload["result"] == "8"


@pytest.mark.asyncio
async def test_claude_execute_tools_with_pubsub():
    """Test that Claude provider emits pub/sub events during tool execution."""
    backend = LocalQueueBackend()
    pubsub = PubSub(backend)

    provider = ClaudeProvider(api_key="test-key")

    # Create a simple tool
    def simple_calculator(expression: str) -> str:
        return str(eval(expression))

    tool_executor = {"calculate": simple_calculator}

    # Create tool calls (Claude format)
    tool_calls = [
        ToolCall(
            id="tool_123",
            type="tool_use",
            name="calculate",
            input={"expression": "2 + 2"},
        )
    ]

    # Collect events
    received_events = []

    async def collect_events():
        async for event in pubsub.subscribe("tools.claude.*"):
            received_events.append(event)
            if len(received_events) >= 2:
                break

    asyncio.create_task(collect_events())
    await asyncio.sleep(0.1)

    # Execute tools
    results = await provider._execute_tools(tool_calls, tool_executor, pubsub)

    await asyncio.sleep(0.1)
    await pubsub.close()

    # Verify results
    assert len(results) == 1
    assert results[0]["content"][0]["content"] == "4"

    # Verify events
    assert len(received_events) >= 2


@pytest.mark.asyncio
async def test_tool_execution_error_events():
    """Test that errors emit error events."""
    backend = LocalQueueBackend()
    pubsub = PubSub(backend)

    provider = OpenAIProvider(api_key="test-key")

    # Create a faulty tool
    def faulty_tool() -> str:
        raise ValueError("Tool execution failed")

    tool_executor = {"faulty": faulty_tool}

    tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function={"name": "faulty", "arguments": "{}"},
        )
    ]

    # Collect events
    received_events = []

    async def collect_events():
        async for event in pubsub.subscribe("tools.openai.*"):
            received_events.append(event)
            if len(received_events) >= 2:
                break

    asyncio.create_task(collect_events())
    await asyncio.sleep(0.1)

    # Execute tools (should raise)
    with pytest.raises(ValueError, match="Tool execution failed"):
        await provider._execute_tools(tool_calls, tool_executor, pubsub)

    await asyncio.sleep(0.1)
    await pubsub.close()

    # Verify error events were emitted
    error_events = [e for e in received_events if "error" in e.topic]
    assert len(error_events) >= 1
    assert "Tool execution failed" in error_events[0].payload["error"]


@pytest.mark.asyncio
async def test_tool_not_found_error_events():
    """Test that tool not found errors emit error events."""
    backend = LocalQueueBackend()
    pubsub = PubSub(backend)

    provider = OpenAIProvider(api_key="test-key")

    tool_executor = {}  # Empty executor

    tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function={"name": "unknown_tool", "arguments": "{}"},
        )
    ]

    # Collect events
    received_events = []

    async def collect_events():
        async for event in pubsub.subscribe("tools.openai.*"):
            received_events.append(event)
            if len(received_events) >= 1:
                break

    asyncio.create_task(collect_events())
    await asyncio.sleep(0.1)

    # Execute tools (should raise ValueError for tool not found)
    with pytest.raises(ValueError, match="Tool unknown_tool not found"):
        await provider._execute_tools(tool_calls, tool_executor, pubsub)

    await asyncio.sleep(0.1)
    await pubsub.close()

    # Verify error events
    error_events = [e for e in received_events if "error" in e.topic]
    assert len(error_events) >= 1
    assert "unknown_tool not found" in error_events[0].payload["error"]


@pytest.mark.asyncio
async def test_execute_tools_without_pubsub():
    """Test that tool execution works without pubsub (backward compatibility)."""
    provider = OpenAIProvider(api_key="test-key")

    def simple_calculator(a: int, b: int) -> int:
        return a + b

    tool_executor = {"calculator": simple_calculator}

    tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function={"name": "calculator", "arguments": '{"a": 10, "b": 5}'},
        )
    ]

    # Execute without pubsub (should work fine)
    results = await provider._execute_tools(tool_calls, tool_executor, pubsub=None)

    assert len(results) == 1
    assert results[0]["content"] == "15"


@pytest.mark.asyncio
async def test_multiple_tools_execution_with_pubsub():
    """Test execution of multiple tools with pubsub events."""
    backend = LocalQueueBackend()
    pubsub = PubSub(backend)

    provider = OpenAIProvider(api_key="test-key")

    def add(a: int, b: int) -> int:
        return a + b

    def multiply(a: int, b: int) -> int:
        return a * b

    tool_executor = {"add": add, "multiply": multiply}

    tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function={"name": "add", "arguments": '{"a": 2, "b": 3}'},
        ),
        ToolCall(
            id="call_2",
            type="function",
            function={"name": "multiply", "arguments": '{"a": 4, "b": 5}'},
        ),
    ]

    # Collect events
    received_events = []

    async def collect_events():
        async for event in pubsub.subscribe("tools.openai.*"):
            received_events.append(event)
            if len(received_events) >= 4:  # 2 tools * (start + complete)
                break

    asyncio.create_task(collect_events())
    await asyncio.sleep(0.1)

    # Execute tools
    results = await provider._execute_tools(tool_calls, tool_executor, pubsub)

    await asyncio.sleep(0.1)
    await pubsub.close()

    # Verify results
    assert len(results) == 2
    assert results[0]["content"] == "5"
    assert results[1]["content"] == "20"

    # Verify events
    assert len(received_events) >= 4

    # Verify both tools appear in events
    tool_names = {e.payload.get("tool_name") for e in received_events}
    assert "add" in tool_names
    assert "multiply" in tool_names
