from unittest.mock import Mock

import pytest

from llm_async.models.tool_call import ToolCall
from llm_async.providers.claude import ClaudeProvider
from llm_async.providers.google import GoogleProvider
from llm_async.providers.openai import OpenAIProvider
from llm_async.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
async def test_tool_execution_openai_format(mock_tool_executor) -> None:
    """Test OpenAI tool execution format"""
    provider = OpenAIProvider(api_key="test_key")

    # Mock OpenAI-style tool call
    tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function={"name": "get_weather", "arguments": '{"location": "NYC"}'},
        )
    ]

    results = []
    for tc in tool_calls:
        res = await provider.execute_tool(tc, mock_tool_executor)
        results.append(res)

    # OpenAI should return tool results in "tool" role format
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert results[0]["tool_call_id"] == "call_123"
    assert results[0]["content"] == "Sunny, 72°F"
    mock_tool_executor["get_weather"].assert_called_once_with(location="NYC")


@pytest.mark.asyncio
async def test_tool_execution_claude_format(mock_tool_executor) -> None:
    """Test Claude tool execution format"""
    provider = ClaudeProvider(api_key="test_key")

    # Mock Claude-style tool call
    tool_calls = [
        ToolCall(id="call_123", type="tool_use", name="get_weather", input={"location": "NYC"})
    ]

    results = []
    for tc in tool_calls:
        res = await provider.execute_tool(tc, mock_tool_executor)
        results.append(res)

    # Claude should return tool results in "user" role with tool_result content
    assert len(results) == 1
    assert results[0]["role"] == "user"
    assert isinstance(results[0]["content"], list)
    assert results[0]["content"][0]["type"] == "tool_result"
    assert results[0]["content"][0]["tool_use_id"] == "call_123"
    assert results[0]["content"][0]["content"] == "Sunny, 72°F"
    mock_tool_executor["get_weather"].assert_called_once_with(location="NYC")


@pytest.mark.asyncio
async def test_tool_execution_google_format(mock_tool_executor) -> None:
    """Test Google tool execution format"""
    provider = GoogleProvider(api_key="test_key")

    # Mock Google-style tool call (similar to OpenAI)
    tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function={"name": "get_weather", "arguments": '{"location": "NYC"}'},
        )
    ]

    results = []
    for tc in tool_calls:
        res = await provider.execute_tool(tc, mock_tool_executor)
        results.append(res)

    # Google should return tool results in a parts-based format with functionResponse
    assert len(results) == 1
    parts = results[0].get("parts", [])
    assert len(parts) == 1
    func_resp = parts[0].get("functionResponse", {})
    assert func_resp.get("name") == "get_weather"
    assert func_resp.get("response", {}).get("result") == "Sunny, 72°F"
    mock_tool_executor["get_weather"].assert_called_once_with(location="NYC")


@pytest.mark.asyncio
async def test_tool_execution_openrouter_format(mock_tool_executor) -> None:
    """Test OpenRouter tool execution format"""
    provider = OpenRouterProvider(api_key="test_key")

    # Mock OpenRouter-style tool call (inherits from OpenAI)
    tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function={"name": "get_weather", "arguments": '{"location": "NYC"}'},
        )
    ]

    results = []
    for tc in tool_calls:
        res = await provider.execute_tool(tc, mock_tool_executor)
        results.append(res)

    # OpenRouter should return tool results in "tool" role format (like OpenAI)
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert results[0]["tool_call_id"] == "call_123"
    assert results[0]["content"] == "Sunny, 72°F"
    mock_tool_executor["get_weather"].assert_called_once_with(location="NYC")


@pytest.mark.asyncio
async def test_tool_execution_multiple_tools(mock_tool_executor) -> None:
    """Test tool execution with multiple tools"""
    provider = OpenAIProvider(api_key="test_key")

    # Mock multiple tool calls
    tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function={"name": "get_weather", "arguments": '{"location": "NYC"}'},
        ),
        ToolCall(
            id="call_124",
            type="function",
            function={"name": "calculate_total", "arguments": '{"items": [10, 20, 30]}'},
        ),
    ]

    results = []
    for tc in tool_calls:
        res = await provider.execute_tool(tc, mock_tool_executor)
        results.append(res)

    # Should execute both tools and return results
    assert len(results) == 2
    assert results[0]["tool_call_id"] == "call_123"
    assert results[1]["tool_call_id"] == "call_124"
    mock_tool_executor["get_weather"].assert_called_once_with(location="NYC")
    mock_tool_executor["calculate_total"].assert_called_once_with(items=[10, 20, 30])


@pytest.mark.asyncio
async def test_tool_execution_missing_tool() -> None:
    """Test tool execution with missing tool in executor"""
    provider = OpenAIProvider(api_key="test_key")

    # Mock tool call for a tool that doesn't exist in executor
    tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function={"name": "non_existent_tool", "arguments": '{"param": "value"}'},
        )
    ]

    tool_executor = {"get_weather": Mock(return_value="Sunny")}  # Missing non_existent_tool

    with pytest.raises(ValueError, match="Tool non_existent_tool not found"):
        # call execute_tool for the failing tool
        await provider.execute_tool(tool_calls[0], tool_executor)


@pytest.mark.asyncio
async def test_tool_execution_json_argument_parsing(mock_tool_executor) -> None:
    """Test JSON argument parsing in tool execution"""
    provider = OpenAIProvider(api_key="test_key")

    # Mock tool call with JSON string arguments
    tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function={
                "name": "get_weather",
                "arguments": '{"location": "NYC", "unit": "fahrenheit"}',
            },
        )
    ]

    results = []
    for tc in tool_calls:
        res = await provider.execute_tool(tc, mock_tool_executor)
        results.append(res)

    # Should parse JSON arguments correctly
    assert len(results) == 1
    mock_tool_executor["get_weather"].assert_called_once_with(location="NYC", unit="fahrenheit")


@pytest.mark.asyncio
async def test_tool_execution_unknown_tool_type() -> None:
    """Test tool execution with unknown tool type"""
    provider = OpenAIProvider(api_key="test_key")

    # Mock tool call with unknown type
    tool_calls = [
        ToolCall(
            id="call_123",
            type="unknown_type",
            function={"name": "get_weather", "arguments": '{"location": "NYC"}'},
        )
    ]

    tool_executor = {"get_weather": Mock(return_value="Sunny")}

    # Unknown tool types should raise an error when executed
    with pytest.raises(Exception, match="no tool defined"):
        await provider.execute_tool(tool_calls[0], tool_executor)


@pytest.mark.asyncio
async def test_tool_execution_empty_tool_calls(mock_tool_executor) -> None:
    """Test tool execution with empty tool calls list"""
    provider = OpenAIProvider(api_key="test_key")

    # Empty tool calls should return empty results (no execution)
    results = []
    for tc in []:
        res = await provider.execute_tool(tc, mock_tool_executor)
        results.append(res)
    assert len(results) == 0
