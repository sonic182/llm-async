from unittest.mock import Mock

import pytest

from llmpy.models.response import ToolCall
from llmpy.providers.claude import ClaudeProvider
from llmpy.providers.google import GoogleProvider
from llmpy.providers.openai import OpenAIProvider
from llmpy.providers.openrouter import OpenRouterProvider


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

    results = await provider._execute_tools(tool_calls, mock_tool_executor)

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

    results = await provider._execute_tools(tool_calls, mock_tool_executor)

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

    results = await provider._execute_tools(tool_calls, mock_tool_executor)

    # Google should return tool results in "tool" role format with function name
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert results[0]["tool_call_id"] == "call_123"
    assert results[0]["name"] == "get_weather"  # Google includes function name
    assert results[0]["content"] == "Sunny, 72°F"
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

    results = await provider._execute_tools(tool_calls, mock_tool_executor)

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

    results = await provider._execute_tools(tool_calls, mock_tool_executor)

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

    with pytest.raises(ValueError, match="Tool non_existent_tool not found in tool_executor"):
        await provider._execute_tools(tool_calls, tool_executor)


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

    results = await provider._execute_tools(tool_calls, mock_tool_executor)

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

    # Unknown tool types should be skipped
    results = await provider._execute_tools(tool_calls, tool_executor)
    assert len(results) == 0  # Should skip unknown tool types


@pytest.mark.asyncio
async def test_tool_execution_empty_tool_calls(mock_tool_executor) -> None:
    """Test tool execution with empty tool calls list"""
    provider = OpenAIProvider(api_key="test_key")

    # Empty tool calls should return empty results
    results = await provider._execute_tools([], mock_tool_executor)
    assert len(results) == 0
