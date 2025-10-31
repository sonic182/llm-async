from unittest.mock import AsyncMock, patch

import pytest

from llm_async.models import Response, Tool
from llm_async.providers.claude import ClaudeProvider
from llm_async.providers.openai import OpenAIProvider
from llm_async.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
async def test_auto_tool_execution_openai_single_tool(mock_tool_executor) -> None:
    """Test auto tool execution with OpenAI provider - single tool call"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client

        # Mock first API call with tool call
        mock_response1 = AsyncMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Mock second API call with final response
        mock_response2 = AsyncMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": "The weather in NYC is sunny and 72°F."}}]
        }

        mock_client.post.side_effect = [mock_response1, mock_response2]

        provider = OpenAIProvider(api_key="test_key")
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        )

        result = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            tools=[tool],
            auto_execute_tools=True,
            tool_executor=mock_tool_executor,
        )

        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.content and "sunny and 72°F" in result.main_response.content
        assert mock_client.post.call_count == 2
        mock_tool_executor["get_weather"].assert_called_once_with(location="NYC")


@pytest.mark.asyncio
async def test_auto_tool_execution_openai_multi_iteration(mock_tool_executor) -> None:
    """Test auto tool execution with OpenAI provider - multiple iterations"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client

        # Mock first API call with tool call
        mock_response1 = AsyncMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll get the weather first.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Mock second API call with another tool call
        mock_response2 = AsyncMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Now let me calculate something.",
                        "tool_calls": [
                            {
                                "id": "call_124",
                                "type": "function",
                                "function": {
                                    "name": "calculate_total",
                                    "arguments": '{"items": [10, 20, 30]}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Mock third API call with final response
        mock_response3 = AsyncMock()
        mock_response3.status_code = 200
        mock_response3.json.return_value = {
            "choices": [{"message": {"content": "All done! Weather is sunny and total is 60."}}]
        }

        mock_client.post.side_effect = [mock_response1, mock_response2, mock_response3]

        provider = OpenAIProvider(api_key="test_key")
        tools = [
            Tool(
                name="get_weather",
                description="Get current weather",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            ),
            Tool(
                name="calculate_total",
                description="Calculate total",
                parameters={"type": "object", "properties": {"items": {"type": "array"}}},
            ),
        ]

        result = await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Get weather and calculate total"}],
            tools=tools,
            auto_execute_tools=True,
            tool_executor=mock_tool_executor,
        )

        assert isinstance(result, Response)
        assert mock_client.post.call_count == 3
        mock_tool_executor["get_weather"].assert_called_once_with(location="NYC")
        mock_tool_executor["calculate_total"].assert_called_once_with(items=[10, 20, 30])


@pytest.mark.asyncio
async def test_auto_tool_execution_claude_single_tool(mock_tool_executor) -> None:
    """Test auto tool execution with Claude provider - single tool call"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client

        # Mock first API call with tool call
        mock_response1 = AsyncMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "content": [
                {"type": "text", "text": "I'll check the weather."},
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"location": "NYC"},
                },
            ]
        }

        # Mock second API call with final response
        mock_response2 = AsyncMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "content": [{"type": "text", "text": "The weather in NYC is sunny and 72°F."}]
        }

        mock_client.post.side_effect = [mock_response1, mock_response2]

        provider = ClaudeProvider(api_key="test_key")
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        )

        result = await provider.acomplete(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            tools=[tool],
            auto_execute_tools=True,
            tool_executor=mock_tool_executor,
            max_tokens=1024,
        )

        assert isinstance(result, Response)
        assert not result.stream
        assert result.main_response is not None
        assert result.main_response.content and "sunny and 72°F" in result.main_response.content
        assert mock_client.post.call_count == 2
        mock_tool_executor["get_weather"].assert_called_once_with(location="NYC")


@pytest.mark.asyncio
async def test_auto_tool_execution_openrouter_with_headers(mock_tool_executor) -> None:
    """Test auto tool execution with OpenRouter provider - preserves custom headers"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client

        # Mock first API call with tool call
        mock_response1 = AsyncMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll check the weather.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Mock second API call with final response
        mock_response2 = AsyncMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "choices": [{"message": {"content": "The weather in NYC is sunny and 72°F."}}]
        }

        mock_client.post.side_effect = [mock_response1, mock_response2]

        provider = OpenRouterProvider(api_key="test_key")
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        )

        result = await provider.acomplete(
            model="openrouter/model",
            messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            tools=[tool],
            auto_execute_tools=True,
            tool_executor=mock_tool_executor,
            http_referer="https://example.com",
            x_title="My App",
        )

        assert isinstance(result, Response)
        assert mock_client.post.call_count == 2

        # Verify custom headers were preserved in both calls
        calls = mock_client.post.call_args_list
        for call in calls:
            headers = call[1]["headers"]
            assert headers["HTTP-Referer"] == "https://example.com"
            assert headers["X-Title"] == "My App"


@pytest.mark.asyncio
async def test_auto_tool_execution_max_iterations(mock_tool_executor) -> None:
    """Test auto tool execution respects max_tool_iterations limit"""
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client

        # Mock API calls that always return tool calls (infinite loop scenario)
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll keep calling tools.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Return the same response for all calls
        mock_client.post.return_value = mock_response

        provider = OpenAIProvider(api_key="test_key")
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        )

        with pytest.raises(RuntimeError, match="Exceeded maximum tool iterations"):
            await provider.acomplete(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=[tool],
                auto_execute_tools=True,
                tool_executor=mock_tool_executor,
                max_tool_iterations=3,  # Low limit to trigger error
            )

        # Should have made exactly max_tool_iterations + 1 calls (initial + iterations)
        assert mock_client.post.call_count == 4


@pytest.mark.asyncio
async def test_auto_tool_execution_streaming_not_supported(mock_tool_executor) -> None:
    """Test that auto tool execution raises error when streaming is enabled"""
    provider = OpenAIProvider(api_key="test_key")
    tool = Tool(name="get_weather", description="Get current weather")

    with pytest.raises(
        NotImplementedError, match="Auto tool execution not supported with streaming"
    ):
        await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[tool],
            stream=True,
            auto_execute_tools=True,
            tool_executor=mock_tool_executor,
        )


@pytest.mark.asyncio
async def test_auto_tool_execution_missing_tool_executor() -> None:
    """Test that auto tool execution requires tool_executor"""
    provider = OpenAIProvider(api_key="test_key")
    tool = Tool(name="get_weather", description="Get current weather")

    with pytest.raises(
        ValueError, match="tool_executor must be provided when auto_execute_tools=True"
    ):
        await provider.acomplete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[tool],
            auto_execute_tools=True,
        )
