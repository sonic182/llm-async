import pytest

from llmpy.providers.claude import ClaudeProvider
from llmpy.providers.google import GoogleProvider
from llmpy.providers.openai import OpenAIProvider
from llmpy.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
async def test_message_cleaning_claude_tool_results() -> None:
    """Test Claude message cleaning for tool results"""
    provider = ClaudeProvider(api_key="test_key")

    # Test tool result conversion to Claude format
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "I'll check the weather.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72°F"},
    ]

    cleaned_messages = provider._clean_messages(messages)

    # Claude should convert tool results to user messages with tool_result content
    assert len(cleaned_messages) == 3
    assert cleaned_messages[0]["role"] == "user"
    assert cleaned_messages[1]["role"] == "assistant"
    assert cleaned_messages[2]["role"] == "user"
    assert isinstance(cleaned_messages[2]["content"], list)
    assert cleaned_messages[2]["content"][0]["type"] == "tool_result"
    assert cleaned_messages[2]["content"][0]["tool_use_id"] == "call_123"
    assert cleaned_messages[2]["content"][0]["content"] == "Sunny, 72°F"


@pytest.mark.asyncio
async def test_message_cleaning_google_tool_results() -> None:
    """Test Google message cleaning for tool results"""
    provider = GoogleProvider(api_key="test_key")

    # Test tool result conversion to Google format
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "I'll check the weather.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "get_weather",
            "content": "Sunny, 72°F",
        },
    ]

    cleaned_messages = provider._clean_messages(messages)

    # Google should convert tool results to user messages with function response
    assert len(cleaned_messages) == 3
    assert cleaned_messages[0]["role"] == "user"
    assert cleaned_messages[1]["role"] == "model"  # Google uses "model" for assistant
    assert cleaned_messages[2]["role"] == "user"
    assert isinstance(cleaned_messages[2]["content"], list)
    assert cleaned_messages[2]["content"][0]["type"] == "tool_result"
    assert cleaned_messages[2]["content"][0]["tool_call_id"] == "call_123"
    assert cleaned_messages[2]["content"][0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_message_cleaning_openai_tool_results() -> None:
    """Test OpenAI message cleaning for tool results"""
    provider = OpenAIProvider(api_key="test_key")

    # Test tool result handling for OpenAI (should remain unchanged)
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "I'll check the weather.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72°F"},
    ]

    cleaned_messages = provider._clean_messages(messages)

    # OpenAI should keep tool messages as-is
    assert cleaned_messages == messages


@pytest.mark.asyncio
async def test_message_cleaning_openrouter_tool_results() -> None:
    """Test OpenRouter message cleaning for tool results"""
    provider = OpenRouterProvider(api_key="test_key")

    # Test tool result handling for OpenRouter (should remain unchanged like OpenAI)
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "I'll check the weather.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72°F"},
    ]

    cleaned_messages = provider._clean_messages(messages)

    # OpenRouter should keep tool messages as-is (inherits from OpenAI)
    assert cleaned_messages == messages


@pytest.mark.asyncio
async def test_message_cleaning_system_messages() -> None:
    """Test system message handling across providers"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    # Test OpenAI (should keep system messages)
    openai_provider = OpenAIProvider(api_key="test_key")
    openai_cleaned = openai_provider._clean_messages(messages)
    assert openai_cleaned == messages

    # Test Claude (should keep system messages)
    claude_provider = ClaudeProvider(api_key="test_key")
    claude_cleaned = claude_provider._clean_messages(messages)
    assert claude_cleaned == messages

    # Test Google (should handle system messages separately)
    google_provider = GoogleProvider(api_key="test_key")
    google_cleaned = google_provider._clean_messages(messages)
    assert len(google_cleaned) == 1  # System message should be handled separately
    assert google_cleaned[0]["role"] == "user"


@pytest.mark.asyncio
async def test_message_cleaning_complex_claude_format() -> None:
    """Test complex Claude message format with mixed content"""
    provider = ClaudeProvider(api_key="test_key")

    # Test complex Claude format with tool results already in Claude format
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check the weather."},
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"location": "NYC"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_123", "content": "Sunny, 72°F"}
            ],
        },
    ]

    cleaned_messages = provider._clean_messages(messages)

    # Claude should keep already-formatted messages as-is
    assert cleaned_messages == messages


@pytest.mark.asyncio
async def test_message_cleaning_unsupported_roles() -> None:
    """Test handling of unsupported message roles"""
    provider = ClaudeProvider(api_key="test_key")

    # Test messages with unsupported roles for Claude
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "system", "content": "Be helpful"},
        {"role": "tool", "content": "Result"},  # Should be converted
        {"role": "function", "content": "Function result"},  # Unsupported, should be skipped
    ]

    cleaned_messages = provider._clean_messages(messages)

    # Claude should skip unsupported roles and convert tool messages
    assert len(cleaned_messages) == 4  # function role should be skipped
    assert cleaned_messages[3]["role"] == "user"  # tool role converted to user
