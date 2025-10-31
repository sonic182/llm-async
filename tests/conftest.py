from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_aiosonic_client() -> AsyncMock:
    """Mock HTTP client with configurable responses"""
    client = AsyncMock()

    # Default successful response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
    client.post.return_value = mock_response

    return client


@pytest.fixture
def mock_tool_executor():
    """Mock tool executor for auto-execution tests"""
    return {
        "get_weather": Mock(return_value="Sunny, 72°F"),
        "calculate_total": Mock(return_value=150.50),
        "search_database": Mock(return_value={"results": ["item1", "item2"]}),
    }


@pytest.fixture
def mock_stream_chunks():
    """Mock stream chunks for different providers"""
    return {
        "openai": [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            b"data: [DONE]\n\n",
        ],
        "claude": [
            b'data: {"delta": {"text": "Hello"}}\n\n',
            b'data: {"delta": {"text": " world"}}\n\n',
            b"data: [DONE]\n\n",
        ],
        "google": [
            b'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}}]}}]}\n\n',
            b'data: {"candidates": [{"content": {"parts": [{"text": " world"}}]}}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    }


@pytest.fixture
def mock_openai_tool_call():
    """Mock OpenAI-style tool call"""
    return {
        "id": "call_123",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
    }


@pytest.fixture
def mock_claude_tool_call():
    """Mock Claude-style tool call"""
    return {
        "id": "call_123",
        "type": "tool_use",
        "name": "get_weather",
        "input": {"location": "NYC"},
    }


@pytest.fixture
def mock_google_tool_call():
    """Mock Google-style tool call"""
    return {"name": "get_weather", "args": {"location": "NYC"}}
