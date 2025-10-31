from typing import Any

import pytest

from llmpy.models.response import Response


def test_response_openai_simple() -> None:
    raw = {"choices": [{"message": {"content": "Hello!"}}]}
    response = Response(raw, "openai")
    assert response.original == raw
    assert response.main_response is not None
    assert response.main_response.content == "Hello!"


def test_response_openai_with_tools() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": "The weather is sunny.",
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
    response = Response(raw, "openai")
    assert response.original == raw
    assert response.main_response is not None
    assert response.main_response.content == "The weather is sunny."
    assert response.main_response.tool_calls is not None
    assert len(response.main_response.tool_calls) == 1
    tc = response.main_response.tool_calls[0]
    assert tc.id == "call_123"
    assert tc.type == "function"
    assert tc.function == {"name": "get_weather", "arguments": '{"location": "NYC"}'}


def test_response_claude_simple() -> None:
    raw = {"content": [{"type": "text", "text": "Hello!"}]}
    response = Response(raw, "claude")
    assert response.original == raw
    assert response.main_response is not None
    assert response.main_response.content == "Hello!"


def test_response_claude_with_tools() -> None:
    raw = {
        "content": [
            {"type": "text", "text": "The weather is sunny."},
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "get_weather",
                "input": {"location": "NYC"},
            },
        ]
    }
    response = Response(raw, "claude")
    assert response.original == raw
    assert response.main_response is not None
    assert response.main_response.content == "The weather is sunny."
    assert response.main_response.tool_calls is not None
    assert len(response.main_response.tool_calls) == 1
    tc = response.main_response.tool_calls[0]
    assert tc.id == "call_123"
    assert tc.type == "tool_use"
    assert tc.name == "get_weather"
    assert tc.input == {"location": "NYC"}


def test_response_unsupported_provider() -> None:
    raw: dict[str, Any] = {}
    with pytest.raises(ValueError, match="Unsupported provider"):
        Response(raw, "unknown")
