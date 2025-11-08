from __future__ import annotations

import pytest

from llm_async.models.message import Message, message_to_dict, normalize_messages
from llm_async.models.tool_call import ToolCall


def test_normalize_converts_plain_dict_message() -> None:
    messages = [{"role": "user", "content": "hello"}]

    normalized = normalize_messages(messages)

    assert len(normalized) == 1
    msg = normalized[0]
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.original == {"role": "user", "content": "hello"}


def test_normalize_preserves_parts_messages() -> None:
    parts_message = {
        "role": "assistant",
        "parts": [
            {"functionResponse": {"name": "calc", "response": {"result": "42"}}},
        ],
    }

    normalized = normalize_messages([parts_message])

    assert isinstance(normalized[0].content, list)
    payload = message_to_dict(normalized[0])
    assert "parts" in payload
    assert payload["parts"][0]["functionResponse"]["response"]["result"] == "42"


def test_message_to_dict_includes_tool_calls() -> None:
    tool_call = ToolCall(id="1", type="function", function={"name": "lookup", "arguments": {}})
    message = Message(role="assistant", content="working", tool_calls=[tool_call])

    payload = message_to_dict(message)

    assert payload["tool_calls"][0]["function"]["name"] == "lookup"


def test_normalize_allows_none_content() -> None:
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "noop", "arguments": {}},
                    }
                ],
            }
        ]
    )

    assert normalized[0].content == ""
    assert normalized[0].tool_calls and normalized[0].tool_calls[0].id == "call_1"


def test_normalize_rejects_invalid_role() -> None:
    with pytest.raises(ValueError):
        normalize_messages([{"role": "model", "content": "hi"}])
