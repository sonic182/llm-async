from unittest.mock import AsyncMock, patch

import pytest

from llm_async.providers.openai_responses import OpenAIResponsesProvider


@pytest.mark.asyncio
async def test_stream_function_call_populates_main_response() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def chunks():
            yield (
                b'data: {"type": "response.output_item.added", "item": '
                b'{"id": "fc_1", "type": "function_call", "status": "in_progress", '
                b'"arguments": "", "call_id": "call_abc", "name": "add_numbers"}, '
                b'"output_index": 0}\n\n'
            )
            yield (
                b'data: {"type": "response.function_call_arguments.delta", '
                b'"delta": "{\\"a\\":1,\\"b\\":2}", "item_id": "fc_1"}\n\n'
            )
            yield (
                b'data: {"type": "response.function_call_arguments.done", '
                b'"arguments": "{\\"a\\":1,\\"b\\":2}", "item_id": "fc_1"}\n\n'
            )
            yield (
                b'data: {"type": "response.output_item.done", "item": '
                b'{"id": "fc_1", "type": "function_call", "status": "completed", '
                b'"arguments": "{\\"a\\":1,\\"b\\":2}", "call_id": "call_abc", '
                b'"name": "add_numbers"}, "output_index": 0}\n\n'
            )
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = chunks
        mock_client.post.return_value = mock_response

        provider = OpenAIResponsesProvider(api_key="test_key")
        response = await provider.acomplete(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "add 1 and 2"}],
            stream=True,
        )

        collected = [chunk async for chunk in response.stream_content()]
        assert collected == []

        assert response.main_response is not None
        assert response.main_response.tool_calls is not None
        tool_call = response.main_response.tool_calls[0]
        assert tool_call.name == "add_numbers"
        assert tool_call.function["arguments"] == '{"a":1,"b":2}'
        assert tool_call.input["call_id"] == "call_abc"


@pytest.mark.asyncio
async def test_stream_text_populates_main_response_and_keeps_deltas() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def chunks():
            yield b'data: {"type": "response.output_text.delta", "delta": "Hello"}\n\n'
            yield b'data: {"type": "response.output_text.delta", "delta": " world"}\n\n'
            yield (
                b'data: {"type": "response.output_item.done", "item": '
                b'{"id": "msg_1", "type": "message", "status": "completed", '
                b'"content": [{"type": "output_text", "text": "Hello world"}], '
                b'"role": "assistant"}, "output_index": 0}\n\n'
            )
            yield b"data: [DONE]\n\n"

        mock_response.read_chunks = chunks
        mock_client.post.return_value = mock_response

        provider = OpenAIResponsesProvider(api_key="test_key")
        response = await provider.acomplete(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "say hello"}],
            stream=True,
        )

        collected = [chunk async for chunk in response.stream_content()]
        assert collected == ["Hello", " world"]

        assert response.main_response is not None
        assert response.main_response.tool_calls is None
        assert response.main_response.content == "Hello world"


def test_parse_response_keeps_call_id_for_continuation() -> None:
    provider = OpenAIResponsesProvider(api_key="test_key")

    original = {
        "output": [
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_abc",
                "name": "add_numbers",
                "arguments": '{"a":1,"b":2}',
            }
        ]
    }

    message = provider._parse_response(original)
    stored_tool_call = message.original["tool_calls"][0]
    assert stored_tool_call["input"]["call_id"] == "call_abc"

    follow_up = provider._messages_to_input(
        [
            {"role": "user", "content": "add 1 and 2"},
            {"role": "assistant", "tool_calls": message.original["tool_calls"]},
        ]
    )
    echoed_function_call = next(item for item in follow_up if item.get("type") == "function_call")
    assert echoed_function_call["call_id"] == "call_abc"
