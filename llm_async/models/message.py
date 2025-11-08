from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from llm_async.models.tool_call import ToolCall

Role = Literal["system", "user", "assistant", "tool"]
Content = Union[str, list[dict[str, Any]]]
_ALLOWED_ROLES: tuple[Role, ...] = ("system", "user", "assistant", "tool")


@dataclass
class Message:
    role: Role
    content: Content
    tool_calls: Optional[list[ToolCall]] = None
    original: Optional[dict[str, Any]] = field(default=None)


def normalize_messages(messages: Sequence[Union["Message", Mapping[str, Any]]]) -> list["Message"]:
    normalized: list[Message] = []
    for message in messages:
        if isinstance(message, Message):
            normalized.append(message)
            continue
        if not isinstance(message, Mapping):
            raise TypeError("Message entries must be Message or mapping")
        role = message.get("role")
        if role not in _ALLOWED_ROLES:
            raise ValueError("Invalid role provided in message")
        content: Any
        if "content" in message:
            content = message.get("content")
        elif "parts" in message:
            content = message.get("parts")
        else:
            content = ""
        if content is None:
            content_value = ""
        elif isinstance(content, list):
            if not all(isinstance(part, Mapping) for part in content):
                raise TypeError("List content must contain mapping entries")
            content_value = [dict(part) for part in content]
        elif isinstance(content, str):
            content_value = content
        else:
            raise TypeError("Content must be str or list of mappings")
        tool_calls = _coerce_tool_calls(message.get("tool_calls"))
        original_payload: Optional[dict[str, Any]] = dict(message)
        inner_original = original_payload.pop("original", None)
        if isinstance(inner_original, Mapping):
            original_payload = dict(inner_original)
        normalized.append(
            Message(
                role=role,
                content=content_value,
                tool_calls=tool_calls,
                original=original_payload,
            )
        )
    return normalized


def validate_messages(messages: Sequence["Message"]) -> None:
    for message in messages:
        if message.role not in _ALLOWED_ROLES:
            raise ValueError("Invalid role on Message instance")
        if not isinstance(message.content, (str, list)):
            raise TypeError("Message content must be str or list")
        if isinstance(message.content, list):
            if not all(isinstance(part, Mapping) for part in message.content):
                raise TypeError("List content entries must be mappings")
        if message.tool_calls is not None and not isinstance(message.tool_calls, list):
            raise TypeError("tool_calls must be a list when provided")
        if message.tool_calls:
            if not all(isinstance(tc, ToolCall) for tc in message.tool_calls):
                raise TypeError("tool_calls entries must be ToolCall instances")


def _coerce_tool_calls(value: Any) -> Optional[list[ToolCall]]:
    if value is None:
        return None
    if isinstance(value, ToolCall):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        calls: list[ToolCall] = []
        for entry in value:
            if isinstance(entry, ToolCall):
                calls.append(entry)
                continue
            if not isinstance(entry, Mapping):
                raise TypeError("tool_calls entries must be mappings or ToolCall")
            calls.append(
                ToolCall(
                    id=str(entry.get("id", "")),
                    type=str(entry.get("type", "")),
                    name=entry.get("name"),
                    input=entry.get("input"),
                    function=entry.get("function"),
                )
            )
        return calls or None
    raise TypeError("tool_calls must be a sequence, ToolCall, or None")


def message_to_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(message.original, Mapping):
        payload = dict(message.original)
    payload["role"] = message.role
    if isinstance(message.content, list):
        payload["content"] = [dict(part) for part in message.content]
    else:
        payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [_tool_call_to_dict(tc) for tc in message.tool_calls]
    else:
        payload.pop("tool_calls", None)
    return payload


def _tool_call_to_dict(tool_call: ToolCall) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": tool_call.id,
        "type": tool_call.type,
    }
    if tool_call.name is not None:
        data["name"] = tool_call.name
    if tool_call.input is not None:
        data["input"] = tool_call.input
    if tool_call.function is not None:
        data["function"] = tool_call.function
    return data
