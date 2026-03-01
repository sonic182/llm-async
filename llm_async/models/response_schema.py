from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass
class ResponseSchema:
    schema: Mapping[str, Any]
    name: str = "response"
    strict: bool = True
    mime_type: str = "application/json"

    def __post_init__(self) -> None:
        self.schema = deepcopy(dict(self.schema))

    @classmethod
    def coerce(cls, value: ResponseSchema | Mapping[str, Any] | None) -> ResponseSchema | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(schema=value)
        msg = "response_schema must be ResponseSchema or mapping"
        raise TypeError(msg)

    def for_openai(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": deepcopy(self.schema),
                "strict": self.strict,
            },
        }

    def for_openrouter(self) -> dict[str, Any]:
        return self.for_openai()

    def for_openai_responses(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "name": self.name,
            "schema": deepcopy(self.schema),
            "strict": self.strict,
        }

    def for_google(self) -> dict[str, Any]:
        return {
            "responseMimeType": self.mime_type,
            "responseSchema": deepcopy(self.schema),
        }
