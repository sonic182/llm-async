from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_async.models.response_schema import ResponseSchema


def test_response_schema_for_openai_format() -> None:
    schema = ResponseSchema(schema={"type": "object"})
    payload = schema.for_openai()
    assert payload["json_schema"]["name"] == "response"
    assert payload["json_schema"]["strict"] is True


def test_response_schema_coerce_accepts_mapping() -> None:
    raw_schema = {"type": "object", "properties": {"field": {"type": "string"}}}
    coerced = ResponseSchema.coerce(raw_schema)
    assert isinstance(coerced, ResponseSchema)
    assert coerced.schema["properties"]["field"] == {"type": "string"}


def test_response_schema_for_google_strips_additional_properties() -> None:
    schema = ResponseSchema(
        schema={
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "x": {"type": "string", "additionalProperties": False},
                "y": [{"type": "object", "additional_properties": True}],
            },
        }
    )
    payload = schema.for_google()
    cleaned = payload["responseSchema"]
    assert "additionalProperties" not in cleaned
    assert "additionalProperties" not in cleaned["properties"]["x"]
    inner_list = cleaned["properties"]["y"]
    assert isinstance(inner_list, list)
    assert "additional_properties" not in inner_list[0]


def test_response_schema_for_openai_responses() -> None:
    schema = ResponseSchema(schema={"type": "object"}, name="story", strict=False)
    payload = schema.for_openai_responses()
    assert payload["name"] == "story"
    assert payload["strict"] is False


def test_response_schema_dataclass_conversion() -> None:
    @dataclass
    class Coordinates:
        lat: float
        lon: float

    @dataclass
    class Forecast:
        location: str
        summary: str
        rain_probability: float
        coords: Coordinates
        tags: list[str] | None = None

    schema = ResponseSchema.from_dataclass(Forecast)
    assert schema.schema["type"] == "object"
    assert set(schema.schema["required"]) == {
        "location",
        "summary",
        "rain_probability",
        "coords",
    }
    coords_schema = schema.schema["properties"]["coords"]
    assert coords_schema["properties"]["lat"] == {"type": "number"}
    coerced = ResponseSchema.coerce(Forecast)
    assert isinstance(coerced, ResponseSchema)


def test_response_schema_coerce_invalid_type() -> None:
    with pytest.raises(TypeError):
        ResponseSchema.coerce("not a schema")  # type: ignore[arg-type]
