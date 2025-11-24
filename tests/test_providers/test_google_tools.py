import pytest

from llm_async.models import Tool
from llm_async.models.response_schema import ResponseSchema
from llm_async.providers.google import GoogleProvider


def test_format_messages_with_tool_calls() -> None:
    provider = GoogleProvider(api_key="test")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "add", "arguments": {"a": 1, "b": 2}},
                }
            ],
        }
    ]

    formatted = provider._format_messages(messages)
    assert isinstance(formatted, list)
    assert formatted and "parts" in formatted[0]
    part = formatted[0]["parts"][0]
    assert "functionCall" in part
    assert part["functionCall"]["name"] == "add"
    assert part["functionCall"]["args"] == {"a": 1, "b": 2}


def test_format_messages_with_tool_result_content() -> None:
    provider = GoogleProvider(api_key="test")
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ignored"},
                {"type": "tool_result", "name": "mytool", "content": "OK"},
            ],
        }
    ]

    formatted = provider._format_messages(messages)
    assert formatted and "parts" in formatted[0]
    # find functionResponse part
    func_parts = [p for p in formatted[0]["parts"] if "functionResponse" in p]
    assert func_parts
    fr = func_parts[0]["functionResponse"]
    assert fr["name"] == "mytool"
    assert fr["response"]["result"] == "OK"


def test_format_tools_and_payload_includes_tools() -> None:
    provider = GoogleProvider(api_key="test")
    tool = Tool(name="sum", description="adds numbers", parameters={"type": "object"})

    formatted = provider._format_tools([tool])
    assert isinstance(formatted, list)
    assert formatted[0]["functionDeclarations"][0]["name"] == "sum"
    assert formatted[0]["functionDeclarations"][0]["description"] == "adds numbers"
    assert "parameters" in formatted[0]["functionDeclarations"][0]

    # Ensure _build_request_payload includes the tools entry
    payload = provider._build_request_payload(
        messages=[{"role": "user", "content": "hi"}], tools=[tool]
    )
    assert "tools" in payload
    assert payload["tools"] == formatted


def test_build_headers_vertex_and_api_key() -> None:
    # API key mode
    provider = GoogleProvider(api_key="apikey")
    headers = provider._build_headers()
    assert headers["X-GOOG-API-KEY"] == "apikey"

    # Vertex mode requires token
    with pytest.raises(ValueError):
        GoogleProvider(vertex_config={"project_id": "p"})._build_headers()

    # Vertex with token
    prov = GoogleProvider(vertex_config={"project_id": "p", "goth_token": "tok"})
    headers2 = prov._build_headers()
    assert headers2["Authorization"] == "Bearer tok"


def test_build_vertex_url_variants() -> None:
    prov = GoogleProvider(vertex_config={"project_id": "proj", "location_id": "global"})
    url = prov._build_vertex_url({"project_id": "proj", "location_id": "global"})
    assert "aiplatform.googleapis.com" in url

    url2 = prov._build_vertex_url({"project_id": "proj", "location_id": "europe-west1"})
    assert "europe-west1-aiplatform.googleapis.com" in url2

    # custom endpoint
    custom = {"project_id": "proj", "api_endpoint": "https://example.com/custom"}
    assert prov._build_vertex_url(custom) == "https://example.com/custom"

    with pytest.raises(ValueError):
        prov._build_vertex_url({})


def test_response_schema_google_helper() -> None:
    schema = ResponseSchema(
        schema={
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "x": {"type": "string", "additionalProperties": False},
                "y": [{"additionalProperties": True}],
            },
        }
    )

    config = schema.for_google()
    cleaned = config["responseSchema"]
    assert "additionalProperties" not in cleaned
    assert "additionalProperties" not in cleaned["properties"]["x"]
