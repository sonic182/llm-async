from typing import Any

from llm_async.models.response import Response


def test_response_unsupported_provider() -> None:
    raw: dict[str, Any] = {}
    response = Response(raw, "unknown")
    assert response.original == raw
    assert response.provider_name == "unknown"
    assert response.main_response is None
