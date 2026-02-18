from typing import Any

from unittest.mock import patch

import pytest

from llm_async.models import Tool
from llm_async.providers.base import BaseProvider


def test_base_init() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as mock_http_client:
        provider = BaseProvider(api_key="test_key", base_url="https://test.com")
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://test.com"
        mock_http_client.assert_called_once_with(http2=False)


def test_base_init_http2_enabled() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as mock_http_client:
        BaseProvider(api_key="test_key", base_url="https://test.com", http2=True)
        mock_http_client.assert_called_once_with(http2=True)


def test_base_init_http2_with_client_kwargs() -> None:
    with patch("llm_async.providers.base.aiosonic.HTTPClient") as mock_http_client:
        BaseProvider(
            api_key="test_key",
            base_url="https://test.com",
            client_kwargs={"verify": False, "timeout": 10},
            http2=True,
        )
        mock_http_client.assert_called_once_with(verify=False, timeout=10, http2=True)


@pytest.mark.asyncio
async def test_base_acomplete_abstract() -> None:
    class MockProvider(BaseProvider):
        pass

    provider = MockProvider(api_key="test")
    with pytest.raises(NotImplementedError):
        await provider.acomplete(model="test", messages=[])


def test_tool_dataclass_creation() -> None:
    tool = Tool(
        name="get_weather",
        description="Get current weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
    )
    assert tool.name == "get_weather"
    assert tool.description == "Get current weather"
    assert tool.parameters == {"type": "object", "properties": {"location": {"type": "string"}}}
    assert tool.input_schema == {"type": "object", "properties": {"location": {"type": "string"}}}

    @pytest.mark.asyncio
    async def test_base_acomplete_with_tools() -> None:
        class MockProvider(BaseProvider):
            async def acomplete(  # type: ignore[no-untyped-def]
                self,
                model,
                messages,
                stream=False,
                tools=None,
                tool_choice=None,
                tool_executor=None,
                **kwargs,
            ) -> Any:
                return {"tools": tools, "tool_choice": tool_choice}

        provider = MockProvider(api_key="test")
        tool = Tool(name="test_tool", description="test")
        result = await provider.acomplete(
            model="test", messages=[], tools=[tool], tool_choice="auto"
        )
        assert result["tools"] == [tool]
        assert result["tool_choice"] == "auto"
