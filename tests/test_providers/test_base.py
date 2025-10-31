from typing import Any

import pytest

from llmpy.models import Tool
from llmpy.providers.base import BaseProvider


def test_base_init() -> None:
    provider = BaseProvider(api_key="test_key", base_url="https://test.com")
    assert provider.api_key == "test_key"
    assert provider.base_url == "https://test.com"


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
                auto_execute_tools=False,
                tool_executor=None,
                max_tool_iterations=10,
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
