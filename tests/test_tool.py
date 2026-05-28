from llm_async.models import Tool
from llm_async.tool import tool


def test_basic_types() -> None:
    @tool
    def func(a: str, b: int, c: float, d: bool) -> None:
        pass

    schema = func.tool.parameters
    assert schema["properties"]["a"] == {"type": "string"}
    assert schema["properties"]["b"] == {"type": "integer"}
    assert schema["properties"]["c"] == {"type": "number"}
    assert schema["properties"]["d"] == {"type": "boolean"}
    assert set(schema["required"]) == {"a", "b", "c", "d"}


def test_list_generic() -> None:
    @tool
    def func(items: list[str]) -> None:
        pass

    schema = func.tool.parameters
    assert schema["properties"]["items"] == {"type": "array", "items": {"type": "string"}}
    assert "items" in schema["required"]


def test_plain_list() -> None:
    @tool
    def func(items: list) -> None:
        pass

    assert func.tool.parameters["properties"]["items"] == {"type": "array"}


def test_dict_type() -> None:
    @tool
    def func(data: dict) -> None:
        pass

    assert func.tool.parameters["properties"]["data"] == {"type": "object"}


def test_optional_not_required() -> None:
    @tool
    def func(x: str | None) -> None:
        pass

    schema = func.tool.parameters
    assert schema["properties"]["x"] == {"type": "string"}
    assert "required" not in schema or "x" not in schema.get("required", [])


def test_no_annotation_omitted() -> None:
    @tool
    def func(x, y: str) -> None:  # type: ignore[no-untyped-def]
        pass

    schema = func.tool.parameters
    assert "x" not in schema["properties"]
    assert "y" in schema["properties"]


def test_default_value_not_required() -> None:
    @tool
    def func(x: str, y: int = 0) -> None:
        pass

    schema = func.tool.parameters
    assert "x" in schema["required"]
    assert "y" not in schema["required"]


def test_docstring_extracted() -> None:
    @tool
    def func(text: str) -> int:
        """Count words in text.

        This docstring is used by the LLM to understand the tool's purpose.
        """
        return len(text.split())

    assert func.tool.description.startswith("Count words in text.")
    assert "LLM" in func.tool.description


def test_no_docstring() -> None:
    @tool
    def func(x: str) -> None:
        pass

    assert func.tool.description == ""


def test_function_still_callable() -> None:
    @tool
    def word_count(text: str) -> int:
        """Count words."""
        return len(text.split())

    assert word_count("a b c") == 3


def test_tool_instance() -> None:
    @tool
    def my_func(x: str) -> None:
        """A description."""
        pass

    assert isinstance(my_func.tool, Tool)
    assert my_func.tool.name == "my_func"
    assert my_func.tool.description == "A description."
    assert my_func.tool.parameters is my_func.tool.input_schema


def test_tool_name_and_schema_match() -> None:
    @tool
    def compute(value: float, label: str) -> str:
        """Compute something."""
        return label

    t = compute.tool
    assert t.name == "compute"
    assert t.parameters == t.input_schema
    assert "value" in t.parameters["properties"]
    assert "label" in t.parameters["properties"]
