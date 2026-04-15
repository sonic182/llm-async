import functools
import inspect
import types
import typing
from collections.abc import Callable
from typing import Any

from llm_async.models import Tool


def _type_to_schema(typ: Any) -> dict[str, Any] | None:
    if typ is inspect.Parameter.empty:
        return None

    origin = typing.get_origin(typ)

    if isinstance(typ, types.UnionType) or origin is typing.Union:
        args = typing.get_args(typ)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        return None

    if origin is list:
        args = typing.get_args(typ)
        if args:
            items = _type_to_schema(args[0])
            if items:
                return {"type": "array", "items": items}
        return {"type": "array"}

    if typ is list:
        return {"type": "array"}

    if typ is dict or origin is dict:
        return {"type": "object"}

    _MAP: dict[Any, dict[str, Any]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    return _MAP.get(typ)


def _build_schema(func: Callable[..., Any]) -> dict[str, Any]:
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = hints.get(name, inspect.Parameter.empty)

        is_optional = False
        origin = typing.get_origin(annotation)
        if isinstance(annotation, types.UnionType) or origin is typing.Union:
            args = typing.get_args(annotation)
            if type(None) in args:
                is_optional = True

        schema = _type_to_schema(annotation)
        if schema is not None:
            properties[name] = schema

        has_default = param.default is not inspect.Parameter.empty
        if not has_default and not is_optional and schema is not None:
            required.append(name)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    schema = _build_schema(func)
    description = inspect.getdoc(func) or ""
    wrapper.tool = Tool(  # type: ignore[attr-defined]
        name=func.__name__,
        description=description,
        parameters=schema,
        input_schema=schema,
    )
    return wrapper
