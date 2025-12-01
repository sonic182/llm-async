from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, dataclass, fields, is_dataclass
import inspect
import sys
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints


@dataclass
class ResponseSchema:
    schema: Mapping[str, Any]
    name: str = "response"
    strict: bool = True
    mime_type: str = "application/json"

    def __post_init__(self) -> None:
        self.schema = deepcopy(dict(self.schema))

    @classmethod
    def coerce(cls, value: Any | None) -> ResponseSchema | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        dataclass_def = _resolve_dataclass_definition(value)
        if dataclass_def is not None:
            return cls.from_dataclass(dataclass_def)
        if isinstance(value, Mapping):
            return cls(schema=value)
        msg = "response_schema must be ResponseSchema, mapping, or dataclass"
        raise TypeError(msg)

    @classmethod
    def from_dataclass(
        cls,
        dataclass_definition: Any,
        *,
        name: str = "response",
        strict: bool = True,
        mime_type: str = "application/json",
    ) -> ResponseSchema:
        dataclass_type = _resolve_dataclass_definition(dataclass_definition)
        if dataclass_type is None:
            msg = "value is not a dataclass type or instance"
            raise TypeError(msg)
        frame = inspect.currentframe()
        caller_locals: Mapping[str, Any] | None = None
        if frame is not None and frame.f_back is not None:
            caller_locals = dict(frame.f_back.f_locals)
        try:
            schema = _schema_from_dataclass_type(dataclass_type, caller_locals)
        finally:  # pragma: no cover - avoid reference cycles if frames stick around
            del frame
        return cls(schema=schema, name=name, strict=strict, mime_type=mime_type)

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
            "responseSchema": self._remove_additional_properties(deepcopy(self.schema)),
        }

    def _remove_additional_properties(self, schema: Any) -> Any:
        if isinstance(schema, dict):
            cleaned: dict[str, Any] = {}
            for key, value in schema.items():
                if key in {"additionalProperties", "additional_properties"}:
                    continue
                cleaned[key] = self._remove_additional_properties(value)
            return cleaned
        if isinstance(schema, list):
            return [self._remove_additional_properties(item) for item in schema]
        return schema


def _resolve_dataclass_definition(value: Any) -> type[Any] | None:
    if inspect.isclass(value) and is_dataclass(value):
        return value
    if is_dataclass(value):
        return type(value)
    return None


def _schema_from_dataclass_type(
    dataclass_type: type[Any],
    caller_locals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    module = sys.modules.get(dataclass_type.__module__)
    globalns = dict(vars(module)) if module is not None else {}
    localns: dict[str, Any] = {dataclass_type.__name__: dataclass_type}
    if caller_locals:
        localns.update(caller_locals)
    try:
        type_hints = get_type_hints(
            dataclass_type,
            globalns=globalns or None,
            localns=localns,
            include_extras=True,
        )
    except Exception:  # pragma: no cover - fallback when resolution fails
        type_hints = {}

    props: dict[str, Any] = {}
    required: list[str] = []
    for field in fields(dataclass_type):
        annotation = type_hints.get(field.name, field.type)
        props[field.name] = _schema_from_annotation(annotation)
        if field.default is MISSING and field.default_factory is MISSING:  # type: ignore[attr-defined]
            required.append(field.name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _schema_from_annotation(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin is None:
        return _schema_from_concrete(annotation)
    if origin in {list, Sequence, tuple, set, frozenset}:  # type: ignore[arg-type]
        args = get_args(annotation)
        if len(args) > 1 and origin is tuple:
            item_schemas = [_schema_from_annotation(arg) for arg in args]
            return {
                "type": "array",
                "prefixItems": item_schemas,
                "minItems": len(args),
                "maxItems": len(args),
            }
        items_type = args[0] if args else Any
        return {"type": "array", "items": _schema_from_annotation(items_type)}
    if origin in {dict, Mapping}:
        args = get_args(annotation)
        value_type = args[1] if len(args) == 2 else Any
        return {
            "type": "object",
            "additionalProperties": _schema_from_annotation(value_type),
        }
    if origin in {Union, UnionType}:
        schemas = []
        for arg in get_args(annotation):
            if arg is type(None):
                schemas.append({"type": "null"})
            else:
                schemas.append(_schema_from_annotation(arg))
        if len(schemas) == 1:
            return schemas[0]
        return {"anyOf": schemas}
    if origin is Literal:
        return {"enum": list(get_args(annotation))}
    if origin is Annotated:
        annotated_args = get_args(annotation)
        return _schema_from_annotation(annotated_args[0]) if annotated_args else {}
    return {}


def _schema_from_concrete(annotation: Any) -> dict[str, Any]:
    if annotation in {str}:
        return {"type": "string"}
    if annotation in {int}:
        return {"type": "integer"}
    if annotation in {float}:
        return {"type": "number"}
    if annotation in {bool}:
        return {"type": "boolean"}
    if annotation is Any:
        return {}
    if inspect.isclass(annotation) and is_dataclass(annotation):
        return _schema_from_dataclass_type(annotation)
    try:
        from enum import Enum

        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            return {"enum": [member.value for member in annotation]}
    except ImportError:  # pragma: no cover - stdlib always available
        pass
    return {}
