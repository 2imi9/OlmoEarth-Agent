# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Validate model-emitted tool arguments against the tool's JSON Schema.

The LLM is nondeterministic; the tool layer should be deterministic about
what it accepts (each layer narrows what the next layer can get wrong).
:class:`~olmoearth_agent.llm.types.ToolCall` has always documented that
"callers should validate against the corresponding ToolSpec before dispatch" —
this module is that validation, enforced in
:meth:`olmoearth_agent.tools.registry.ToolRegistry.dispatch`. A malformed call
is rejected *before* the handler runs, with a message that tells the model
exactly which argument is wrong and what was expected, instead of surfacing as
a bare ``KeyError`` from handler internals.

Deliberately a small, permissive subset of JSON Schema:

- checked: ``required``, ``type`` (including type lists), ``enum``, nested
  ``properties``/``required`` on objects, ``items`` on arrays;
- lenient: arguments not declared in ``properties`` pass through (handlers
  ignore extras today), and any scalar satisfies ``type: string`` (handlers
  coerce with ``str()``, so rejecting an int project id would be a
  regression, not a safety win);
- unknown schema keywords are ignored.
"""

from __future__ import annotations

from typing import Any

_SIMPLE_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "boolean": bool,
}


def validate_arguments(
    arguments: dict[str, Any], schema: dict[str, Any]
) -> list[str]:
    """Check ``arguments`` against ``schema``; return problems (empty = valid).

    Each problem is a single human/model-readable sentence naming the argument,
    what was expected, and what arrived.
    """
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        return []
    properties = schema.get("properties") or {}
    problems: list[str] = [
        f"missing required argument {name!r}"
        for name in schema.get("required") or []
        if name not in arguments
    ]
    for name, value in arguments.items():
        subschema = properties.get(name)
        if isinstance(subschema, dict):
            problems.extend(_check(name, value, subschema))
    return problems


def schema_summary(schema: dict[str, Any]) -> dict[str, Any]:
    """A compact restatement of the schema for error envelopes.

    Small enough to feed back to the model verbatim: required names plus a
    one-line ``type``/``enum`` sketch per declared property.
    """
    properties = schema.get("properties") or {}
    sketch: dict[str, Any] = {}
    for name, sub in properties.items():
        if not isinstance(sub, dict):
            continue
        if "enum" in sub:
            sketch[name] = {"enum": sub["enum"]}
        else:
            sketch[name] = sub.get("type", "any")
    return {"required": list(schema.get("required") or []), "properties": sketch}


def _check(path: str, value: Any, schema: dict[str, Any]) -> list[str]:
    """Validate one value against its subschema, recursing into containers."""
    problems: list[str] = []
    declared = schema.get("type")
    if declared:
        types = declared if isinstance(declared, list) else [declared]
        if not any(_is_type(value, t) for t in types):
            problems.append(
                f"argument {path!r} should be {' or '.join(map(str, types))}, "
                f"got {type(value).__name__}"
            )
            return problems  # wrong shape: deeper checks would just cascade
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        problems.append(
            f"argument {path!r} must be one of {enum!r}, got {value!r}"
        )
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            problems.extend(_check(f"{path}[{i}]", item, schema["items"]))
    if isinstance(value, dict):
        problems.extend(
            f"argument {path!r} is missing required key {req!r}"
            for req in schema.get("required") or []
            if req not in value
        )
        subproperties = schema.get("properties") or {}
        for key, sub in subproperties.items():
            if key in value and isinstance(sub, dict):
                problems.extend(_check(f"{path}.{key}", value[key], sub))
    return problems


def _is_type(value: Any, type_name: str) -> bool:
    """Does ``value`` satisfy the JSON-Schema ``type_name``?

    ``bool`` is excluded from ``number``/``integer`` (Python's bool subclasses
    int); ``string`` accepts any scalar because handlers ``str()``-coerce.
    """
    if type_name == "string":
        return isinstance(value, (str, int, float, bool))
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "null":
        return value is None
    expected = _SIMPLE_TYPES.get(type_name)
    if expected is None:
        return True  # unknown type keyword: be permissive
    return isinstance(value, expected)
