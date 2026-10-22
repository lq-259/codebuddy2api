"""Normalize tool schemas and recognize upstream channel-check responses."""
from __future__ import annotations

from copy import deepcopy
import re
from urllib.parse import urlsplit, urlunsplit

_SCHEMA_MAPS = frozenset(("properties", "patternProperties", "$defs", "definitions",
                          "dependentSchemas", "dependencies"))
_SCHEMA_ARRAYS = frozenset(("allOf", "anyOf", "oneOf", "prefixItems", "items"))
_SCHEMA_CHILDREN = frozenset(("items", "additionalItems", "contains", "unevaluatedItems",
                              "additionalProperties", "unevaluatedProperties", "propertyNames",
                              "not", "if", "then", "else", "contentSchema"))
_CHANNEL_CODE = re.compile(r"""(['"])code\1\s*:\s*(['"]?)(?:11128|11-128)\2""", re.IGNORECASE)
_CHANNEL_TEXT = ("channel verification", "official client", "illegal api invocation from an unapproved channel")


def _normalize_schema_patterns(schema) -> None:
    """Normalize RE2-incompatible underscore escapes only in schema pattern positions."""
    if not isinstance(schema, dict):
        return
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and "\\_" in pattern:
        schema["pattern"] = pattern.replace("\\_", "_")
    properties = schema.get("patternProperties")
    if isinstance(properties, dict):
        schema["patternProperties"] = {
            key.replace("\\_", "_") if isinstance(key, str) else key: value
            for key, value in properties.items()
        }
    for key in _SCHEMA_MAPS:
        value = schema.get(key)
        if isinstance(value, dict):
            for child in value.values():
                _normalize_schema_patterns(child)
    for key in _SCHEMA_ARRAYS:
        value = schema.get(key)
        if isinstance(value, list):
            for child in value:
                _normalize_schema_patterns(child)
    for key in _SCHEMA_CHILDREN:
        _normalize_schema_patterns(schema.get(key))


def normalize_tool_patterns(body: dict) -> dict:
    """Copy and normalize Chat and legacy function tool declarations."""
    result = dict(body)
    for key in ("tools", "functions"):
        tools = result.get(key)
        if not isinstance(tools, list):
            continue
        result[key] = deepcopy(tools)
        for tool in result[key]:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                _normalize_schema_patterns(function.get("parameters"))
            _normalize_schema_patterns(tool.get("parameters"))
    return result


def is_channel_check_error(raw: bytes | str) -> bool:
    """Require both a channel-check business code and matching explanatory text."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    if not isinstance(text, str) or not _CHANNEL_CODE.search(text):
        return False
    lower = text.lower()
    return "渠道校验" in text or any(marker in lower for marker in _CHANNEL_TEXT)


def channel_fallback_url(url: str) -> str | None:
    """Return the console chat endpoint only for the primary v2 chat endpoint."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.path != "/v2/chat/completions":
        return None
    return urlunsplit(parts._replace(path="/console/chat/completions"))
