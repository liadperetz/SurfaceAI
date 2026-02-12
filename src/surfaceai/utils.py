"""Shared utilities for surfaceai."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse JSON from an LLM response.

    Handles markdown code blocks (```json ... ```) and bare JSON objects.
    Falls back to extracting the first top-level ``{...}`` object on parse error.

    Args:
        text: Raw LLM output that should contain JSON.
        fallback: Default dict to return when parsing fails.
                  If None, returns ``{"error": "Failed to parse JSON"}``.

    Returns:
        Parsed dictionary.
    """
    if fallback is None:
        fallback = {"error": "Failed to parse JSON"}

    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return fallback
