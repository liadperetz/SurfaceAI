"""Shared utilities for MAS orchestrators."""

from __future__ import annotations

import re


def parse_steps(text: str) -> list[str]:
    """Parse numbered/bulleted steps from planner output.

    Supports formats:
    - "1. action" / "2. action"
    - "1) action" / "2) action"
    - "- action" / "* action"
    - "Step 1: action" / "Step 2: action"

    Falls back to treating the entire output as a single step.
    """
    lines = text.strip().splitlines()
    steps: list[str] = []

    pattern = re.compile(
        r"^\s*(?:"
        r"(?:\d+)[.)]\s+"        # "1. " or "1) "
        r"|[-*]\s+"              # "- " or "* "
        r"|Step\s+\d+[.:]\s*"   # "Step 1: " or "Step 1."
        r")(.*)",
        re.IGNORECASE,
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            action = match.group(1).strip()
            if action:
                steps.append(action)

    if not steps:
        text = text.strip()
        if text:
            steps.append(text)

    return steps
