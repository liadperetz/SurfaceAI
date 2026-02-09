"""Base types and protocol for MAS orchestrators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from surfaceai.mas.agents.base import AgentResponse


@dataclass
class OrchestrationResult:
    """Result of a multi-agent orchestration run."""
    final_response: str
    agent_traces: list[AgentResponse] = field(default_factory=list)
    refused: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Orchestrator(Protocol):
    """Protocol for MAS orchestrators."""

    def run(
        self,
        task: str,
        website_url: Optional[str] = None,
    ) -> OrchestrationResult: ...

    def cleanup(self) -> None: ...
