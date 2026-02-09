"""Base types and protocol for MAS agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class AgentRole(str, Enum):
    planner = "planner"
    executor = "executor"
    safety_critic = "safety_critic"
    coordinator = "coordinator"
    peer = "peer"
    chain_node = "chain_node"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent after processing a message."""
    agent_id: str
    role: AgentRole
    content: str
    refused: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "content": self.content,
            "refused": self.refused,
            "steps": self.steps,
            "metadata": self.metadata,
        }


@runtime_checkable
class Agent(Protocol):
    """Protocol for MAS agents."""

    def act(
        self,
        message: AgentMessage,
        context: Optional[list[AgentMessage]] = None,
    ) -> AgentResponse: ...

    def reset(self) -> None: ...
