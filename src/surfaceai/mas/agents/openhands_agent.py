"""OpenHands-based executor agent wrapping the existing OpenHandsRunner."""

from __future__ import annotations

import logging
from typing import Optional

from surfaceai.config.schemas import OpenHandsSettings, Provider
from surfaceai.mas.agents.base import AgentMessage, AgentResponse, AgentRole
from surfaceai.openhands.runner import OpenHandsRunner

logger = logging.getLogger(__name__)


class OpenHandsAgent:
    """Web agent that wraps OpenHandsRunner.

    Used as the Executor agent in E1/E2 experiments.
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        system_prompt: str,
        openhands_settings: OpenHandsSettings,
        provider: Provider,
        model: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self._runner = OpenHandsRunner(
            settings=openhands_settings,
            provider=provider,
            model=model,
        )

    def act(
        self,
        message: AgentMessage,
        context: Optional[list[AgentMessage]] = None,
    ) -> AgentResponse:
        """Execute a web action via OpenHands.

        Prepends the system prompt to the message content, extracts
        website_url from message metadata, and delegates to the runner.
        """
        full_prompt = f"{self.system_prompt}\n\n{message.content}"
        website_url = message.metadata.get("website_url")

        result = self._runner.run(
            prompt=full_prompt,
            website_url=website_url,
        )

        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            content=result.get("response", ""),
            steps=result.get("steps", []),
            metadata={
                "conversation_id": result.get("conversation_id", ""),
                "success": result.get("success", False),
            },
        )

    def reset(self) -> None:
        """No persistent state to reset."""
        pass

    def cleanup(self) -> None:
        """Delegate cleanup to the underlying runner."""
        self._runner.cleanup()
