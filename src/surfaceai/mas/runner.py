"""MAS Runner — entry point dispatched from the main runner."""

from __future__ import annotations

import logging
from typing import Any, Optional

from surfaceai.config.schemas import MASExperiment, OpenHandsSettings, Provider
from surfaceai.mas.agents.base import AgentRole
from surfaceai.mas.agents.llm_agent import LLMAgent
from surfaceai.mas.agents.openhands_agent import OpenHandsAgent
from surfaceai.mas.orchestrators.base import OrchestrationResult
from surfaceai.mas.orchestrators.planner_executor import PlannerExecutorOrchestrator
from surfaceai.mas.prompts import (
    E1_EXECUTOR_SYSTEM,
    E1_PLANNER_SYSTEM,
    E2_EXECUTOR_SYSTEM,
    E2_PLANNER_SYSTEM,
)

logger = logging.getLogger(__name__)


class MASRunner:
    """Runner for multi-agent system experiments.

    Factory method builds the right orchestrator based on the experiment type.
    """

    def __init__(
        self,
        experiment: MASExperiment,
        provider: str,
        model: Optional[str] = None,
        openhands_settings: Optional[OpenHandsSettings] = None,
        max_steps: int = 10,
    ):
        self.experiment = experiment
        self.provider = provider
        self.model = model
        self.openhands_settings = openhands_settings or OpenHandsSettings()
        self.max_steps = max_steps
        self._orchestrator = self._build_orchestrator()

    def _build_orchestrator(self) -> PlannerExecutorOrchestrator:
        """Build the orchestrator for the configured experiment."""
        if self.experiment in (MASExperiment.e1, MASExperiment.e2):
            return self._build_planner_executor()
        raise ValueError(f"Experiment {self.experiment.value} is not yet implemented")

    def _build_planner_executor(self) -> PlannerExecutorOrchestrator:
        """Build a PlannerExecutorOrchestrator for E1 or E2."""
        if self.experiment == MASExperiment.e2:
            planner_prompt = E2_PLANNER_SYSTEM
            executor_prompt = E2_EXECUTOR_SYSTEM
            safety_filtering = True
        else:
            planner_prompt = E1_PLANNER_SYSTEM
            executor_prompt = E1_EXECUTOR_SYSTEM
            safety_filtering = False

        planner = LLMAgent(
            agent_id="planner",
            role=AgentRole.planner,
            system_prompt=planner_prompt,
            provider=self.provider,
            model=self.model,
            safety_filtering=safety_filtering,
        )

        # Convert string provider to Provider enum for OpenHandsAgent
        provider_enum = Provider(self.provider)
        executor = OpenHandsAgent(
            agent_id="executor",
            role=AgentRole.executor,
            system_prompt=executor_prompt,
            openhands_settings=self.openhands_settings,
            provider=provider_enum,
            model=self.model,
        )

        return PlannerExecutorOrchestrator(
            planner=planner,
            executor=executor,
            max_steps=self.max_steps,
        )

    def run(
        self,
        prompt: str,
        website_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run the MAS experiment on a single prompt.

        Returns a dict compatible with the existing runner interface:
        response, steps, agent_traces, metadata.
        """
        result: OrchestrationResult = self._orchestrator.run(
            task=prompt,
            website_url=website_url,
        )

        agent_traces_dicts = [t.to_dict() for t in result.agent_traces]

        return {
            "response": result.final_response,
            "steps": result.steps,
            "agent_traces": agent_traces_dicts,
            "refused": result.refused,
            "metadata": result.metadata,
        }

    def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        self._orchestrator.cleanup()
