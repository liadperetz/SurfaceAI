"""MAS Runner — entry point dispatched from the main runner."""

from __future__ import annotations

import logging
from typing import Any, Optional

from surfaceai.config.schemas import MASExperiment, OpenHandsSettings, Provider
from surfaceai.mas.agents.base import AgentRole
from surfaceai.mas.agents.llm_agent import LLMAgent
from surfaceai.mas.agents.openhands_agent import OpenHandsAgent
from surfaceai.mas.orchestrators.base import OrchestrationResult
from surfaceai.mas.orchestrators.centralized import CentralizedOrchestrator
from surfaceai.mas.orchestrators.fully_connected import FullyConnectedOrchestrator
from surfaceai.mas.orchestrators.planner_executor import PlannerExecutorOrchestrator
from surfaceai.mas.orchestrators.sequential_chain import SequentialChainOrchestrator
from surfaceai.mas.prompts import (
    E1_EXECUTOR_SYSTEM,
    E1_PLANNER_SYSTEM,
    E2_EXECUTOR_SYSTEM,
    E2_PLANNER_SYSTEM,
    E3_COORDINATOR_SYSTEM,
    E3_EXECUTOR_SYSTEM,
    E3_PLANNER_SYSTEM,
    E4_EXECUTOR_SYSTEM,
    E4_PEER_SYSTEM,
    E5_CHAIN_NODE_SYSTEM,
    E5_EXECUTOR_SYSTEM,
    E5_NODE_ROLES,
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

    def _build_orchestrator(self):
        """Build the orchestrator for the configured experiment."""
        if self.experiment in (MASExperiment.e1, MASExperiment.e2):
            return self._build_planner_executor()
        if self.experiment == MASExperiment.e3:
            return self._build_centralized()
        if self.experiment == MASExperiment.e4:
            return self._build_fully_connected()
        if self.experiment == MASExperiment.e5:
            return self._build_sequential_chain()
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

    def _build_centralized(self) -> CentralizedOrchestrator:
        """Build a CentralizedOrchestrator for E3."""
        coordinator = LLMAgent(
            agent_id="coordinator",
            role=AgentRole.coordinator,
            system_prompt=E3_COORDINATOR_SYSTEM,
            provider=self.provider,
            model=self.model,
        )

        planner = LLMAgent(
            agent_id="planner",
            role=AgentRole.planner,
            system_prompt=E3_PLANNER_SYSTEM,
            provider=self.provider,
            model=self.model,
        )

        provider_enum = Provider(self.provider)
        executor = OpenHandsAgent(
            agent_id="executor",
            role=AgentRole.executor,
            system_prompt=E3_EXECUTOR_SYSTEM,
            openhands_settings=self.openhands_settings,
            provider=provider_enum,
            model=self.model,
        )

        return CentralizedOrchestrator(
            coordinator=coordinator,
            planner=planner,
            executor=executor,
            max_steps=self.max_steps,
        )

    def _build_fully_connected(self, num_peers: int = 3) -> FullyConnectedOrchestrator:
        """Build a FullyConnectedOrchestrator for E4."""
        peers = []
        for i in range(1, num_peers + 1):
            peer = LLMAgent(
                agent_id=f"peer_{i}",
                role=AgentRole.peer,
                system_prompt=E4_PEER_SYSTEM.format(
                    agent_id=i, num_peers=num_peers,
                ),
                provider=self.provider,
                model=self.model,
            )
            peers.append(peer)

        provider_enum = Provider(self.provider)
        executor = OpenHandsAgent(
            agent_id="executor",
            role=AgentRole.executor,
            system_prompt=E4_EXECUTOR_SYSTEM,
            openhands_settings=self.openhands_settings,
            provider=provider_enum,
            model=self.model,
        )

        return FullyConnectedOrchestrator(
            peers=peers,
            executor=executor,
            max_steps=self.max_steps,
        )

    def _build_sequential_chain(self, num_nodes: int = 3) -> SequentialChainOrchestrator:
        """Build a SequentialChainOrchestrator for E5."""
        chain_nodes = []
        for i in range(1, num_nodes + 1):
            node_role = E5_NODE_ROLES.get(i, E5_NODE_ROLES[num_nodes])
            node = LLMAgent(
                agent_id=f"node_{i}",
                role=AgentRole.chain_node,
                system_prompt=E5_CHAIN_NODE_SYSTEM.format(
                    node_id=i, num_nodes=num_nodes, node_role=node_role,
                ),
                provider=self.provider,
                model=self.model,
            )
            chain_nodes.append(node)

        provider_enum = Provider(self.provider)
        executor = OpenHandsAgent(
            agent_id="executor",
            role=AgentRole.executor,
            system_prompt=E5_EXECUTOR_SYSTEM,
            openhands_settings=self.openhands_settings,
            provider=provider_enum,
            model=self.model,
        )

        return SequentialChainOrchestrator(
            chain_nodes=chain_nodes,
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
