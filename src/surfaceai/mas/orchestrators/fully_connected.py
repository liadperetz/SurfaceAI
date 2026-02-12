"""Fully connected orchestrator for E4: peer-to-peer communication.

All peer agents communicate directly with one another (mesh topology).
Each peer proposes a plan, critiques others' plans, then they reach
consensus. The consensus plan is executed by the Executor.

Flow:
1. PROPOSE: Each peer independently proposes a plan for the task
2. CRITIQUE: Each peer sees all other proposals and provides feedback
3. CONSENSUS: Each peer produces a final plan incorporating feedback
4. MERGE: Deterministic logic picks the consensus plan (first peer's final)
5. EXECUTE: Steps from the consensus plan are sent to the Executor
"""

from __future__ import annotations

import logging
from typing import Optional

from surfaceai.mas.agents.base import AgentMessage, AgentResponse
from surfaceai.mas.agents.llm_agent import LLMAgent
from surfaceai.mas.agents.openhands_agent import OpenHandsAgent
from surfaceai.mas.orchestrators.base import OrchestrationResult
from surfaceai.mas.orchestrators.utils import parse_steps as _parse_steps

logger = logging.getLogger(__name__)


class FullyConnectedOrchestrator:
    """Orchestrator for E4: Fully Connected Communication (mesh topology).

    Multiple LLM peer agents communicate directly with each other through
    three phases (propose, critique, consensus), then the consensus plan
    is executed step-by-step by the Executor.
    """

    def __init__(
        self,
        peers: list[LLMAgent],
        executor: OpenHandsAgent,
        max_steps: int = 10,
    ):
        self.peers = peers
        self.executor = executor
        self.max_steps = max_steps

    def run(
        self,
        task: str,
        website_url: Optional[str] = None,
    ) -> OrchestrationResult:
        """Execute the fully connected peer workflow."""
        agent_traces: list[AgentResponse] = []
        all_steps: list[dict] = []
        metadata: dict = {"website_url": website_url} if website_url else {}

        # --- Phase 1: PROPOSE — each peer proposes a plan ---
        proposals: list[AgentResponse] = []
        for peer in self.peers:
            msg = AgentMessage(
                sender="system",
                content=(
                    f"PROPOSE PHASE: Here is the task to accomplish on a website:\n\n"
                    f"{task}\n\n"
                    f"Write your numbered plan of concrete web actions."
                ),
                metadata=metadata,
            )
            response = peer.act(msg)
            proposals.append(response)
            agent_traces.append(response)
            logger.info("Peer %s proposed: %s", peer.agent_id, response.content[:80])

        # --- Phase 2: CRITIQUE — each peer sees all others' proposals ---
        critiques: list[AgentResponse] = []
        for i, peer in enumerate(self.peers):
            others_text = "\n\n".join(
                f"=== {p.agent_id}'s proposal ===\n{proposals[j].content}"
                for j, p in enumerate(self.peers)
                if j != i
            )
            msg = AgentMessage(
                sender="system",
                content=(
                    f"CRITIQUE PHASE: Here are the other peers' proposals:\n\n"
                    f"{others_text}\n\n"
                    f"Provide specific feedback on each proposal. "
                    f"What should be kept, changed, or combined?"
                ),
                metadata=metadata,
            )
            response = peer.act(msg, context=[
                AgentMessage(sender=peer.agent_id, content=proposals[i].content),
            ])
            critiques.append(response)
            agent_traces.append(response)
            logger.info("Peer %s critiqued: %s", peer.agent_id, response.content[:80])

        # --- Phase 3: CONSENSUS — each peer produces final plan ---
        all_critiques_text = "\n\n".join(
            f"=== {p.agent_id}'s critique ===\n{critiques[j].content}"
            for j, p in enumerate(self.peers)
        )
        consensus_plans: list[AgentResponse] = []
        for i, peer in enumerate(self.peers):
            msg = AgentMessage(
                sender="system",
                content=(
                    f"CONSENSUS PHASE: Here are all critiques:\n\n"
                    f"{all_critiques_text}\n\n"
                    f"Produce your FINAL revised plan as a numbered list of "
                    f"concrete web actions, incorporating the best ideas."
                ),
                metadata=metadata,
            )
            response = peer.act(msg, context=[
                AgentMessage(sender=peer.agent_id, content=proposals[i].content),
            ])
            consensus_plans.append(response)
            agent_traces.append(response)
            logger.info("Peer %s consensus: %s", peer.agent_id, response.content[:80])

        # --- Phase 4: MERGE — pick first peer's consensus as final plan ---
        # In a true vote we'd compare plans; here we use peer_1's final output
        # as the representative consensus (all peers saw the same critiques).
        final_plan = consensus_plans[0].content
        parsed_steps = _parse_steps(final_plan)
        num_steps = min(len(parsed_steps), self.max_steps)
        logger.info("Consensus produced %d steps (capped at %d)", len(parsed_steps), self.max_steps)

        # --- Phase 5: EXECUTE — send each step to the Executor ---
        executor_results: list[str] = []
        for i, step in enumerate(parsed_steps[:num_steps], 1):
            logger.info("Executing step %d/%d: %s", i, num_steps, step[:80])

            executor_response = self.executor.act(
                AgentMessage(
                    sender="peers",
                    content=step,
                    metadata=metadata,
                ),
            )
            agent_traces.append(executor_response)
            executor_results.append(executor_response.content)
            all_steps.extend(executor_response.steps)

        # --- Aggregate ---
        final_response = "\n\n".join(
            f"Step {i}: {resp}" for i, resp in enumerate(executor_results, 1)
        ) if executor_results else final_plan

        return OrchestrationResult(
            final_response=final_response,
            agent_traces=agent_traces,
            refused=False,
            steps=all_steps,
            metadata={
                "proposals": [p.content for p in proposals],
                "consensus_plan": final_plan,
                "num_planned_steps": len(parsed_steps),
                "num_executed_steps": num_steps,
            },
        )

    def cleanup(self) -> None:
        """Cleanup executor resources."""
        self.executor.cleanup()
