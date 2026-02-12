"""Sequential chain orchestrator for E5: linear pipeline topology.

Agents are arranged in a fixed sequence. Each agent's output becomes the
next agent's input. The final node's output is the action plan, which is
then executed by the Executor.

Chain: Task → Node 1 (Interpreter) → Node 2 (Strategist) → Node 3 (Action Planner) → Executor

This tests whether safety-relevant context degrades as decisions propagate
through a chain.
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


class SequentialChainOrchestrator:
    """Orchestrator for E5: Sequential Action Chain (linear pipeline).

    Each chain node receives only the output of the previous node — it
    never sees the original task or outputs from non-adjacent nodes.
    The final node produces an action plan executed by the Executor.
    """

    def __init__(
        self,
        chain_nodes: list[LLMAgent],
        executor: OpenHandsAgent,
        max_steps: int = 10,
    ):
        self.chain_nodes = chain_nodes
        self.executor = executor
        self.max_steps = max_steps

    def run(
        self,
        task: str,
        website_url: Optional[str] = None,
    ) -> OrchestrationResult:
        """Execute the sequential chain workflow."""
        agent_traces: list[AgentResponse] = []
        all_steps: list[dict] = []
        metadata: dict = {"website_url": website_url} if website_url else {}

        # --- Chain propagation: each node sees only the previous output ---
        current_input = task
        node_outputs: list[str] = []

        for i, node in enumerate(self.chain_nodes):
            logger.info(
                "Chain node %s processing: %s",
                node.agent_id, current_input[:80],
            )

            response = node.act(
                AgentMessage(
                    sender=self.chain_nodes[i - 1].agent_id if i > 0 else "user",
                    content=current_input,
                    metadata=metadata,
                ),
            )
            agent_traces.append(response)
            node_outputs.append(response.content)

            logger.info(
                "Chain node %s output: %s",
                node.agent_id, response.content[:80],
            )

            # Output becomes next node's input — context fragmentation
            current_input = response.content

        # --- Parse final node's output into executable steps ---
        final_plan = current_input  # Last node's output
        parsed_steps = _parse_steps(final_plan)
        num_steps = min(len(parsed_steps), self.max_steps)
        logger.info(
            "Chain produced %d steps (capped at %d)",
            len(parsed_steps), self.max_steps,
        )

        # --- Execute each step ---
        executor_results: list[str] = []
        for i, step in enumerate(parsed_steps[:num_steps], 1):
            logger.info("Executing step %d/%d: %s", i, num_steps, step[:80])

            executor_response = self.executor.act(
                AgentMessage(
                    sender="chain",
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
                "chain_outputs": node_outputs,
                "final_plan": final_plan,
                "num_planned_steps": len(parsed_steps),
                "num_executed_steps": num_steps,
            },
        )

    def cleanup(self) -> None:
        """Cleanup executor resources."""
        self.executor.cleanup()
