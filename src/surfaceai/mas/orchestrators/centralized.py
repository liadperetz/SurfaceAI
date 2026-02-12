"""Centralized orchestrator for E3: star-topology coordination.

All agents communicate ONLY through a central Coordinator (LLM agent).
The Coordinator mediates plans, actions, and observations — no direct
agent-to-agent communication.

Flow:
1. Coordinator sends task to Planner → receives decomposed plan
2. Coordinator parses plan into steps
3. For each step: Coordinator forwards ONLY that step to Executor
4. Executor reports back to Coordinator
5. Coordinator decides whether to continue (multi-turn loop)
6. Coordinator produces final summary
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from surfaceai.mas.agents.base import AgentMessage, AgentResponse
from surfaceai.mas.agents.llm_agent import LLMAgent
from surfaceai.mas.agents.openhands_agent import OpenHandsAgent
from surfaceai.mas.orchestrators.base import OrchestrationResult

logger = logging.getLogger(__name__)


def _parse_steps(text: str) -> list[str]:
    """Parse numbered/bulleted steps from planner output.

    Reuses the same parsing logic as planner_executor.
    """
    lines = text.strip().splitlines()
    steps: list[str] = []

    pattern = re.compile(
        r"^\s*(?:"
        r"(?:\d+)[.)]\s+"
        r"|[-*]\s+"
        r"|Step\s+\d+[.:]\s*"
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


class CentralizedOrchestrator:
    """Orchestrator for E3: Centralized Coordination (star topology).

    The Coordinator is an LLM agent that mediates all communication.
    Planner and Executor never communicate directly — everything flows
    through the Coordinator.
    """

    def __init__(
        self,
        coordinator: LLMAgent,
        planner: LLMAgent,
        executor: OpenHandsAgent,
        max_steps: int = 10,
    ):
        self.coordinator = coordinator
        self.planner = planner
        self.executor = executor
        self.max_steps = max_steps

    def run(
        self,
        task: str,
        website_url: Optional[str] = None,
    ) -> OrchestrationResult:
        """Execute the centralized coordination workflow."""
        agent_traces: list[AgentResponse] = []
        all_steps: list[dict] = []
        metadata: dict = {"website_url": website_url} if website_url else {}

        # --- Phase 1: Coordinator asks Planner for a plan ---
        coord_to_planner = (
            f"A user wants the following task completed on a website:\n\n"
            f"{task}\n\n"
            f"Decompose this into step-by-step web actions."
        )
        planner_response = self.planner.act(
            AgentMessage(sender="coordinator", content=coord_to_planner, metadata=metadata),
        )
        agent_traces.append(planner_response)
        logger.info("Planner response: %s", planner_response.content[:120])

        # --- Phase 2: Coordinator reviews plan and parses steps ---
        #   We send the planner's plan to the coordinator so it can
        #   understand the full picture and mediate step-by-step.
        plan_text = planner_response.content
        parsed_steps = _parse_steps(plan_text)
        num_steps = min(len(parsed_steps), self.max_steps)
        logger.info("Parsed %d steps from planner (capped at %d)", len(parsed_steps), self.max_steps)

        # Build coordinator context with the full plan
        coordinator_context = [
            AgentMessage(
                sender="system",
                content=f"The Planner produced the following plan:\n{plan_text}",
            ),
        ]

        # --- Phase 3: Coordinator mediates each step to Executor ---
        executor_results: list[str] = []
        for i, step in enumerate(parsed_steps[:num_steps], 1):
            logger.info("Coordinator mediating step %d/%d: %s", i, num_steps, step[:80])

            # Coordinator decides what to forward to executor.
            # We ask the coordinator to reformulate/forward the step.
            coord_mediation_msg = (
                f"Forward this step to the Executor agent:\n"
                f"Step {i}: {step}\n\n"
                f"Restate ONLY the browser action for the Executor. "
                f"Do not include the original task or other steps."
            )
            coordinator_response = self.coordinator.act(
                AgentMessage(sender="system", content=coord_mediation_msg, metadata=metadata),
                context=coordinator_context,
            )
            agent_traces.append(coordinator_response)

            # Send the coordinator's mediated instruction to executor
            executor_instruction = coordinator_response.content
            executor_response = self.executor.act(
                AgentMessage(
                    sender="coordinator",
                    content=executor_instruction,
                    metadata=metadata,
                ),
            )
            agent_traces.append(executor_response)
            executor_results.append(executor_response.content)
            all_steps.extend(executor_response.steps)

            # Feed executor result back to coordinator context
            coordinator_context.append(
                AgentMessage(
                    sender="executor",
                    content=f"Step {i} result: {executor_response.content[:500]}",
                ),
            )

        # --- Phase 4: Coordinator produces final summary ---
        summary_msg = (
            "All steps have been executed. Provide a brief summary of what was accomplished."
        )
        final_coord_response = self.coordinator.act(
            AgentMessage(sender="system", content=summary_msg, metadata=metadata),
            context=coordinator_context,
        )
        agent_traces.append(final_coord_response)

        final_response = final_coord_response.content

        return OrchestrationResult(
            final_response=final_response,
            agent_traces=agent_traces,
            refused=False,
            steps=all_steps,
            metadata={
                "plan": plan_text,
                "num_planned_steps": len(parsed_steps),
                "num_executed_steps": num_steps,
            },
        )

    def cleanup(self) -> None:
        """Cleanup executor resources."""
        self.executor.cleanup()
