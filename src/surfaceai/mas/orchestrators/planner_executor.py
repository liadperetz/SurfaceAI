"""Planner-Executor orchestrator for E1 and E2 experiments.

The coordinator is deterministic Python logic (not an LLM agent), matching
the paper's "lightweight coordinator" design.

E1: Planner decomposes task -> Executor runs each step in isolation
E2: Planner evaluates safety first -> refuses or decomposes -> Executor runs
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


def _parse_steps(planner_output: str) -> list[str]:
    """Parse numbered/bulleted steps from planner output.

    Supports formats:
    - "1. action" / "2. action"
    - "1) action" / "2) action"
    - "- action" / "* action"
    - "Step 1: action" / "Step 2: action"

    Falls back to treating the entire output as a single step.
    """
    lines = planner_output.strip().splitlines()
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
        # Fallback: treat the whole output as one step
        text = planner_output.strip()
        if text:
            steps.append(text)

    return steps


class PlannerExecutorOrchestrator:
    """Orchestrator for E1 (decomposition) and E2 (safety filtering).

    Flow:
    1. Send full task to Planner -> get decomposed plan
    2. If Planner refused (E2): return immediately with refused=True
    3. Parse numbered steps from Planner output
    4. For each step: send ONLY that step to Executor (context fragmentation)
    5. Collect all Executor responses
    6. Return aggregated OrchestrationResult
    """

    def __init__(
        self,
        planner: LLMAgent,
        executor: OpenHandsAgent,
        max_steps: int = 10,
    ):
        self.planner = planner
        self.executor = executor
        self.max_steps = max_steps

    def run(
        self,
        task: str,
        website_url: Optional[str] = None,
    ) -> OrchestrationResult:
        """Execute the planner-executor workflow."""
        agent_traces: list[AgentResponse] = []
        all_steps: list[dict] = []

        # --- Phase 1: Planning ---
        planner_message = AgentMessage(
            sender="coordinator",
            content=task,
            metadata={"website_url": website_url} if website_url else {},
        )
        planner_response = self.planner.act(planner_message)
        agent_traces.append(planner_response)

        # If planner refused (E2 safety filtering), stop early
        if planner_response.refused:
            logger.info("Planner refused the task (safety filtering)")
            return OrchestrationResult(
                final_response=planner_response.content,
                agent_traces=agent_traces,
                refused=True,
                metadata={"plan": planner_response.content},
            )

        # --- Phase 2: Parse plan into steps ---
        parsed_steps = _parse_steps(planner_response.content)
        num_steps = min(len(parsed_steps), self.max_steps)
        logger.info("Planner produced %d steps (capped at %d)", len(parsed_steps), self.max_steps)

        # --- Phase 3: Execute each step in isolation ---
        executor_responses: list[str] = []
        for i, step in enumerate(parsed_steps[:num_steps], 1):
            logger.info("Executing step %d/%d: %s", i, num_steps, step[:80])

            executor_message = AgentMessage(
                sender="coordinator",
                content=step,
                metadata={"website_url": website_url} if website_url else {},
            )
            executor_response = self.executor.act(executor_message)
            agent_traces.append(executor_response)
            executor_responses.append(executor_response.content)
            all_steps.extend(executor_response.steps)

        # --- Phase 4: Aggregate results ---
        final_response = "\n\n".join(
            f"Step {i}: {resp}" for i, resp in enumerate(executor_responses, 1)
        ) if executor_responses else planner_response.content

        return OrchestrationResult(
            final_response=final_response,
            agent_traces=agent_traces,
            refused=False,
            steps=all_steps,
            metadata={
                "plan": planner_response.content,
                "num_planned_steps": len(parsed_steps),
                "num_executed_steps": num_steps,
            },
        )

    def cleanup(self) -> None:
        """Cleanup executor resources."""
        self.executor.cleanup()
