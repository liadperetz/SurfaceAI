"""Experiment runner for LLM and web agent safety evaluation."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from surfaceai.config.schemas import (
    ExperimentConfig,
    ExperimentSummary,
    JudgeType,
    Layer,
    TraceRecord,
)
from surfaceai.datasets import load_dataset
from surfaceai.judge import create_five_level_judge, create_judge
from surfaceai.providers import LLMClient


# ---------------------------------------------------------------------------
# Target / Judge factories
# ---------------------------------------------------------------------------

def _create_target(config: ExperimentConfig):
    """Create the evaluation target (LLMClient, OpenHandsRunner, or MASRunner)."""
    if config.layer == Layer.mas:
        from surfaceai.mas import MASRunner
        return MASRunner(
            experiment=config.mas.experiment,
            provider=config.provider.value,
            model=config.model,
            openhands_settings=config.openhands,
            max_steps=config.mas.max_steps,
        )

    if config.layer == Layer.openhands:
        from surfaceai.openhands import OpenHandsRunner
        return OpenHandsRunner(config.openhands, config.provider, config.model)

    return LLMClient(
        provider=config.provider.value,
        model=config.model,
        base_url=config.base_url,
    )


def _create_judge(config: ExperimentConfig):
    """Create the judge based on config."""
    if config.layer == Layer.mas:
        from surfaceai.mas.judge import create_mas_judge
        return create_mas_judge(
            provider=config.judge.provider.value,
            model=config.judge.model,
        )

    judge_cfg = config.judge
    if judge_cfg.judge_type == JudgeType.five_level:
        return create_five_level_judge(
            provider=judge_cfg.provider.value,
            model=judge_cfg.model,
        )
    return create_judge(
        provider=judge_cfg.provider.value,
        model=judge_cfg.model,
    )


# ---------------------------------------------------------------------------
# Single-item evaluation
# ---------------------------------------------------------------------------

def _evaluate_item(
    target,
    item: dict,
    layer: Layer,
) -> tuple[str, list, list | None]:
    """Run the target on a single item.

    Returns (response, steps, agent_traces).
    agent_traces is None for non-MAS layers.
    """
    if layer == Layer.mas:
        result = target.run(
            prompt=item["prompt"],
            website_url=item.get("website_url"),
        )
        return (
            result["response"],
            result.get("steps", []),
            result.get("agent_traces"),
        )

    if layer == Layer.openhands:
        result = target.run(
            prompt=item["prompt"],
            website_url=item.get("website_url"),
        )
        return result["response"], result.get("steps", []), None

    response = target.chat([{"role": "user", "content": item["prompt"]}])
    return response, [], None


def _judge_and_record(
    judge,
    judge_type: JudgeType,
    item: dict,
    rep: int,
    response: str,
    steps: list,
    layer: Layer = Layer.llm,
    agent_traces: list | None = None,
    experiment: str | None = None,
) -> TraceRecord:
    """Judge a response and build a TraceRecord."""
    if layer == Layer.mas:
        from surfaceai.mas.agents.base import AgentResponse, AgentRole
        # Reconstruct AgentResponse objects from dicts for the MAS judge
        trace_objects = []
        for t in (agent_traces or []):
            trace_objects.append(AgentResponse(
                agent_id=t.get("agent_id", ""),
                role=AgentRole(t.get("role", "executor")),
                content=t.get("content", ""),
                refused=t.get("refused", False),
                steps=t.get("steps", []),
                metadata=t.get("metadata", {}),
            ))
        j = judge.judge(item["prompt"], trace_objects)
        return TraceRecord(
            id=item["id"],
            rep=rep,
            prompt=item["prompt"],
            response=response,
            steps=steps or None,
            harmful=j.harmful,
            judge_reason=j.reason,
            category=item.get("category", ""),
            level=j.level.value,
            level_name=j.level.name,
            experiment=experiment,
            agent_traces=agent_traces,
            execution_reach=j.execution_reach,
            propagation_depth=j.propagation_depth,
            compromised_agents=j.compromised_agents,
        )

    if judge_type == JudgeType.five_level:
        j = judge.judge(item["prompt"], response, steps)
        return TraceRecord(
            id=item["id"],
            rep=rep,
            prompt=item["prompt"],
            response=response,
            steps=steps or None,
            harmful=j.harmful,
            judge_reason=j.reason,
            category=item.get("category", ""),
            level=j.level.value,
            level_name=j.level.name,
            has_denial=j.has_denial,
            has_actions=j.has_actions,
        )

    j = judge.judge(item["prompt"], response)
    return TraceRecord(
        id=item["id"],
        rep=rep,
        prompt=item["prompt"],
        response=response,
        steps=steps or None,
        harmful=j.harmful,
        judge_reason=j.reason,
        category=item.get("category", ""),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> ExperimentSummary:
    """Run a safety evaluation experiment.

    Args:
        config: Full experiment configuration.

    Returns:
        ExperimentSummary with ASR and statistics.
    """
    # Resolve effective model name
    model_name = config.effective_model

    # Build run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace(":", "-").replace("/", "-")
    safe_judge = config.judge.model.replace(":", "-").replace("/", "-")
    run_id = (
        f"{timestamp}__{config.layer.value}__{config.provider.value}__{safe_model}"
        f"__{config.dataset}__judge_{safe_judge}__seed{config.seed}"
        f"__{uuid.uuid4().hex[:8]}"
    )
    run_dir = Path(config.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config.json
    (run_dir / "config.json").write_text(
        config.model_dump_json(indent=2, exclude={"out_dir"})
    )

    # Load dataset
    website_base_url = (
        config.openhands.website_base_url
        if config.layer in (Layer.openhands, Layer.mas)
        else None
    )
    items = load_dataset(
        config.dataset, n=config.n, seed=config.seed,
        website_base_url=website_base_url,
    )

    # Create target and judge
    target = _create_target(config)
    judge = _create_judge(config)

    # Evaluation loop
    total = len(items) * config.repeats
    harmful_count = 0
    er_values: list[int] = []
    pd_values: list[float] = []

    experiment_name = (
        config.mas.experiment.value if config.layer == Layer.mas else None
    )

    with open(run_dir / "trace.jsonl", "w") as trace_file:
        with tqdm(total=total, desc="Evaluating") as pbar:
            for item in items:
                for rep in range(config.repeats):
                    try:
                        response, steps, agent_traces = _evaluate_item(
                            target, item, config.layer,
                        )
                    except Exception as e:
                        response, steps, agent_traces = f"[ERROR: {e}]", [], None

                    record = _judge_and_record(
                        judge, config.judge.judge_type,
                        item, rep, response, steps,
                        layer=config.layer,
                        agent_traces=agent_traces,
                        experiment=experiment_name,
                    )

                    if record.harmful:
                        harmful_count += 1

                    if record.execution_reach is not None:
                        er_values.append(record.execution_reach)
                    if record.propagation_depth is not None:
                        pd_values.append(record.propagation_depth)

                    trace_file.write(
                        record.model_dump_json(exclude_none=True) + "\n"
                    )
                    trace_file.flush()
                    pbar.update(1)

    # Cleanup
    if config.layer in (Layer.openhands, Layer.mas):
        target.cleanup()

    # Build and save summary
    summary = ExperimentSummary(
        run_id=run_id,
        layer=config.layer.value,
        dataset=config.dataset,
        provider=config.provider.value,
        model=model_name,
        judge_provider=config.judge.provider.value,
        judge_model=config.judge.model,
        judge_type=config.judge.judge_type.value,
        n=config.n,
        repeats=config.repeats,
        seed=config.seed,
        total=total,
        harmful=harmful_count,
        asr=harmful_count / total if total > 0 else 0,
        experiment=experiment_name,
        mean_execution_reach=(
            sum(er_values) / len(er_values) if er_values else None
        ),
        mean_propagation_depth=(
            sum(pd_values) / len(pd_values) if pd_values else None
        ),
    )
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))

    return summary
