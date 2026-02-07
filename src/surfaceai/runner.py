"""Experiment runner for LLM safety evaluation."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from surfaceai.datasets import load_dataset
from surfaceai.providers import LLMClient
from surfaceai.judge import Judge, create_judge


def run_experiment(
    layer: str,
    dataset: str,
    provider: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o-mini",
    n: int = 100,
    repeats: int = 5,
    seed: int = 42,
    out_dir: str = "runs",
) -> dict:
    """Run a safety evaluation experiment.

    Args:
        layer: Execution layer (llm, openhands, mas)
        dataset: Dataset name (browserart, do_not_answer, jailbreakbench)
        provider: Target LLM provider
        model: Target model ID (uses provider default if None)
        base_url: Custom API base URL (for Ollama on different ports)
        judge_provider: Judge LLM provider
        judge_model: Judge model ID
        n: Number of samples from dataset
        repeats: Number of times to repeat each sample
        seed: Random seed for reproducibility
        out_dir: Output directory

    Returns:
        Summary dict with ASR and statistics
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = (model or "default").replace(":", "-").replace("/", "-")
    judge_name = judge_model.replace(":", "-").replace("/", "-")
    run_id = f"{timestamp}__{layer}__{provider}__{model_name}__{dataset}__judge_{judge_name}__seed{seed}__{uuid.uuid4().hex[:8]}"
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize clients
    target = LLMClient(provider=provider, model=model, base_url=base_url)
    judge = create_judge(provider=judge_provider, model=judge_model)

    # Load dataset
    items = load_dataset(dataset, n=n, seed=seed)

    # Save config
    config = {
        "layer": layer,
        "dataset": dataset,
        "provider": provider,
        "model": target.model,
        "base_url": base_url,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "n": n,
        "repeats": repeats,
        "seed": seed,
        "run_id": run_id,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Run evaluations
    trace_path = run_dir / "trace.jsonl"
    total_evaluations = len(items) * repeats
    harmful_count = 0

    with open(trace_path, "w") as trace_file:
        with tqdm(total=total_evaluations, desc="Evaluating") as pbar:
            for item in items:
                for rep in range(repeats):
                    # Get response from target model
                    try:
                        response = target.chat([{"role": "user", "content": item["prompt"]}])
                    except Exception as e:
                        response = f"[ERROR: {e}]"

                    # Judge the response
                    judgment = judge.judge(item["prompt"], response)

                    if judgment.harmful:
                        harmful_count += 1

                    # Write trace record
                    record = {
                        "id": item["id"],
                        "rep": rep,
                        "prompt": item["prompt"],
                        "response": response,
                        "harmful": judgment.harmful,
                        "judge_reason": judgment.reason,
                        "category": item.get("category", ""),
                    }
                    trace_file.write(json.dumps(record) + "\n")
                    trace_file.flush()
                    pbar.update(1)

    # Calculate summary
    asr = harmful_count / total_evaluations if total_evaluations > 0 else 0
    summary = {
        "run_id": run_id,
        "layer": layer,
        "dataset": dataset,
        "provider": provider,
        "model": target.model,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "n": n,
        "repeats": repeats,
        "seed": seed,
        "total": total_evaluations,
        "harmful": harmful_count,
        "asr": asr,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary
