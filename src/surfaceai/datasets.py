"""Dataset loaders for safety evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

DatasetName = Literal["browserart", "do_not_answer", "jailbreakbench"]

DATA_DIR = Path("data")

DATASET_FILES = {
    "browserart": "browserart_100.json",
    "do_not_answer": "do_not_answer_100.json",
    "jailbreakbench": "jailbreakbench_100.json",
}


def _resolve_data_path(filename: str) -> Path:
    """Resolve data file path."""
    path = DATA_DIR / filename
    if path.exists():
        return path

    # Try relative to module
    module_path = Path(__file__).parent.parent.parent / "data" / filename
    if module_path.exists():
        return module_path

    raise FileNotFoundError(f"Data file not found: {filename}")


def load_dataset(name: DatasetName, n: int, seed: int) -> list[dict]:
    """Load a dataset by name.

    Args:
        name: Dataset name (browserart, do_not_answer, jailbreakbench)
        n: Number of samples to load
        seed: Random seed for reproducibility

    Returns:
        List of items with 'id', 'prompt', and 'category' keys
    """
    if name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_FILES.keys())}")

    path = _resolve_data_path(DATASET_FILES[name])

    with open(path) as f:
        data = json.load(f)

    n = min(n, len(data))
    random.seed(seed)
    sampled = random.sample(data, n)

    # Normalize to common format
    items = []
    for item in sampled:
        # Handle browserart format (uses 'behavior' instead of 'prompt')
        prompt = item.get("prompt") or item.get("behavior", "")
        items.append({
            "id": f"{name}_{item['id']}",
            "prompt": prompt,
            "category": item.get("category", item.get("semantic_category", "")),
        })

    return items
