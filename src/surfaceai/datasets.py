"""Dataset loaders for safety evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal, Optional

DatasetName = Literal["browserart", "do_not_answer", "jailbreakbench"]

DATA_DIR = Path("data")

DATASET_FILES = {
    "browserart": "browserart_100.json",
    "do_not_answer": "do_not_answer_100.json",
    "jailbreakbench": "jailbreakbench_100.json",
}

# Map BrowserART website identifiers to paths
WEBSITE_PATHS = {
    "local:email": "/email/index.html",
    "local:facebook": "/facebook/index.html",
    "local:instagram": "/instagram/index.html",
    "local:linkedin": "/linkedin/index.html",
    "local:twitter": "/twitter/index.html",
    "local:tiktok": "/tiktok/index.html",
    "local:tinder": "/tinder/index.html",
    "local:forms": "/forms/index.html",
    "local:login": "/login/index.html",
    "local:hospital": "/hospital/index.html",
    "local:dmv": "/dmv/index.html",
    "local:ec2": "/ec2/admin.html",
    "local:github": "/github/pr.html",
    "local:gitlab": "/gitlab_issue/index.html",
    "local:notion": "/notion/index.html",
    "local:google": "/google/index.html",
    "local:generic": "/generic/index.html",
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


def _resolve_website_url(website: Optional[str], base_url: Optional[str]) -> Optional[str]:
    """Resolve a website identifier to a full URL.

    Args:
        website: Website identifier (e.g., 'local:email')
        base_url: Base URL for local websites (e.g., 'http://host.docker.internal:8080')

    Returns:
        Full URL or None if not resolvable
    """
    if not website or not base_url:
        return None

    # Handle local: prefixed websites
    if website.startswith("local:"):
        path = WEBSITE_PATHS.get(website)
        if path:
            return f"{base_url.rstrip('/')}{path}"
        # Try to construct path from website name
        site_name = website.replace("local:", "")
        return f"{base_url.rstrip('/')}/{site_name}/index.html"

    # Return external URLs as-is
    if website.startswith("http://") or website.startswith("https://"):
        return website

    return None


def load_dataset(
    name: DatasetName,
    n: int,
    seed: int,
    website_base_url: Optional[str] = None,
) -> list[dict]:
    """Load a dataset by name.

    Args:
        name: Dataset name (browserart, do_not_answer, jailbreakbench)
        n: Number of samples to load
        seed: Random seed for reproducibility
        website_base_url: Base URL for local websites (for BrowserART with OpenHands)

    Returns:
        List of items with 'id', 'prompt', 'category', and optionally 'website_url' keys
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
        entry = {
            "id": f"{name}_{item['id']}",
            "prompt": prompt,
            "category": item.get("category", item.get("semantic_category", "")),
        }

        # Resolve website URL for BrowserART
        website = item.get("website")
        if website:
            entry["website_url"] = _resolve_website_url(website, website_base_url)

        items.append(entry)

    return items
