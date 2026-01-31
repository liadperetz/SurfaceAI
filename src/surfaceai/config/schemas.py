"""Configuration schemas for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Dict, Any, Optional


# Type aliases
LayerName = Literal["llm"]  # Phase 1 only
DatasetName = Literal["do_not_answer", "jailbreakbench", "browserart"]
ProviderName = Literal["ollama", "openai", "groq", "deepseek"]


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider: ProviderName = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    max_tokens: int = 512
    base_url: Optional[str] = None  # Override default base URL


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for the judge (uses same provider abstraction)."""
    provider: ProviderName = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 256
    use_llm_classifier: bool = True  # False = heuristics only


@dataclass
class RunConfig:
    """Main configuration for an experiment run."""
    layer: LayerName
    dataset: DatasetName

    # Target model (the model being evaluated)
    target: ProviderConfig = field(default_factory=ProviderConfig)

    # Judge model (for classifying outputs)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    # Experiment settings
    n: int = 100
    repeats: int = 5
    seed: int = 42
    out_dir: str = "runs"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)
