"""Pydantic schemas for experiment configuration, results, and trace records."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Layer(str, Enum):
    llm = "llm"
    openhands = "openhands"
    mas = "mas"


class MASExperiment(str, Enum):
    e1 = "e1"  # Planner-Executor Decomposition
    e2 = "e2"  # Planner with Safety Filtering
    e3 = "e3"  # placeholder
    e4 = "e4"  # placeholder
    e5 = "e5"  # placeholder
    e6 = "e6"  # placeholder
    e7 = "e7"  # placeholder
    e8 = "e8"  # placeholder
    e9 = "e9"  # placeholder


class Provider(str, Enum):
    ollama = "ollama"
    openai = "openai"
    groq = "groq"
    deepseek = "deepseek"


class JudgeType(str, Enum):
    binary = "binary"
    five_level = "five_level"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class JudgeConfig(BaseModel):
    """Judge LLM settings."""
    provider: Provider = Provider.openai
    model: str = "gpt-4o-mini"
    judge_type: JudgeType = JudgeType.binary


class OpenHandsSettings(BaseModel):
    """OpenHands-specific settings (only used when layer=openhands)."""
    api_host: str = "localhost"
    api_port: int = 3000
    website_host: str = "host.docker.internal"
    website_port: int = 8080
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    docker_compose_dir: str = "docker"
    container_startup_timeout: int = 60
    runtime_startup_timeout: int = 120
    response_timeout: int = 180
    poll_interval: float = 2.0
    auto_start_container: bool = True
    auto_stop_container: bool = False

    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def website_base_url(self) -> str:
        return f"http://{self.website_host}:{self.website_port}"


class MASConfig(BaseModel):
    """MAS experiment settings (only used when layer=mas)."""
    experiment: MASExperiment = MASExperiment.e1
    max_steps: int = 10


# ---------------------------------------------------------------------------
# Experiment config  (this is what gets saved to config.json)
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Full experiment configuration."""
    layer: Layer
    dataset: str
    provider: Provider
    model: Optional[str] = None
    base_url: Optional[str] = None
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    openhands: OpenHandsSettings = Field(default_factory=OpenHandsSettings)
    mas: MASConfig = Field(default_factory=MASConfig)
    n: int = 100
    repeats: int = 5
    seed: int = 42
    out_dir: str = "runs"

    @property
    def effective_model(self) -> str:
        if self.model:
            return self.model
        defaults = {
            Provider.ollama: "llama3.1:8b",
            Provider.openai: "gpt-4o-mini",
            Provider.groq: "llama-3.1-8b-instant",
            Provider.deepseek: "deepseek-chat",
        }
        return defaults[self.provider]


# ---------------------------------------------------------------------------
# Trace records  (each line in trace.jsonl)
# ---------------------------------------------------------------------------

class TraceRecord(BaseModel):
    """Single evaluation record written to trace.jsonl."""
    id: str
    rep: int
    prompt: str
    response: str
    harmful: bool
    judge_reason: str
    category: str = ""
    steps: Optional[list[dict[str, Any]]] = None
    # Five-level fields (only present when judge_type=five_level)
    level: Optional[int] = None
    level_name: Optional[str] = None
    has_denial: Optional[bool] = None
    has_actions: Optional[bool] = None
    # MAS fields (only present when layer=mas)
    experiment: Optional[str] = None
    agent_traces: Optional[list[dict[str, Any]]] = None
    execution_reach: Optional[int] = None
    propagation_depth: Optional[float] = None
    compromised_agents: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Experiment summary  (saved to summary.json)
# ---------------------------------------------------------------------------

class ExperimentSummary(BaseModel):
    """Aggregated experiment results."""
    run_id: str
    layer: str
    dataset: str
    provider: str
    model: str
    judge_provider: str
    judge_model: str
    judge_type: str
    n: int
    repeats: int
    seed: int
    total: int
    harmful: int
    asr: float
    # MAS fields (only present when layer=mas)
    experiment: Optional[str] = None
    mean_execution_reach: Optional[float] = None
    mean_propagation_depth: Optional[float] = None
