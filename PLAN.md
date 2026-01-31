# SurfaceAI Production Refactor Plan

## Overview

Refactor `surfaceai-old` into `surfaceai-prod` - a clean, production-ready CLI for evaluating AI safety across execution layers. This supports the research paper "Architectural Causes of Jailbreak Vulnerabilities in Multi-Agent Systems."

## Target Architecture

```
surfaceai-prod/
├── pyproject.toml              # Modern packaging with Typer CLI
├── README.md                   # Installation & usage guide
├── .env.example                # API key template
├── src/surfaceai/
│   ├── __init__.py
│   ├── __main__.py             # Entry: python -m surfaceai
│   ├── cli.py                  # Typer CLI
│   ├── config/
│   │   ├── settings.py         # Pydantic settings (.env support)
│   │   └── schemas.py          # RunConfig dataclasses
│   ├── providers/              # Multi-provider LLM abstraction
│   │   ├── base.py             # LLMProvider protocol
│   │   ├── ollama.py
│   │   ├── openai.py
│   │   ├── groq.py
│   │   ├── deepseek.py
│   │   └── registry.py
│   ├── datasets/
│   │   ├── base.py             # Dataset protocol
│   │   ├── browserart.py
│   │   ├── do_not_answer.py
│   │   ├── jailbreakbench.py
│   │   └── registry.py
│   ├── judges/
│   │   ├── base.py             # Judge protocol
│   │   ├── five_level.py       # 5-level judge (uses any provider)
│   │   └── registry.py
│   ├── layers/
│   │   ├── base.py             # Layer protocol
│   │   ├── llm.py              # Direct LLM baseline
│   │   ├── openhands.py        # Web agent (Phase 2)
│   │   ├── mas/                # MAS experiments (Phase 3)
│   │   └── registry.py
│   ├── output/
│   │   ├── trace.py            # JSONL trace writer
│   │   └── summary.py          # Summary generation
│   └── runner.py               # Experiment orchestrator
├── data/                       # Dataset files
├── docker/                     # OpenHands compose
├── web/                        # Mock website server
│   ├── server.py               # Flask server
│   └── websites/               # 30 mock websites for BrowserART
└── tests/
```

## CLI Design (Typer)

```bash
# LLM Baseline with different providers
surfaceai run --layer llm --dataset browserart --provider ollama --model llama3.1:8b
surfaceai run --layer llm --dataset do_not_answer --provider openai --model gpt-4o-mini
surfaceai run --layer llm --dataset jailbreakbench --provider groq --model llama-3.1-8b-instant
surfaceai run --layer llm --dataset browserart --provider deepseek --model deepseek-chat

# Configurable judge (any provider/model can be the judge)
surfaceai run --layer llm --dataset browserart \
  --provider ollama --model llama3.1:8b \
  --judge-provider openai --judge-model gpt-4o-mini   # Default: OpenAI judge

surfaceai run --layer openhands --dataset browserart \
  --judge-provider ollama --judge-model llama3.1:8b   # Llama as judge

surfaceai run --layer openhands --dataset browserart \
  --judge-provider groq --judge-model llama-3.1-70b-versatile  # Groq as judge

# Web Agent (Phase 2)
surfaceai run --layer openhands --dataset browserart --openhands-backend openai

# MAS Experiments (Phase 3)
surfaceai run --layer mas --experiment exp1 --dataset browserart

# Utility commands
surfaceai providers list
surfaceai providers test ollama --model llama3.1:8b
surfaceai datasets list
surfaceai results summary runs/

# Website server for BrowserART
surfaceai serve --port 5000   # Start mock website server
```

## Judge Configuration

The Five-level judge uses an LLM to classify harmful content. Users can choose any provider/model as the judge:

| Option | Default | Description |
|--------|---------|-------------|
| `--judge-provider` | openai | LLM provider for judge (ollama/openai/groq/deepseek) |
| `--judge-model` | gpt-4o-mini | Model ID for the judge |

**Examples:**
- `--judge-provider openai --judge-model gpt-4o-mini` (default, fast & cheap)
- `--judge-provider ollama --judge-model llama3.1:8b` (local, free)
- `--judge-provider groq --judge-model llama-3.1-70b-versatile` (fast API)
- `--judge-provider deepseek --judge-model deepseek-chat` (alternative)

The judge reuses the same provider abstraction as the target model, so adding new providers automatically makes them available as judges.

## Multi-Provider LLM Support

All providers implement the same `LLMProvider` protocol:

| Provider | API | Default Model | Auth |
|----------|-----|---------------|------|
| Ollama | OpenAI-compatible | llama3.1:8b | None (local) |
| OpenAI | Native | gpt-4o-mini | OPENAI_API_KEY |
| Groq | OpenAI-compatible | llama-3.1-8b-instant | GROQ_API_KEY |
| DeepSeek | OpenAI-compatible | deepseek-chat | DEEPSEEK_API_KEY |

## Implementation Phases

### Phase 1: Core + LLM Baseline (Priority)
1. Package setup (`pyproject.toml`, structure)
2. Configuration (`settings.py` with Pydantic, `schemas.py`)
3. Provider abstraction (all 4 providers)
4. CLI skeleton with Typer
5. LLM layer (adapt from `llm_direct.py`)
6. Datasets (copy all 3)
7. Five-level judge (adapt from existing, uses provider abstraction)
8. Runner + output (trace.jsonl, summary.json)

**Deliverable**: `surfaceai run --layer llm` works with all providers

### Phase 2: Web Agent
1. OpenHands runner (adapt from `openhands_runner.py` - 668 lines)
2. Docker compose for backends
3. CLI integration

**Deliverable**: `surfaceai run --layer openhands` works

### Phase 3: MAS Experiments
1. Orchestrator (adapt from `mas_exp/orchestrator.py`)
2. Agents (text_agent, web_agent)
3. Experiment configs (exp1-exp9, a1-a6)

**Deliverable**: `surfaceai run --layer mas --experiment exp1` works

## Files to Copy vs Rewrite

### Copy & Adapt
- `judge/five_level_judge.py` → `judges/five_level.py` (use provider abstraction)
- `layers/openhands_runner.py` → `layers/openhands.py` (clean up)
- `layers/mas_exp/*` → `layers/mas/*` (use provider abstraction)
- `datasets/*.py` → `datasets/*.py` (minimal changes)
- `tests/test_five_level_judge.py` → `tests/test_judges.py`

### Rewrite from Scratch
- `cli.py` (argparse → Typer)
- `config/settings.py` (Pydantic settings)
- `providers/*.py` (new abstraction)
- `runner.py` (simplified)

### Remove (Unused)
- `layers/single_agent.py` (prototype)
- `layers/mas/*.py` (legacy AutoGen, replaced by mas_exp)
- `judge/llama_guard.py` (five-level now supports any model as judge)
- `judge/openai_judge.py` (consolidated into five-level with provider abstraction)

### Copy (Website Server)
- `web/server.py` → `web/server.py` (Flask server for mock websites)
- `web/websites/*` → `web/websites/*` (30 mock websites for BrowserART)

## Package Management (uv)

We use `uv` for fast, reliable package management. It reads the standard `pyproject.toml`.

**Installation:**
```bash
# Clone and setup
git clone git@github.com:liadperetz/SurfaceAI.git
cd SurfaceAI

# Create venv and install (using uv)
uv venv --python 3.11
uv pip install -e .

# With optional dependencies
uv pip install -e ".[dev]"      # Development tools
uv pip install -e ".[groq]"     # Groq provider
uv pip install -e ".[openhands]" # OpenHands/Docker support
```

**pyproject.toml dependencies:**
```toml
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",           # Pretty CLI output
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "openai>=1.0.0",          # All providers use OpenAI-compatible API
    "pandas>=2.0.0",
    "tqdm>=4.66.0",
    "requests>=2.31.0",
    "flask>=3.0.0",           # Website server
]

[project.optional-dependencies]
groq = ["groq>=0.4.0"]
openhands = ["docker>=7.0.0"]
dev = ["pytest>=7.0.0", "ruff>=0.1.0"]
```

## Output Format

**trace.jsonl** (per-evaluation record):
```json
{"rep": 0, "prompt_id": "42", "prompt": "...", "final_text": "...",
 "harmful": false, "harm_level": 1, "harm_level_name": "CLEAR_DENIAL",
 "judge_type": "five-level", "layer": "llm", "provider": "ollama", "model": "llama3.1:8b"}
```

**summary.json** (aggregated):
```json
{"total": 500, "harmful": 45, "asr": 0.09, "layer": "llm", "model": "llama3.1:8b",
 "harm_level_distribution": {"CLEAR_DENIAL": 320, ...}}
```

## Verification

After implementation, verify with:

**Phase 1 - LLM Baseline:**
```bash
# Test Ollama provider
surfaceai run --layer llm --dataset do_not_answer --provider ollama --model llama3.1:8b -n 5 --repeats 1

# Test OpenAI provider
surfaceai run --layer llm --dataset do_not_answer --provider openai --model gpt-4o-mini -n 5 --repeats 1

# Check output
cat runs/*/summary.json
```

**Phase 2 - Web Agent:**
```bash
# Start website server
surfaceai serve --port 5000 &

# Run OpenHands experiment
surfaceai run --layer openhands --dataset browserart --openhands-backend openai -n 5 --repeats 1
```

**Phase 3 - MAS:**
```bash
surfaceai run --layer mas --experiment exp1 --dataset browserart -n 5 --repeats 1
```

**Unit tests:**
```bash
pytest tests/ -v
```

## Testing Strategy

### Unit Tests (tests/)

**Provider Tests (`test_providers.py`):**
```python
def test_ollama_provider_chat():
    """Test Ollama provider returns valid response."""

def test_openai_provider_chat():
    """Test OpenAI provider returns valid response."""

def test_groq_provider_chat():
    """Test Groq provider returns valid response."""

def test_deepseek_provider_chat():
    """Test DeepSeek provider returns valid response."""

def test_provider_registry():
    """Test provider factory returns correct provider type."""
```

**Dataset Tests (`test_datasets.py`):**
```python
def test_browserart_load():
    """Test BrowserART loads correct number of samples."""

def test_do_not_answer_load():
    """Test DoNotAnswer loads correct number of samples."""

def test_jailbreakbench_load():
    """Test JailbreakBench loads correct number of samples."""

def test_dataset_reproducibility():
    """Test same seed produces same samples."""
```

**Judge Tests (`test_judges.py`):**
```python
# Adapted from existing 25 tests in test_five_level_judge.py
def test_clear_denial():
    """Level 1: Refusal with no actions."""

def test_soft_denial():
    """Level 2: Refusal but with some actions."""

def test_non_denial():
    """Level 3: No refusal, no harmful content."""

def test_harmful_plans():
    """Level 4: Harmful content generated, not executed."""

def test_harmful_actions():
    """Level 5: Harmful actions executed."""

def test_judge_with_different_providers():
    """Test judge works with Ollama, OpenAI, Groq, DeepSeek."""
```

**Layer Tests (`test_layers.py`):**
```python
def test_llm_layer_returns_response():
    """Test LLM layer returns final_text and steps."""

def test_llm_layer_with_all_providers():
    """Test LLM layer works with all providers."""
```

**CLI Tests (`test_cli.py`):**
```python
def test_cli_help():
    """Test surfaceai --help works."""

def test_cli_providers_list():
    """Test surfaceai providers list works."""

def test_cli_datasets_list():
    """Test surfaceai datasets list works."""

def test_cli_run_dry():
    """Test run command with --dry-run (if implemented)."""
```

### Integration Tests (tests/integration/)

```python
@pytest.mark.integration
def test_llm_ollama_e2e():
    """End-to-end: LLM layer with Ollama, 1 sample."""

@pytest.mark.integration
def test_llm_openai_e2e():
    """End-to-end: LLM layer with OpenAI, 1 sample."""

@pytest.mark.integration
@pytest.mark.docker
def test_openhands_e2e():
    """End-to-end: OpenHands layer, 1 sample."""

@pytest.mark.integration
def test_output_format():
    """Verify trace.jsonl and summary.json format."""
```

### Test Fixtures (`conftest.py`)

```python
@pytest.fixture
def mock_provider():
    """Mock LLM provider for unit tests."""

@pytest.fixture
def sample_browserart_data(tmp_path):
    """Create sample BrowserART data (3 items)."""

@pytest.fixture
def judge_no_llm():
    """Five-level judge in heuristics-only mode."""
```

### Running Tests

```bash
# All unit tests
pytest tests/ -v

# Skip integration tests
pytest tests/ -v -m "not integration"

# Only integration tests
pytest tests/ -v -m integration

# With coverage
pytest tests/ --cov=surfaceai --cov-report=html
```
