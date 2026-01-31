# LLM Baseline Experiments

This document describes how to run LLM baseline experiments to measure Attack Success Rate (ASR) across different models and datasets.

## Overview

LLM baseline experiments evaluate how language models respond to harmful prompts **without** any agent scaffolding (no tools, no web browsing). This provides a baseline measurement of model safety before adding agent capabilities.

**Metrics:**
- **ASR (Attack Success Rate)**: Percentage of prompts where the model complied with the harmful request
- A lower ASR indicates better safety alignment

## Prerequisites

1. **Python 3.10+** and **uv** package manager
2. **API keys** for the providers you want to use:
   - OpenAI: `OPENAI_API_KEY` (required for judge by default)
   - Groq: `GROQ_API_KEY` (optional)
   - DeepSeek: `DEEPSEEK_API_KEY` (optional)
3. **Ollama** (optional, for local model evaluation)

## Installation

```bash
# Clone and setup
git clone git@github.com:liadperetz/SurfaceAI.git
cd SurfaceAI

# Create virtual environment and install
uv venv --python 3.11
uv pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

```bash
# Run a small test experiment
surfaceai run \
  --layer llm \
  --dataset do_not_answer \
  --provider openai \
  --model gpt-4o-mini \
  -n 10 -r 1

# Check results
cat runs/*/summary.json
```

## Available Providers

| Provider | Default Model | Requires |
|----------|---------------|----------|
| `ollama` | llama3.1:8b | Local Ollama server |
| `openai` | gpt-4o-mini | OPENAI_API_KEY |
| `groq` | llama-3.1-8b-instant | GROQ_API_KEY |
| `deepseek` | deepseek-chat | DEEPSEEK_API_KEY |

```bash
# List providers
surfaceai providers list

# Test a provider connection
surfaceai providers test openai
surfaceai providers test ollama --model llama3.1:8b
```

## Available Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| `browserart` | BrowserART-100: Web safety behaviors | 100 |
| `do_not_answer` | Harmful questions dataset | 100 |
| `jailbreakbench` | Jailbreak attempts dataset | 100 |

```bash
# List datasets
surfaceai datasets list
```

## Running Experiments

### Basic Usage

```bash
surfaceai run \
  --layer llm \
  --dataset <dataset> \
  --provider <provider> \
  --model <model> \
  -n <samples> \
  -r <repeats>
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--layer` | `-l` | llm | Execution layer |
| `--dataset` | `-d` | (required) | Dataset name |
| `--provider` | `-p` | ollama | LLM provider |
| `--model` | `-m` | (provider default) | Model ID |
| `--base-url` | | (provider default) | Custom API URL |
| `--judge-provider` | | openai | Judge LLM provider |
| `--judge-model` | | gpt-4o-mini | Judge model ID |
| `-n` | | 100 | Number of samples |
| `--repeats` | `-r` | 5 | Repetitions per sample |
| `--seed` | `-s` | 42 | Random seed |
| `--out-dir` | `-o` | runs | Output directory |

### Examples

```bash
# Evaluate GPT-4o-mini on DoNotAnswer
surfaceai run -d do_not_answer -p openai -m gpt-4o-mini -n 100 -r 5

# Evaluate Llama 3.1 via Ollama
surfaceai run -d browserart -p ollama -m llama3.1:8b -n 100 -r 5

# Evaluate via Groq (fast inference)
surfaceai run -d jailbreakbench -p groq -m llama-3.1-70b-versatile -n 100 -r 5

# Use Llama as judge instead of GPT-4o-mini
surfaceai run -d do_not_answer -p openai -m gpt-4o-mini \
  --judge-provider ollama --judge-model llama3.1:8b
```

## Running Multiple Ollama Instances in Parallel

To evaluate multiple models simultaneously, run separate Ollama instances on different ports.

### Step 1: Start Multiple Ollama Instances

```bash
# Terminal 1 - Llama 3.1:8b on port 11501
OLLAMA_HOST=0.0.0.0:11501 ollama serve

# Terminal 2 - Gemma:7b on port 11502
OLLAMA_HOST=0.0.0.0:11502 ollama serve

# Terminal 3 - Qwen:7b on port 11503
OLLAMA_HOST=0.0.0.0:11503 ollama serve
```

Or use tmux for background execution:

```bash
# Start instances in background
tmux new-session -d -s ollama-11501 "OLLAMA_HOST=0.0.0.0:11501 ollama serve"
tmux new-session -d -s ollama-11502 "OLLAMA_HOST=0.0.0.0:11502 ollama serve"
tmux new-session -d -s ollama-11503 "OLLAMA_HOST=0.0.0.0:11503 ollama serve"

# Verify they're running
curl -s http://localhost:11501/api/tags | jq '.models[].name'
curl -s http://localhost:11502/api/tags | jq '.models[].name'
curl -s http://localhost:11503/api/tags | jq '.models[].name'
```

### Step 2: Pull Models to Each Instance

```bash
OLLAMA_HOST=localhost:11501 ollama pull llama3.1:8b
OLLAMA_HOST=localhost:11502 ollama pull gemma:7b
OLLAMA_HOST=localhost:11503 ollama pull qwen:7b
```

### Step 3: Run Experiments in Parallel

```bash
# Run all 3 experiments in parallel (background processes)
surfaceai run -d browserart -p ollama -m llama3.1:8b \
  --base-url http://localhost:11501/v1 \
  -n 100 -r 5 > logs/llama.log 2>&1 &

surfaceai run -d browserart -p ollama -m gemma:7b \
  --base-url http://localhost:11502/v1 \
  -n 100 -r 5 > logs/gemma.log 2>&1 &

surfaceai run -d browserart -p ollama -m qwen:7b \
  --base-url http://localhost:11503/v1 \
  -n 100 -r 5 > logs/qwen.log 2>&1 &

# Wait for all to complete
wait
echo "All experiments completed!"
```

### Step 4: View Results

```bash
# View summaries
for dir in runs/*__llm__ollama__*; do
  echo "=== $(basename $dir) ==="
  cat "$dir/summary.json" | jq '{model, asr, harmful, total}'
done
```

## Remote Ollama Server

You can also connect to Ollama running on a remote server:

```bash
# Connect to remote Ollama (e.g., on a GPU server)
surfaceai run -d browserart -p ollama -m llama3.1:70b \
  --base-url http://gpu-server.example.com:11434/v1 \
  -n 100 -r 5
```

## Output Structure

Each experiment run creates a directory:

```
runs/<timestamp>__llm__<provider>__<model>__<dataset>__judge_<judge>__seed<seed>__<uuid>/
├── config.json      # Experiment configuration
├── trace.jsonl      # Per-prompt results
└── summary.json     # Aggregated statistics
```

### config.json

```json
{
  "layer": "llm",
  "dataset": "browserart",
  "provider": "ollama",
  "model": "llama3.1:8b",
  "base_url": "http://localhost:11501/v1",
  "judge_provider": "openai",
  "judge_model": "gpt-4o-mini",
  "n": 100,
  "repeats": 5,
  "seed": 42,
  "run_id": "..."
}
```

### trace.jsonl

Each line is a single evaluation:

```json
{
  "id": "browserart_42",
  "rep": 0,
  "prompt": "Write a phishing email...",
  "response": "I cannot help with that...",
  "harmful": false,
  "judge_reason": "Contains refusal: 'i cannot'",
  "category": "content/phishing"
}
```

### summary.json

```json
{
  "run_id": "...",
  "layer": "llm",
  "dataset": "browserart",
  "provider": "ollama",
  "model": "llama3.1:8b",
  "judge_provider": "openai",
  "judge_model": "gpt-4o-mini",
  "n": 100,
  "repeats": 5,
  "seed": 42,
  "total": 500,
  "harmful": 45,
  "asr": 0.09
}
```

## Judge Configuration

The judge classifies whether the model's response complied with the harmful request.

**Default:** OpenAI GPT-4o-mini (recommended for accuracy and cost)

```bash
# Use different judge
surfaceai run -d browserart -p ollama -m llama3.1:8b \
  --judge-provider groq --judge-model llama-3.1-70b-versatile

# Use local Ollama as judge (free but less accurate)
surfaceai run -d browserart -p openai -m gpt-4o-mini \
  --judge-provider ollama --judge-model llama3.1:8b
```

**Cost estimate:** GPT-4o-mini judge costs ~$0.05 per 100 evaluations.

## Reproducibility

Results are reproducible when using:
- Same `--seed` value
- Same dataset version
- Deterministic model (`temperature=0`, which is the default)

```bash
# Two runs with same seed should produce identical samples
surfaceai run -d browserart -p openai -m gpt-4o-mini -n 10 -r 1 --seed 42
surfaceai run -d browserart -p openai -m gpt-4o-mini -n 10 -r 1 --seed 42
```

## Troubleshooting

### "API key not found"

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set in current shell
export OPENAI_API_KEY="sk-..."

# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Ollama connection refused

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Rate limiting

If you hit rate limits with cloud providers, reduce parallelism or add delays:

```bash
# Run sequentially instead of parallel
for model in llama3.1:8b gemma:7b qwen:7b; do
  surfaceai run -d browserart -p ollama -m $model -n 100 -r 5
done
```
