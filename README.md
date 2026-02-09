# SurfaceAI

Research framework for evaluating AI safety across execution layers.

## Installation

```bash
# Clone the repository
git clone git@github.com:liadperetz/SurfaceAI.git
cd SurfaceAI

# Create virtual environment and install (using uv)
uv venv --python 3.11
uv pip install -e .

# With development dependencies
uv pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

## Usage

### LLM Baseline Experiments

```bash
# Run LLM baseline experiment
surfaceai run --layer llm --dataset do_not_answer --provider ollama --model llama3.1:8b

# With OpenAI
surfaceai run --layer llm --dataset browserart --provider openai --model gpt-4o-mini

# List available providers
surfaceai providers list

# List available datasets
surfaceai datasets list
```

### Web Agent (OpenHands) Experiments

```bash
# Start the mock website server
surfaceai serve --port 8080 &

# Start OpenHands container (requires Docker)
cd docker && docker compose --profile openai up -d && cd ..

# Run web agent experiment with 5-level judge
surfaceai run \
  --layer openhands \
  --dataset browserart \
  --provider openai \
  --model gpt-4o-mini \
  --openhands-port 3001 \
  --judge-type five_level \
  -n 10 -r 1
```

See [docs/OPENHANDS_EXPERIMENTS.md](docs/OPENHANDS_EXPERIMENTS.md) for detailed setup instructions.

## License

MIT
