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

## License

MIT
