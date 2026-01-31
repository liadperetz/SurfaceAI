"""Command-line interface for SurfaceAI."""

from __future__ import annotations

from typing import Optional
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="surfaceai",
    help="SurfaceAI: Research framework for evaluating AI safety",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


class Provider(str, Enum):
    ollama = "ollama"
    openai = "openai"
    groq = "groq"
    deepseek = "deepseek"


class Dataset(str, Enum):
    browserart = "browserart"
    do_not_answer = "do_not_answer"
    jailbreakbench = "jailbreakbench"


# --- Providers subcommand ---
providers_app = typer.Typer(help="Manage LLM providers", invoke_without_command=True)
app.add_typer(providers_app, name="providers")


@providers_app.callback()
def providers_callback(ctx: typer.Context):
    """List or manage LLM providers."""
    if ctx.invoked_subcommand is None:
        providers_list()


@providers_app.command("list")
def providers_list():
    """List available LLM providers."""
    from surfaceai.providers import PROVIDER_DEFAULTS

    table = Table(title="Available Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Default Model", style="green")
    table.add_column("Base URL", style="dim")

    for name, config in PROVIDER_DEFAULTS.items():
        table.add_row(name, config["default_model"], config.get("base_url", ""))

    console.print(table)


@providers_app.command("test")
def providers_test(
    provider: Provider = typer.Argument(..., help="Provider to test"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
):
    """Test connection to a provider."""
    from surfaceai.providers import create_client

    console.print(f"Testing {provider.value}...", style="dim")

    try:
        client = create_client(provider.value, model=model)
        response = client.chat([{"role": "user", "content": "Say 'hello' and nothing else."}])
        console.print(f"[green]Success![/green] Response: {response}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# --- Datasets subcommand ---
datasets_app = typer.Typer(help="Manage datasets", invoke_without_command=True)
app.add_typer(datasets_app, name="datasets")


@datasets_app.callback()
def datasets_callback(ctx: typer.Context):
    """List or manage datasets."""
    if ctx.invoked_subcommand is None:
        datasets_list()


@datasets_app.command("list")
def datasets_list():
    """List available datasets."""
    table = Table(title="Available Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Description", style="dim")

    datasets = [
        ("browserart", "BrowserART-100: Web agent safety behaviors"),
        ("do_not_answer", "DoNotAnswer: Harmful questions dataset"),
        ("jailbreakbench", "JailbreakBench: Jailbreak attempts dataset"),
    ]

    for name, desc in datasets:
        table.add_row(name, desc)

    console.print(table)


class Layer(str, Enum):
    llm = "llm"


# --- Run command ---
@app.command()
def run(
    layer: Layer = typer.Option(Layer.llm, "--layer", "-l", help="Execution layer"),
    dataset: Dataset = typer.Option(..., "--dataset", "-d", help="Dataset to use"),
    provider: Provider = typer.Option(Provider.ollama, "--provider", "-p", help="LLM provider"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Custom API base URL (for Ollama on different ports)"),
    judge_provider: Provider = typer.Option(
        Provider.openai, "--judge-provider", help="Judge LLM provider"
    ),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge-model", help="Judge model ID"),
    n: int = typer.Option(100, "--n", "-n", help="Number of samples"),
    repeats: int = typer.Option(5, "--repeats", "-r", help="Repetitions per sample"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    out_dir: str = typer.Option("runs", "--out-dir", "-o", help="Output directory"),
):
    """Run a safety evaluation experiment."""
    from surfaceai.runner import run_experiment

    console.print("[bold]SurfaceAI Experiment[/bold]")
    console.print(f"  Layer: {layer.value}")
    console.print(f"  Dataset: {dataset.value}")
    console.print(f"  Target: {provider.value}/{model or 'default'}")
    console.print(f"  Judge: {judge_provider.value}/{judge_model}")
    console.print(f"  Samples: {n} x {repeats} repeats = {n * repeats} evaluations")
    console.print()

    try:
        summary = run_experiment(
            layer=layer.value,
            dataset=dataset.value,
            provider=provider.value,
            model=model,
            base_url=base_url,
            judge_provider=judge_provider.value,
            judge_model=judge_model,
            n=n,
            repeats=repeats,
            seed=seed,
            out_dir=out_dir,
        )

        console.print()
        console.print("[bold green]Experiment completed![/bold green]")
        console.print(f"  ASR: {summary['asr']:.2%} ({summary['harmful']}/{summary['total']} harmful)")
        console.print(f"  Results: {out_dir}/{summary['run_id']}/")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
