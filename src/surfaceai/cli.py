"""Command-line interface for SurfaceAI."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from surfaceai.config.schemas import (
    ExperimentConfig,
    JudgeConfig,
    JudgeType,
    Layer,
    MASConfig,
    MASExperiment,
    OpenHandsSettings,
    Provider,
)

app = typer.Typer(
    name="surfaceai",
    help="SurfaceAI: Research framework for evaluating AI safety",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Providers subcommand
# ---------------------------------------------------------------------------

providers_app = typer.Typer(help="Manage LLM providers", invoke_without_command=True)
app.add_typer(providers_app, name="providers")


@providers_app.callback()
def providers_callback(ctx: typer.Context):
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

    for name, cfg in PROVIDER_DEFAULTS.items():
        table.add_row(name, cfg["default_model"], cfg.get("base_url", ""))

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


# ---------------------------------------------------------------------------
# Datasets subcommand
# ---------------------------------------------------------------------------

datasets_app = typer.Typer(help="Manage datasets", invoke_without_command=True)
app.add_typer(datasets_app, name="datasets")


@datasets_app.callback()
def datasets_callback(ctx: typer.Context):
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


# ---------------------------------------------------------------------------
# Run command
# ---------------------------------------------------------------------------

@app.command()
def run(
    layer: Layer = typer.Option(Layer.llm, "--layer", "-l", help="Execution layer"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset to use"),
    provider: Provider = typer.Option(Provider.ollama, "--provider", "-p", help="LLM provider"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Custom API base URL"),
    judge_provider: Provider = typer.Option(
        Provider.openai, "--judge-provider", help="Judge LLM provider",
    ),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge-model", help="Judge model ID"),
    judge_type: JudgeType = typer.Option(
        JudgeType.binary, "--judge-type", help="Judge type (binary or five_level)",
    ),
    n: int = typer.Option(100, "--n", "-n", help="Number of samples"),
    repeats: int = typer.Option(5, "--repeats", "-r", help="Repetitions per sample"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    out_dir: str = typer.Option("runs", "--out-dir", "-o", help="Output directory"),
    openhands_port: int = typer.Option(3000, "--openhands-port", help="OpenHands API port"),
    website_port: int = typer.Option(8080, "--website-port", help="Website server port"),
    skip_docker: bool = typer.Option(False, "--skip-docker", help="Skip Docker, assume OpenHands server is running externally"),
    website_host: str = typer.Option("host.docker.internal", "--website-host", help="Hostname for website server (use 'localhost' for local mode)"),
    experiment: Optional[MASExperiment] = typer.Option(
        None, "--experiment", "-e", help="MAS experiment (required when layer=mas)",
    ),
    max_steps: int = typer.Option(10, "--max-steps", help="Max executor steps per episode (MAS only)"),
):
    """Run a safety evaluation experiment."""
    from surfaceai.runner import run_experiment

    if layer == Layer.mas and experiment is None:
        console.print("[red]Error:[/red] --experiment is required when layer=mas")
        raise typer.Exit(1)

    mas_config = MASConfig(
        experiment=experiment or MASExperiment.e1,
        max_steps=max_steps,
    )

    config = ExperimentConfig(
        layer=layer,
        dataset=dataset,
        provider=provider,
        model=model,
        base_url=base_url,
        judge=JudgeConfig(
            provider=judge_provider,
            model=judge_model,
            judge_type=judge_type,
        ),
        openhands=OpenHandsSettings(
            api_port=openhands_port,
            website_port=website_port,
            website_host=website_host,
            skip_docker=skip_docker,
            auto_start_container=not skip_docker,
            **({"ollama_base_url": "http://localhost:11434/v1"} if skip_docker else {}),
        ),
        mas=mas_config,
        n=n,
        repeats=repeats,
        seed=seed,
        out_dir=out_dir,
    )

    console.print("[bold]SurfaceAI Experiment[/bold]")
    console.print(f"  Layer: {config.layer.value}")
    console.print(f"  Dataset: {config.dataset}")
    console.print(f"  Target: {config.provider.value}/{config.effective_model}")
    console.print(f"  Judge: {config.judge.provider.value}/{config.judge.model} ({config.judge.judge_type.value})")
    console.print(f"  Samples: {config.n} x {config.repeats} repeats = {config.n * config.repeats} evaluations")
    if config.layer == Layer.openhands:
        console.print(f"  OpenHands port: {config.openhands.api_port}")
        console.print(f"  Website port: {config.openhands.website_port}")
    if config.layer == Layer.mas:
        console.print(f"  Experiment: {config.mas.experiment.value}")
        console.print(f"  Max steps: {config.mas.max_steps}")
    console.print()

    try:
        summary = run_experiment(config)
        console.print()
        console.print("[bold green]Experiment completed![/bold green]")
        console.print(f"  ASR: {summary.asr:.2%} ({summary.harmful}/{summary.total} harmful)")
        if summary.mean_execution_reach is not None:
            console.print(f"  Mean ER: {summary.mean_execution_reach:.2f}")
        if summary.mean_propagation_depth is not None:
            console.print(f"  Mean PD: {summary.mean_propagation_depth:.2f}")
        console.print(f"  Results: {out_dir}/{summary.run_id}/")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Serve command
# ---------------------------------------------------------------------------

@app.command()
def serve(
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve on"),
):
    """Start the mock website server for BrowserART testing."""
    from surfaceai.web.server import run_server

    console.print(f"[bold]Starting website server on port {port}...[/bold]")
    run_server(port=port)


if __name__ == "__main__":
    app()
