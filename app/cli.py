"""CLI for Eval Harness."""

import asyncio
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from uuid import UUID

from app.core.config import settings
from app.services.evaluation_service import evaluation_service
from app.models.schemas import (
    BenchmarkSuite,
    BenchmarkTask,
    BenchmarkType,
    EvaluationRequest,
    ComparisonRequest,
    ComparisonType,
    StatisticalTest,
)

app = typer.Typer(help="Eval Harness CLI")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8005, "--port", "-p", help="Port to bind"),
):
    """Start the API server."""
    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


@app.command()
def interactive():
    """Interactive evaluation session."""
    asyncio.run(_interactive_async())


async def _interactive_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] v{settings.app_version}"))
    console.print("[dim]Commands: benchmark, evaluate, compare, stats, leaderboard, help, quit[/dim]")
    console.print()

    await evaluation_service.initialize()

    while True:
        try:
            cmd = Prompt.ask("[bold green]eval>[/bold green]").strip().lower()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                _show_help()
            elif cmd == "benchmark":
                _list_benchmarks()
            elif cmd == "evaluate":
                await _evaluate_interactive()
            elif cmd == "compare":
                await _compare_interactive()
            elif cmd == "stats":
                await _stats_interactive()
            elif cmd == "leaderboard":
                _leaderboard_interactive()
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
        except (KeyboardInterrupt, EOFError):
            break


def _show_help():
    console.print("[bold]Available commands:[/bold]")
    console.print("  benchmark    - List available benchmarks")
    console.print("  evaluate     - Evaluate model outputs")
    console.print("  compare      - Compare two models")
    console.print("  stats        - Run statistical tests")
    console.print("  leaderboard  - Show benchmark leaderboard")
    console.print("  help         - Show this help")
    console.print("  quit         - Exit")


def _list_benchmarks():
    benchmarks = evaluation_service.list_benchmarks()
    if not benchmarks:
        console.print("[yellow]No benchmarks available[/yellow]")
        return

    table = Table(title="Available Benchmarks")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Type", style="green")
    table.add_column("Tasks", style="yellow")

    for b in benchmarks:
        table.add_row(str(b.id)[:8], b.name, b.benchmark_type.value, str(len(b.tasks)))

    console.print(table)


async def _evaluate_interactive():
    _list_benchmarks()
    benchmark_id = Prompt.ask("Benchmark ID")
    benchmark = evaluation_service.get_benchmark(UUID(benchmark_id))
    if not benchmark:
        console.print("[red]Benchmark not found[/red]")
        return

    model_name = Prompt.ask("Model name")
    console.print("[dim]Enter outputs (one per task, empty line to finish):[/dim]")

    outputs = []
    for i, task in enumerate(benchmark.tasks):
        output = Prompt.ask(f"Task {i+1} ({task.category})", default="")
        if not output:
            break
        outputs.append(output)

    if not outputs:
        console.print("[yellow]No outputs provided[/yellow]")
        return

    request = EvaluationRequest(
        benchmark_id=UUID(benchmark_id),
        model_outputs={model_name: outputs},
    )

    with console.status("[bold yellow]Evaluating...[/bold yellow]", spinner="dots"):
        results = await evaluation_service.evaluate(request)

    for result in results:
        console.print(f"[green]Model: {result.model_name}[/green]")
        console.print(f"  Mean score: [cyan]{result.scores.get('mean', 0):.3f}[/cyan]")
        console.print(f"  Std dev: [cyan]{result.scores.get('std', 0):.3f}[/cyan]")


async def _compare_interactive():
    model_a = Prompt.ask("Model A name")
    model_b = Prompt.ask("Model B name")

    console.print("[dim]Enter prompts (empty line to finish):[/dim]")
    prompts = []
    while True:
        prompt = Prompt.ask("Prompt", default="")
        if not prompt:
            break
        prompts.append(prompt)

    if not prompts:
        console.print("[yellow]No prompts provided[/yellow]")
        return

    request = ComparisonRequest(
        comparison_type=ComparisonType.PAIRWISE,
        model_a=model_a,
        model_b=model_b,
        prompts=prompts,
    )

    with console.status("[bold yellow]Comparing...[/bold yellow]", spinner="dots"):
        result = await evaluation_service.compare(request)

    console.print(f"[green]Comparison complete![/green]")
    console.print(f"  {model_a} wins: [cyan]{result.wins_a}[/cyan]")
    console.print(f"  {model_b} wins: [cyan]{result.wins_b}[/cyan]")
    console.print(f"  Ties: [cyan]{result.ties}[/cyan]")
    console.print(f"  {model_a} win rate: [cyan]{result.win_rate_a:.1%}[/cyan]")
    console.print(f"  {model_b} win rate: [cyan]{result.win_rate_b:.1%}[/cyan]")
    if result.p_value is not None:
        console.print(f"  p-value: [cyan]{result.p_value:.4f}[/cyan]")


async def _stats_interactive():
    console.print("[dim]Enter sample A (comma-separated):[/dim]")
    sample_a_str = Prompt.ask("Sample A")
    sample_a = [float(x.strip()) for x in sample_a_str.split(",") if x.strip()]

    console.print("[dim]Enter sample B (comma-separated):[/dim]")
    sample_b_str = Prompt.ask("Sample B")
    sample_b = [float(x.strip()) for x in sample_b_str.split(",") if x.strip()]

    if not sample_a or not sample_b:
        console.print("[yellow]Both samples required[/yellow]")
        return

    test_type = Prompt.ask("Test type", choices=["bootstrap", "t_test", "wilcoxon", "permutation"], default="bootstrap")

    test = StatisticalTest(test_type=test_type)

    with console.status("[bold yellow]Running test...[/bold yellow]", spinner="dots"):
        result = evaluation_service.run_statistical_test(sample_a, sample_b, test)

    console.print(f"[green]Test complete![/green]")
    console.print(f"  Test: [cyan]{result.test_type}[/cyan]")
    console.print(f"  p-value: [cyan]{result.p_value:.6f}[/cyan]")
    console.print(f"  Effect size: [cyan]{result.effect_size:.3f}[/cyan]")
    console.print(f"  CI: [cyan]({result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f})[/cyan]")
    console.print(f"  Significant: [cyan]{result.significant}[/cyan]")
    console.print(f"  Interpretation: [cyan]{result.interpretation}[/cyan]")


def _leaderboard_interactive():
    benchmarks = evaluation_service.list_benchmarks()
    if not benchmarks:
        console.print("[yellow]No benchmarks[/yellow]")
        return

    _list_benchmarks()
    benchmark_id = Prompt.ask("Benchmark ID")
    benchmark = evaluation_service.get_benchmark(UUID(benchmark_id))
    if not benchmark:
        console.print("[red]Benchmark not found[/red]")
        return

    leaderboard = evaluation_service.get_leaderboard(UUID(benchmark_id))
    if not leaderboard or not leaderboard.entries:
        console.print("[yellow]No entries yet[/yellow]")
        return

    table = Table(title=f"Leaderboard: {leaderboard.benchmark_name}")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Score", style="yellow")

    for entry in leaderboard.entries[:20]:
        table.add_row(str(entry.rank), entry.model_name, f"{entry.score:.3f}")

    console.print(table)


@app.command()
def health():
    """Check service health."""
    asyncio.run(_health_async())


async def _health_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - Health Check"))
    health = await evaluation_service.get_health()
    console.print(f"Status: [green]{health.status}[/green]")
    console.print(f"Benchmarks: [cyan]{health.benchmarks_available}[/cyan]")
    console.print(f"Judge ready: [cyan]{health.judge_ready}[/cyan]")


@app.command()
def list_benchmarks():
    """List benchmarks."""
    _list_benchmarks()


@app.command()
def leaderboard(
    benchmark_id: str = typer.Argument(..., help="Benchmark ID"),
):
    """Show leaderboard."""
    leaderboard = evaluation_service.get_leaderboard(UUID(benchmark_id))
    if not leaderboard:
        console.print("[red]Benchmark not found[/red]")
        return

    table = Table(title=f"Leaderboard: {leaderboard.benchmark_name}")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Score", style="yellow")

    for entry in leaderboard.entries:
        table.add_row(str(entry.rank), entry.model_name, f"{entry.score:.3f}")

    console.print(table)


if __name__ == "__main__":
    app()