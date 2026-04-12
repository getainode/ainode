"""AINode CLI — main entry point with Rich terminal output."""

import argparse
import os
import signal
import sys
import time
import uuid

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ainode import __version__
from ainode.core.config import NodeConfig, ensure_dirs, AINODE_HOME, LOGS_DIR

console = Console()

PID_FILE = AINODE_HOME / "ainode.pid"
VLLM_LOG = LOGS_DIR / "vllm.log"


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _banner():
    """Return a Rich Panel banner for AINode."""
    title_text = Text()
    title_text.append("A", style="bold cyan")
    title_text.append("I", style="bold cyan")
    title_text.append("N", style="bold cyan")
    title_text.append("ode", style="bold white")
    title_text.append("  v" + __version__, style="dim")

    body = Text.from_markup(
        "[bold white]Turn any NVIDIA GPU into a local AI platform[/bold white]\n"
        "[dim]Inference + fine-tuning in your browser. One command to start.[/dim]"
    )

    return Panel(
        body,
        title=title_text,
        subtitle="[dim italic]Powered by argentos.ai[/dim italic]",
        border_style="cyan",
        padding=(1, 4),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pid():
    """Write current PID to the PID file."""
    AINODE_HOME.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _read_pid():
    """Read PID from file, return int or None."""
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def _remove_pid():
    """Remove the PID file."""
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _pid_alive(pid):
    """Check if a PID is alive."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _tail_log(path, lines=10):
    """Return the last N lines of a log file."""
    try:
        with open(path) as f:
            all_lines = f.readlines()
        return all_lines[-lines:]
    except Exception:
        return []


def _gpu_info_table(gpu):
    """Build a Rich table for GPU info."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="bold cyan", no_wrap=True)
    table.add_column("value")

    mem_gb = gpu.memory_total_mb / 1024
    um = " (unified memory)" if gpu.unified_memory else ""
    table.add_row("GPU", f"{gpu.name} | {mem_gb:.0f} GB{um}")
    table.add_row("CUDA", f"{gpu.cuda_version} | Driver {gpu.driver_version}")
    table.add_row("Compute", f"SM {gpu.compute_capability}")
    return table


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_start(args):
    """Start AINode."""
    from ainode.core.gpu import gpu_summary, detect_gpu
    console.print(_banner())
    ensure_dirs()

    config = NodeConfig.load()

    # Override from CLI flags
    if hasattr(args, "model") and args.model:
        config.model = args.model
        config.save()
    if hasattr(args, "port") and args.port:
        config.api_port = args.port
        config.save()

    # First run — onboarding
    if not config.onboarded:
        from ainode.onboarding.setup import run_onboarding
        config = run_onboarding(config)

    # Assign node ID if needed
    if not config.node_id:
        config.node_id = str(uuid.uuid4())[:8]
        config.save()

    # Detect GPU
    gpu = detect_gpu()
    if gpu:
        console.print(_gpu_info_table(gpu))
    else:
        console.print("  [yellow]No NVIDIA GPU detected[/yellow] — running in CPU mode")
    console.print()

    # Service info
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("key", style="bold green", no_wrap=True)
    info_table.add_column("value")
    info_table.add_row("Model", config.model)
    info_table.add_row("API", f"http://localhost:{config.api_port}/v1")
    info_table.add_row("Web", f"http://localhost:{config.web_port}")
    info_table.add_row("Node", config.node_id or "pending")
    console.print(info_table)
    console.print()

    # Write PID file
    _write_pid()

    # Start vLLM engine
    from ainode.engine.vllm_engine import VLLMEngine

    engine = VLLMEngine(config)
    if not engine.start():
        console.print("  [red]Failed to start engine.[/red] Check logs in ~/.ainode/logs/")
        _remove_pid()
        sys.exit(1)

    # Wait for readiness with spinner
    with Live(Spinner("dots", text="Starting inference engine..."), console=console, transient=True):
        ready = engine.wait_ready(timeout=300)

    if not ready:
        console.print("  [red]Engine failed to become ready within 5 minutes.[/red]")
        if engine.log_path and engine.log_path.exists():
            console.print("  [dim]Last log lines:[/dim]")
            for line in _tail_log(engine.log_path, lines=10):
                console.print(f"    [dim]{line.rstrip()}[/dim]")
        engine.stop()
        _remove_pid()
        sys.exit(1)

    console.print("  [bold green]Engine ready.[/bold green] Open your browser to get started.\n")

    # Keep running until interrupted
    try:
        engine.process.wait()
    except KeyboardInterrupt:
        console.print("\n  [yellow]Shutting down...[/yellow]")
        engine.stop()
    finally:
        _remove_pid()


def cmd_stop(args):
    """Stop a running AINode instance."""
    console.print(_banner())

    pid = _read_pid()

    if pid and _pid_alive(pid):
        console.print(f"  Stopping AINode (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait up to 10 seconds for graceful shutdown
            for _ in range(20):
                if not _pid_alive(pid):
                    break
                time.sleep(0.5)
            else:
                # Force kill if still alive
                console.print("  [yellow]Process did not exit gracefully, sending SIGKILL...[/yellow]")
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _remove_pid()
        console.print("  [green]AINode stopped.[/green]\n")
        return

    # No PID file or stale PID — try to find the process
    _remove_pid()

    # Fallback: check for python processes running ainode
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ainode.cli.main|ainode start"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip() and int(p) != os.getpid()]
    except Exception:
        pids = []

    if pids:
        for p in pids:
            console.print(f"  Stopping AINode process (PID {p})...")
            try:
                os.kill(p, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        console.print("  [green]AINode stopped.[/green]\n")
    else:
        console.print("  [dim]No running AINode instance found.[/dim]\n")


def cmd_status(args):
    """Show cluster status with Rich formatting."""
    from ainode.core.gpu import gpu_summary, detect_gpu
    console.print(_banner())

    config = NodeConfig.load()
    gpu = detect_gpu()

    # Node info table
    table = Table(title="Node Info", border_style="cyan", show_lines=False)
    table.add_column("Property", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Node ID", config.node_id or "[dim]not configured[/dim]")
    table.add_row("Model", config.model)
    table.add_row("API", f"http://localhost:{config.api_port}/v1")
    table.add_row("Web", f"http://localhost:{config.web_port}")
    table.add_row("Email", config.email or "[dim]not set[/dim]")

    if gpu:
        mem_gb = gpu.memory_total_mb / 1024
        um = " (unified)" if gpu.unified_memory else ""
        table.add_row("GPU", f"{gpu.name} | {mem_gb:.0f} GB{um}")
        table.add_row("CUDA", f"{gpu.cuda_version} | Driver {gpu.driver_version}")
    else:
        table.add_row("GPU", "[yellow]No NVIDIA GPU detected[/yellow]")

    console.print(table)
    console.print()

    # Engine health check
    from ainode.engine.vllm_engine import VLLMEngine
    engine = VLLMEngine(config)
    health = engine.health_check()

    if health["api_responding"]:
        console.print("  Engine:  [bold green]running[/bold green]")
        if health["models_loaded"]:
            console.print(f"  Models:  {', '.join(health['models_loaded'])}")
    elif health["process_alive"]:
        console.print("  Engine:  [bold yellow]starting[/bold yellow] (process alive, API not ready)")
    else:
        console.print("  Engine:  [bold red]stopped[/bold red]")

    # PID file status
    pid = _read_pid()
    if pid and _pid_alive(pid):
        console.print(f"  PID:     {pid}")
    console.print()


def cmd_models(args):
    """List available models with Rich table and GPU-aware recommendations."""
    from ainode.core.gpu import detect_gpu
    console.print(_banner())

    gpu = detect_gpu()
    gpu_mem_gb = (gpu.memory_total_mb / 1024) if gpu else 0

    models = [
        ("llama-3.2-3b", "Llama 3.2 3B Instruct", "~6 GB", "Quick start"),
        ("llama-3.1-8b", "Llama 3.1 8B Instruct", "~16 GB", "Recommended"),
        ("llama-3.1-70b-4bit", "Llama 3.1 70B (AWQ 4-bit)", "~35 GB", "High quality"),
        ("qwen-2.5-72b", "Qwen 2.5 72B Instruct", "~40 GB", "Coding + multilingual"),
        ("deepseek-r1-7b", "DeepSeek R1 Distill 7B", "~14 GB", "Reasoning"),
    ]

    # Memory thresholds for recommendations
    mem_thresholds = {
        "~6 GB": 6,
        "~14 GB": 14,
        "~16 GB": 16,
        "~35 GB": 35,
        "~40 GB": 40,
    }

    table = Table(title="Available Models", border_style="cyan")
    table.add_column("Name", style="bold white", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Description")
    table.add_column("Tag")

    for short, name, mem, note in models:
        threshold = mem_thresholds.get(mem, 999)
        fits = gpu_mem_gb >= threshold

        if note == "Recommended":
            tag_style = "bold green" if fits else "dim green"
        elif fits:
            tag_style = "green"
        else:
            tag_style = "dim red"

        tag = Text(note, style=tag_style)
        if not fits and gpu_mem_gb > 0:
            tag.append(" (needs more VRAM)", style="dim")

        table.add_row(short, mem, name, tag)

    console.print(table)
    console.print()
    console.print("  Set model:  [bold]ainode config --model <name>[/bold]")
    console.print("  Full list:  [link=https://ainode.dev/models]https://ainode.dev/models[/link]")
    console.print()


def cmd_config(args):
    """Show or update AINode configuration."""
    config = NodeConfig.load()

    if args.model:
        config.model = args.model
        config.save()
        console.print(f"  [green]Model set to:[/green] {args.model}")
        return

    if args.port:
        config.api_port = args.port
        config.save()
        console.print(f"  [green]API port set to:[/green] {args.port}")
        return

    # Default: --show
    console.print(_banner())

    from dataclasses import asdict
    data = asdict(config)

    table = Table(title="Configuration", border_style="cyan")
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    for key, value in data.items():
        display = str(value) if value is not None else "[dim]not set[/dim]"
        table.add_row(key, display)

    console.print(table)
    console.print()
    console.print(f"  Config file: [dim]{AINODE_HOME / 'config.json'}[/dim]")
    console.print()


def cmd_logs(args):
    """Show or tail vLLM logs."""
    log_file = VLLM_LOG

    if not log_file.exists():
        console.print("  [dim]No log file found at[/dim] ~/.ainode/logs/vllm.log")
        console.print("  [dim]Start AINode first: [bold]ainode start[/bold][/dim]")
        return

    if args.follow:
        console.print(f"  [dim]Tailing {log_file} (Ctrl+C to stop)[/dim]\n")
        try:
            import subprocess
            proc = subprocess.Popen(
                ["tail", "-f", str(log_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in proc.stdout:
                console.print(line.rstrip())
        except KeyboardInterrupt:
            console.print("\n  [dim]Stopped tailing.[/dim]")
            if proc:
                proc.terminate()
    else:
        lines = _tail_log(log_file, lines=args.lines)
        if not lines:
            console.print("  [dim]Log file is empty.[/dim]")
            return
        console.print(f"  [dim]Last {len(lines)} lines of {log_file}:[/dim]\n")
        for line in lines:
            console.print(f"  {line.rstrip()}")
        console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="ainode",
        description="AINode -- Turn any NVIDIA GPU into a local AI platform.",
    )
    parser.add_argument("--version", action="version", version=f"ainode {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # start
    start_parser = subparsers.add_parser("start", help="Start AINode")
    start_parser.add_argument("--model", help="Model to serve")
    start_parser.add_argument("--port", type=int, help="API port")
    start_parser.set_defaults(func=cmd_start)

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop AINode")
    stop_parser.set_defaults(func=cmd_stop)

    # status
    status_parser = subparsers.add_parser("status", help="Show cluster status")
    status_parser.set_defaults(func=cmd_status)

    # models
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.set_defaults(func=cmd_models)

    # config
    config_parser = subparsers.add_parser("config", help="Show or update configuration")
    config_parser.add_argument("--show", action="store_true", default=True, help="Show current config")
    config_parser.add_argument("--model", help="Set the model")
    config_parser.add_argument("--port", type=int, help="Set the API port")
    config_parser.set_defaults(func=cmd_config)

    # logs
    logs_parser = subparsers.add_parser("logs", help="Show vLLM logs")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Tail logs in real-time")
    logs_parser.add_argument("--lines", "-n", type=int, default=50, help="Number of lines to show (default: 50)")
    logs_parser.set_defaults(func=cmd_logs)

    args = parser.parse_args()

    if args.command is None:
        # No subcommand — default to start
        cmd_start(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
