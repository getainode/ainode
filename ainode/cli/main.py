"""AINode CLI — main entry point."""

import argparse
import sys
import uuid
import time
import threading

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from ainode import __version__
from ainode.core.config import NodeConfig, ensure_dirs
from ainode.core.gpu import gpu_summary, detect_gpu

console = Console()


def _tail_log(path, lines=10):
    """Print the last N lines of a log file."""
    try:
        with open(path) as f:
            all_lines = f.readlines()
        for line in all_lines[-lines:]:
            console.print(f"    {line.rstrip()}")
    except Exception:
        pass


BANNER = """
    ╔══════════════════════════════════════╗
    ║            A I N o d e               ║
    ║   Your local AI platform for NVIDIA  ║
    ╚══════════════════════════════════════╝
"""


def cmd_start(args):
    """Start AINode."""
    print(BANNER)
    ensure_dirs()

    config = NodeConfig.load()

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
        mem_gb = gpu.memory_total_mb / 1024
        um = " (unified memory)" if gpu.unified_memory else ""
        print(f"  GPU:   {gpu.name} | {mem_gb:.0f} GB{um}")
        print(f"  CUDA:  {gpu.cuda_version} | Driver {gpu.driver_version}")
    else:
        print("  GPU:   No NVIDIA GPU detected (CPU mode)")
    print()

    # Start vLLM engine
    console.print(f"  Model: {config.model}")
    console.print(f"  API:   http://localhost:{config.api_port}/v1")
    console.print(f"  Web:   http://localhost:{config.web_port}")
    console.print()

    from ainode.engine.vllm_engine import VLLMEngine

    engine = VLLMEngine(config)
    if not engine.start():
        console.print("  [red]Failed to start engine.[/red] Check logs in ~/.ainode/logs/")
        sys.exit(1)

    # Wait for readiness with spinner and log tailing
    with Live(Spinner("dots", text="Starting inference engine..."), console=console, transient=True):
        ready = engine.wait_ready(timeout=300)

    if not ready:
        console.print("  [red]Engine failed to become ready within 5 minutes.[/red]")
        if engine.log_path and engine.log_path.exists():
            console.print(f"  Last log lines:")
            _tail_log(engine.log_path, lines=10)
        engine.stop()
        sys.exit(1)

    console.print("  [green]Engine ready.[/green] Open your browser to get started.")
    console.print()
    console.print("  Powered by argentos.ai")
    console.print()

    # Keep running until interrupted
    try:
        engine.process.wait()
    except KeyboardInterrupt:
        console.print("\n  Shutting down...")
        engine.stop()


def cmd_stop(args):
    """Stop AINode."""
    print("Stopping AINode...")
    # TODO: Signal running instance to stop
    print("Done.")


def cmd_status(args):
    """Show cluster status."""
    print(BANNER)
    console.print(f"  GPU: {gpu_summary()}")

    config = NodeConfig.load()
    console.print(f"  Node ID: {config.node_id or 'not configured'}")
    console.print(f"  Model: {config.model}")
    console.print(f"  API: http://localhost:{config.api_port}/v1")
    console.print(f"  Email: {config.email or 'not set'}")
    console.print()

    # Engine health check
    from ainode.engine.vllm_engine import VLLMEngine
    engine = VLLMEngine(config)
    health = engine.health_check()

    if health["api_responding"]:
        console.print("  Engine: [green]running[/green]")
        if health["models_loaded"]:
            console.print(f"  Models: {', '.join(health['models_loaded'])}")
    elif health["process_alive"]:
        console.print("  Engine: [yellow]starting[/yellow] (process alive, API not ready)")
    else:
        console.print("  Engine: [red]stopped[/red]")
    console.print()


def cmd_models(args):
    """List available models."""
    print("Popular models for AINode:\n")
    models = [
        ("llama-3.2-3b", "Llama 3.2 3B Instruct", "~6 GB", "Quick start"),
        ("llama-3.1-8b", "Llama 3.1 8B Instruct", "~16 GB", "Recommended"),
        ("llama-3.1-70b-4bit", "Llama 3.1 70B (AWQ 4-bit)", "~35 GB", "High quality"),
        ("qwen-2.5-72b", "Qwen 2.5 72B Instruct", "~40 GB", "Coding + multilingual"),
        ("deepseek-r1-7b", "DeepSeek R1 Distill 7B", "~14 GB", "Reasoning"),
    ]
    for short, name, mem, note in models:
        print(f"  {short:<25} {mem:<10} {note}")
    print()
    print("  Set model: ainode config --model <name>")
    print("  Full list: https://ainode.dev/models")
    print()


def cmd_auth(args):
    """Manage API key authentication."""
    from ainode.auth.middleware import AuthConfig

    action = getattr(args, "auth_action", None)
    auth_cfg = AuthConfig.load()

    if action == "enable":
        entry = auth_cfg.enable()
        console.print("  [green]Auth enabled.[/green]")
        console.print(f"  API key: {entry['key']}")
        console.print(f"  Key ID:  {entry['id']}")
        console.print()
        console.print("  Use: Authorization: Bearer <key>")
        console.print("  Powered by argentos.ai")

    elif action == "disable":
        auth_cfg.disable()
        console.print("  [yellow]Auth disabled.[/yellow] All requests allowed.")
        console.print("  Powered by argentos.ai")

    elif action == "status":
        state = "[green]enabled[/green]" if auth_cfg.enabled else "[dim]disabled[/dim]"
        console.print(f"  Auth: {state}")
        console.print(f"  Keys: {len(auth_cfg.api_keys)}")
        console.print("  Powered by argentos.ai")

    elif action == "new-key":
        entry = auth_cfg.generate_key()
        console.print("  [green]New API key generated.[/green]")
        console.print(f"  API key: {entry['key']}")
        console.print(f"  Key ID:  {entry['id']}")
        console.print("  Powered by argentos.ai")

    else:
        console.print("  Usage: ainode auth {enable|disable|status|new-key}")


def main():
    parser = argparse.ArgumentParser(
        prog="ainode",
        description="AINode — Turn any NVIDIA GPU into a local AI platform.",
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


    # auth
    auth_parser = subparsers.add_parser("auth", help="Manage API key authentication")
    auth_sub = auth_parser.add_subparsers(dest="auth_action")
    auth_sub.add_parser("enable", help="Enable API key auth")
    auth_sub.add_parser("disable", help="Disable API key auth")
    auth_sub.add_parser("status", help="Show auth status")
    auth_sub.add_parser("new-key", help="Generate a new API key")
    auth_parser.set_defaults(func=cmd_auth)

    args = parser.parse_args()

    if args.command is None:
        # No subcommand — default to start
        cmd_start(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
