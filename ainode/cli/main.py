"""AINode CLI — main entry point."""

import argparse
import sys
import uuid

from ainode import __version__
from ainode.core.config import NodeConfig, ensure_dirs
from ainode.core.gpu import gpu_summary, detect_gpu


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
    print(f"  Model: {config.model}")
    print(f"  API:   http://localhost:{config.api_port}/v1")
    print(f"  Web:   http://localhost:{config.web_port}")
    print()
    print("  Starting inference engine...")

    from ainode.engine.vllm_engine import VLLMEngine

    engine = VLLMEngine(config)
    if engine.start():
        print("  Engine started. Open your browser to get started.")
        print()
        print(f"  Powered by argentos.ai")
        print()

        # Keep running until interrupted
        try:
            engine.process.wait()
        except KeyboardInterrupt:
            print("\n  Shutting down...")
            engine.stop()
    else:
        print("  Failed to start engine. Check logs in ~/.ainode/logs/")
        sys.exit(1)


def cmd_stop(args):
    """Stop AINode."""
    print("Stopping AINode...")
    # TODO: Signal running instance to stop
    print("Done.")


def cmd_status(args):
    """Show cluster status."""
    print(BANNER)
    print(f"  GPU: {gpu_summary()}")

    config = NodeConfig.load()
    print(f"  Node ID: {config.node_id or 'not configured'}")
    print(f"  Model: {config.model}")
    print(f"  API: http://localhost:{config.api_port}/v1")
    print(f"  Email: {config.email or 'not set'}")
    print()


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

    args = parser.parse_args()

    if args.command is None:
        # No subcommand — default to start
        cmd_start(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
