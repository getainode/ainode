"""AINode CLI — main entry point with Rich terminal output."""

import argparse
import importlib.util
import os
import shutil
import signal
import sys
import time
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ainode import __version__
from ainode.core.config import (
    AINODE_HOME,
    DEFAULT_ENGINE_BACKEND,
    LOGS_DIR,
    NodeConfig,
    ensure_dirs,
)

console = Console()

PID_FILE = AINODE_HOME / "ainode.pid"
VLLM_LOG = LOGS_DIR / "vllm.log"



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
        subtitle="[dim italic]Made in Texas[/dim italic]",
        border_style="cyan",
        padding=(1, 4),
    )



def _write_pid():
    AINODE_HOME.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _read_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def _remove_pid():
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _pid_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _tail_log(path, lines=10):
    """Return the last N lines of a log file without reading the entire file."""
    from collections import deque
    try:
        with open(path) as f:
            return list(deque(f, maxlen=lines))
    except Exception:
        return []


def _gpu_info_table(gpu):
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="bold cyan", no_wrap=True)
    table.add_column("value")

    mem_gb = gpu.memory_total_mb / 1024
    um = " (unified memory)" if gpu.unified_memory else ""
    table.add_row("GPU", f"{gpu.name} | {mem_gb:.0f} GB{um}")
    table.add_row("CUDA", f"{gpu.cuda_version} | Driver {gpu.driver_version}")
    table.add_row("Compute", f"SM {gpu.compute_capability}")
    return table


def _fabric_summary(config) -> str:
    """Cluster interface + its IPv4 for the start banner, never raising."""
    try:
        from ainode.cluster.netdev import describe_cluster_interface
        return describe_cluster_interface(config)
    except Exception:
        return "unknown"


def cmd_start(args):
    """Start AINode."""
    from ainode.core.gpu import detect_gpu
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

    # First run: the terminal wizard (ainode/onboarding/setup.py).
    # Skip when there is no TTY (e.g. running as a systemd service): there is no
    # browser wizard to fall back to any more (#208), and there never usefully
    # was. A non-interactive start marks the node onboarded and the dashboard is
    # where the model, the cluster and the rest are configured.
    if not config.onboarded:
        if sys.stdin.isatty():
            from ainode.onboarding.setup import run_onboarding
            config = run_onboarding(config)
        else:
            # Non-interactive (systemd, CI, docker without -it).
            # Do NOT auto-assign a model — leave it null so the server
            # starts immediately with no engine. User picks a model from
            # the web UI. Workers never need a model at all.
            if not config.node_id:
                config.node_id = str(uuid.uuid4())[:8]
            config.onboarded = True
            config.save()
            console.print("  [dim]Non-interactive start — open http://localhost:3000 to configure.[/dim]\n")

    # Assign node ID if needed
    if not config.node_id:
        config.node_id = str(uuid.uuid4())[:8]
        config.save()

    # Start-clean (closet #310): one-shot operator knob to skip replaying
    # persisted models this boot, so `restart` actually frees a node (replay
    # otherwise reloads config.model + the stacked manifest). Non-destructive —
    # on-disk config is untouched, so a normal restart resumes serving.
    from ainode.models.api_routes import consume_start_clean, sweep_engines_before_boot
    if consume_start_clean():
        if config.model:
            console.print("  [yellow]Start-clean — not replaying persisted model(s) this boot.[/yellow]\n")
        config.model = None
        config._skip_replay = True

    # Nothing may launch while an engine from this node's previous life is still
    # running. A vLLM engine container is a sibling spawned through docker.sock, so
    # an `ainode update` restart does NOT stop it: on the 0.5.11 roll the old
    # stacked engine was only reaped 14 s AFTER the new primary had started, so the
    # primary profiled against a node that still had the previous model resident
    # and both engines then under-sized their KV caches (#96). Sweep first, wait
    # for the containers to be gone, and only then launch anything. Also what makes
    # start-clean actually free the node: with config.model cleared there is no
    # boot engine to reclaim the old container.
    freed = sweep_engines_before_boot()
    if freed:
        console.print(
            f"  [dim]Freed {len(freed)} engine container(s) from the previous run.[/dim]\n")

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
    # Which NIC cross-node traffic will bind to. Printed because it is
    # autodetected when config.cluster_interface is empty or names a device
    # this host doesn't have. The user should be able to see the choice
    # without opening config.json (issues #34, #61).
    info_table.add_row("Fabric", _fabric_summary(config))
    console.print(info_table)
    console.print()

    # Write PID file
    _write_pid()

    # Select engine path:
    #   distributed_mode="member"  → no local vLLM. aiohttp + discovery only so
    #                                the head can place a Ray worker on us via
    #                                eugr's launcher.
    #   distributed_mode="solo"    → single-node vLLM on this host.
    #   distributed_mode="head"    → vLLM sharded across this host + peer_ips
    #                                via eugr's launch-cluster.sh.
    mode = (config.distributed_mode or "solo").lower()
    in_container = os.environ.get("AINODE_IN_CONTAINER") == "1" or getattr(args, "in_container", False)

    # Member mode: no local engine — just run the API server and wait for
    # the head to assign a distributed shard via eugr's launcher.
    if mode == "member":
        console.print(
            "  [bold cyan]Member mode[/bold cyan] — no local inference engine. "
            "Awaiting work from the cluster head.\n"
        )
        from ainode.api.server import run_server
        try:
            run_server(config=config, engine=None)
        except KeyboardInterrupt:
            console.print("\n  [yellow]Shutting down...[/yellow]")
        finally:
            _remove_pid()
        return

    # Solo/head mode with no model configured: start the server and wait.
    # The user picks a model from the web UI — no model required at startup.
    if not config.model:
        console.print(
            "  [dim]No model configured — open http://localhost:3000 to pick one.[/dim]\n"
        )
        from ainode.api.server import run_server
        try:
            run_server(config=config, engine=None)
        except KeyboardInterrupt:
            console.print("\n  [yellow]Shutting down...[/yellow]")
        finally:
            _remove_pid()
        return

    from ainode.engine.backends import get_backend
    from ainode.engine.backends.eugr import NO_VLLM_MESSAGE, EugrBackendError
    backend_name = (config.engine_backend or DEFAULT_ENGINE_BACKEND).lower()

    # Host start guard (issue #61). The eugr backend drives `vllm serve`
    # directly. It is the in-container path, NOT a way to run a container
    # from the host. Outside the container with no vLLM on PATH its Popen
    # dies with FileNotFoundError, which used to reach the user as a raw
    # traceback. Say what is actually wrong instead.
    if not in_container and backend_name == "eugr" and shutil.which("vllm") is None:
        console.print(
            f"  [red]Cannot start the engine on this host.[/red]\n\n  {NO_VLLM_MESSAGE}\n"
        )
        _remove_pid()
        sys.exit(1)

    if in_container or config.engine_strategy == "docker":
        engine = get_backend(config)
    elif importlib.util.find_spec("vllm") is None:
        # Legacy host-venv path is dev-only and needs vLLM importable in THIS
        # interpreter. When it isn't, VLLMEngine starts, dies with "No module
        # named 'vllm'", and the reason lands only in ~/.ainode/logs/vllm.log —
        # the boot banner still says "Engine starting in background", so the node
        # looks healthy while serving nothing (observed on spark-4, 2026-08-14).
        # A node configured for a container backend should use it rather than
        # launch a certain failure.
        console.print(
            f"  [dim]vLLM not importable in this interpreter, using the "
            f"{backend_name} engine backend.[/dim]"
        )
        engine = get_backend(config)
    else:
        from ainode.engine.vllm_engine import VLLMEngine
        engine = VLLMEngine(config)

    # Start the engine in the background — do NOT block the web server.
    # The web UI comes up immediately and shows a loading state while the
    # model warms up. Users should never have to stare at a terminal.
    try:
        launched = engine.start()
    except EugrBackendError as exc:
        console.print(f"  [red]Cannot start the engine.[/red]\n\n  {exc}\n")
        _remove_pid()
        sys.exit(1)
    if not launched:
        console.print("  [red]Failed to launch engine process.[/red] Check logs in ~/.ainode/logs/")
        _remove_pid()
        sys.exit(1)

    console.print("  [dim]Engine starting in background — web UI is ready now.[/dim]\n")
    console.print("  [bold green]Open http://localhost:3000 to get started.[/bold green]\n")

    # Run the API/web server immediately — it handles the engine's loading
    # state and surfaces it to the browser (spinner, status endpoint, etc.).
    from ainode.api.server import run_server
    try:
        run_server(config=config, engine=engine)
    except KeyboardInterrupt:
        console.print("\n  [yellow]Shutting down...[/yellow]")
    finally:
        try:
            engine.stop()
        except Exception:
            pass
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
    from ainode.core.gpu import detect_gpu
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

    # Engine health check — dispatch by engine_strategy for parity with cmd_start.
    if config.engine_strategy == "docker":
        from ainode.engine.backends import get_backend
        engine = get_backend(config)
    else:
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

    if getattr(args, "hf_token", None) is not None:
        token = args.hf_token.strip()
        if token:
            config.hf_token = token
            config.save()
            console.print("  [green]Hugging Face token saved.[/green]")
            console.print("  [dim]Gated models (Llama, Gemma, etc.) will now be accessible.[/dim]")
        else:
            config.hf_token = None
            config.save()
            console.print("  [yellow]Hugging Face token cleared.[/yellow]")
        return

    # Default: --show
    console.print(_banner())

    from dataclasses import asdict
    data = asdict(config)

    table = Table(title="Configuration", border_style="cyan")
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    for key, value in data.items():
        # The cluster secret is what a joined node signs discovery with. It is
        # scrubbed from GET /api/config for the same reason it is masked here:
        # `ainode config` output gets pasted into issues and chat.
        if key == "cluster_secret":
            display = "[dim]set (hidden)[/dim]" if value else "[dim]not set[/dim]"
        else:
            display = str(value) if value is not None else "[dim]not set[/dim]"
        table.add_row(key, display)

    console.print(table)
    console.print()
    console.print(f"  Config file: [dim]{AINODE_HOME / 'config.json'}[/dim]")
    console.print()


def _engine_log_file(config):
    """The log file THIS node's configured backend actually writes.

    Each backend owns its own filename: the nvidia backend writes
    ``nvidia-vllm.log`` (``nvidia-distributed.log`` when this node is a
    distributed head), the eugr backend writes ``vllm.log``. So ask the backend
    instead of hardcoding one of them here. Hardcoding ``vllm.log`` is why
    ``ainode logs -f`` tailed a long-dead file on every node running the default
    backend, and why every documented ``ainode logs -f | grep ...`` step read
    nothing (issue #164).

    Falls back to ``vllm.log`` for a config naming a backend this build does not
    know: `ainode logs` must still show something rather than raise.
    """
    try:
        from ainode.engine.backends import get_backend
        return get_backend(config).log_path
    except Exception:
        return VLLM_LOG


def cmd_logs(args):
    """Show or tail the engine log the configured backend writes."""
    config = NodeConfig.load()
    backend_name = (config.engine_backend or DEFAULT_ENGINE_BACKEND).lower()
    log_file = _engine_log_file(config)

    if not log_file.exists():
        console.print(f"  [dim]No log file found at[/dim] {log_file}")
        console.print(
            f"  [dim]({backend_name} backend; it writes that file once an engine"
            " starts.)[/dim]"
        )
        present = sorted(f.name for f in LOGS_DIR.glob("*.log")) if LOGS_DIR.exists() else []
        if present:
            console.print(f"  [dim]Logs present in {LOGS_DIR}:[/dim] {', '.join(present)}")
        console.print("  [dim]Start AINode first: [bold]ainode start[/bold][/dim]")
        return

    if args.follow:
        console.print(
            f"  [dim]Tailing {log_file} ({backend_name} backend; Ctrl+C to stop)[/dim]\n"
        )
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
        console.print(
            f"  [dim]Last {len(lines)} lines of {log_file}"
            f" ({backend_name} backend):[/dim]\n"
        )
        for line in lines:
            console.print(f"  {line.rstrip()}")
        console.print()



def cmd_role(args):
    """Set or show this node's cluster role."""
    config = NodeConfig.load()

    job = getattr(args, "job", None)

    _JOB_TO_MODE = {
        "master": "head",
        "worker": "member",
        "solo":   "solo",
    }
    _MODE_TO_JOB = {v: k for k, v in _JOB_TO_MODE.items()}

    if job is None:
        # Show current role
        current = config.distributed_mode or "solo"
        label = _MODE_TO_JOB.get(current, current)
        console.print(f"\n  This node is: [bold cyan]{label}[/bold cyan]  (distributed_mode={current})\n")
        console.print("  To change:  [dim]ainode role master | worker | solo[/dim]\n")
        return

    mode = _JOB_TO_MODE[job]
    config.distributed_mode = mode

    # Workers don't need a model — clear it so the engine is skipped on start
    if job == "worker":
        config.model = None
        console.print("\n  [bold cyan]Role set to: worker[/bold cyan]")
        console.print("  This node will start immediately and wait for the")
        console.print("  master to assign work. No model required.\n")
    elif job == "master":
        console.print("\n  [bold green]Role set to: master[/bold green]")
        console.print("  This node will manage the cluster and run the inference")
        console.print("  engine. Pick a model via the web UI at http://localhost:3000\n")
    else:
        console.print("\n  [bold]Role set to: solo[/bold]")
        console.print("  Standalone node — not part of a cluster.\n")

    config.save()
    console.print("  [dim]Restart AINode to apply: sudo systemctl restart ainode[/dim]\n")


def cmd_service(args):
    """Manage AINode systemd service."""
    from ainode.service.systemd import (
        install_service,
        enable_service,
        start_service,
        uninstall_service,
        status_service,
        get_journal_lines,
        is_installed,
    )

    user_mode = getattr(args, "user", False)
    action = getattr(args, "service_action", None)

    if action == "install":
        # The systemd unit lives on the HOST, not in this container. Inside the
        # container there is no systemd bus, no `systemctl` binary, and no
        # bind-mount of /etc/systemd/system — so writing the unit + enabling it
        # here would vanish into the container overlay AND then crash on the
        # first `systemctl` call, all while printing a false "✓ installed". The
        # host wrapper (install.sh) renders the unit directly; there is no valid
        # in-container install path. Refuse clearly and point at the host-side
        # migration instead of silently succeeding then crashing.
        in_container = os.environ.get("AINODE_IN_CONTAINER") == "1"
        if in_container:
            console.print(
                "  [yellow]`ainode service install` must run on the host, not inside the container.[/yellow]"
            )
            console.print(
                "  The systemd unit is managed on the host — there is no systemd bus here."
            )
            console.print("  To (re)install or migrate the unit on this node, re-run the installer:")
            console.print("    [bold]curl -fsSL https://ainode.dev/install | bash[/bold]")
            console.print("  (idempotent — it re-renders the unit and preserves your config.json).")
            console.print("  Made in Texas")
            return
        force = getattr(args, "force", False)
        if is_installed(user_mode=user_mode) and not force:
            console.print("  [yellow]AINode service is already installed.[/yellow]")
            console.print("  [dim](use 'ainode service install --force' to re-render the unit)[/dim]")
        else:
            console.print("  Installing AINode service...")
            install_service(user_mode=user_mode, reload=not in_container, force=force)
            console.print("  [green]✓[/green] Unit file written")
        console.print("  Enabling service...")
        enable_service(user_mode=user_mode)
        console.print("  [green]✓[/green] Service enabled")
        console.print("  Starting service...")
        start_service(user_mode=user_mode)
        console.print("  [green]✓[/green] Service started")
        console.print()
        console.print("  AINode will now start automatically on boot.")
        console.print("  Made in Texas")

    elif action == "uninstall":
        if not is_installed(user_mode=user_mode):
            console.print("  [yellow]AINode service is not installed.[/yellow]")
            return
        console.print("  Stopping and removing AINode service...")
        uninstall_service(user_mode=user_mode)
        console.print("  [green]✓[/green] Service removed")
        console.print("  Made in Texas")

    elif action == "status":
        if not is_installed(user_mode=user_mode):
            console.print("  AINode service: [dim]not installed[/dim]")
            return
        info = status_service(user_mode=user_mode)
        state = info["state"]
        color = {"active": "green", "inactive": "dim", "failed": "red"}.get(state, "yellow")
        console.print(f"  AINode service: [{color}]{state}[/{color}]")
        console.print(f"  Enabled: {'yes' if info['enabled'] else 'no'}")
        if info["journal_lines"]:
            console.print()
            console.print("  Recent logs:")
            for line in info["journal_lines"][-10:]:
                console.print(f"    {line}")
        console.print()
        console.print("  Made in Texas")

    elif action == "logs":
        lines = getattr(args, "lines", 50)
        journal = get_journal_lines(user_mode=user_mode, lines=lines)
        if journal:
            for line in journal:
                console.print(line)
        else:
            console.print("  No journal entries found for AINode.")

    else:
        console.print("  Usage: ainode service {install|uninstall|status|logs}")


def cmd_auth(args):
    """Manage API key authentication."""
    from ainode.auth.middleware import AuthConfig

    action = getattr(args, "auth_action", None)
    auth_cfg = AuthConfig.load()

    if action == "enable":
        entry = auth_cfg.enable()
        console.print("  [green]Auth enabled.[/green]")
        if entry["key"]:
            console.print(f"  API key: {entry['key']}")
            console.print(f"  Key ID:  {entry['id']}")
        else:
            # Keys are stored hashed, so an existing one cannot be printed again.
            console.print(f"  Using the {len(auth_cfg.api_keys)} key(s) this node "
                          "already has (stored hashed, so not shown again).")
            console.print("  Lost it? ainode auth new-key")
        console.print()
        console.print("  Use: Authorization: Bearer <key>")
        console.print("  Dashboard: paste the key under Config > API access.")
        console.print("  Made in Texas")

    elif action == "disable":
        auth_cfg.disable()
        console.print("  [yellow]Auth disabled.[/yellow] All requests allowed.")
        console.print("  Made in Texas")

    elif action == "status":
        state = "[green]enabled[/green]" if auth_cfg.enabled else "[dim]disabled[/dim]"
        console.print(f"  Auth: {state}")
        console.print(f"  Keys: {len(auth_cfg.api_keys)}")
        if not auth_cfg.enabled:
            console.print("  API open, no key set" if not auth_cfg.api_keys
                          else "  API open, key set but not required")
        console.print("  Made in Texas")

    elif action == "new-key":
        entry = auth_cfg.generate_key()
        console.print("  [green]New API key generated.[/green]")
        console.print(f"  API key: {entry['key']}")
        console.print(f"  Key ID:  {entry['id']}")
        console.print("  Made in Texas")

    else:
        console.print("  Usage: ainode auth {enable|disable|status|new-key}")


def cmd_prune_images(args):
    """Remove the AINode images an update replaced.

    ``ainode update`` pulls a release and never removed the one it replaced, so
    a node carried every release it had ever run: 180 images and 226 GB
    reclaimable on Spark-1 (#184). This is the step that reclaims them, run by
    the host wrapper AFTER the new version is confirmed serving, so a failed
    update still has an image to fall back to.

    Decisions live in ainode.core.image_prune, which is a pure function of a
    ``docker images`` listing: ``--images-from`` replays somebody else's listing
    (a node you are not on) and prints what would happen, touching nothing.
    """
    from ainode.core.image_prune import (
        DOCKER_IMAGES_FORMAT,
        format_plan,
        parse_images,
        prune_images,
        running_image,
    )

    keep = max(0, int(getattr(args, "keep_images", 1) or 0))
    listing = getattr(args, "images_from", None)
    rows = None
    dry_run = bool(getattr(args, "dry_run", False))
    if listing:
        # A listing from elsewhere can only ever be a plan: the images it names
        # are not the images on this host.
        dry_run = True
        try:
            rows = parse_images(Path(listing).read_text())
        except OSError as exc:
            console.print(f"  [red]Cannot read {listing}: {exc}[/red]")
            return 1

    current = getattr(args, "current", None) or running_image(__version__)
    if not current:
        console.print("  [red]Cannot tell which image this node runs.[/red]")
        console.print("  Name it: ainode prune-images --current "
                      "ghcr.io/getainode/ainode:X.Y.Z")
        return 1

    try:
        plan, log = prune_images(current, keep, dry_run=dry_run, rows=rows)
    except RuntimeError as exc:
        console.print(f"  [red]{exc}[/red]")
        console.print(f"  (listing command: docker images --format "
                      f"'{DOCKER_IMAGES_FORMAT}')")
        return 1

    console.print(format_plan(plan, verbose=bool(getattr(args, "verbose", False))))
    for line in log:
        console.print(f"  {line}")
    if dry_run and not plan.refused:
        console.print("  (dry run: nothing was removed)")
    console.print("  Made in Texas")
    return 0
# =============================================================================
#  Joining a cluster
# =============================================================================
#
# Two halves, and neither of them is the web. On the MASTER, `ainode cluster
# token` mints a single-use expiring token and prints the line to paste. On the
# JOINER, `ainode join <master> <token>` spends it and writes the config keys
# that make this node a member. Joining used to mean hand-editing config.json on
# the new box, which is what joining pollux took (#208).
#
# The token is minted ON THE BOX, never over the API: a minting endpoint would
# have to be keyless to be useful to a node that has not joined, and a keyless
# endpoint that hands out credentials is the thing this design exists to avoid.


def _http_post_json(url: str, payload: dict, timeout: float = 15.0):
    """POST JSON with the stdlib and return ``(status, body_dict_or_None)``.

    urllib rather than aiohttp or requests: this runs from `ainode join` on a
    fresh node, and the CLI's startup cost is paid by every other subcommand too.
    A status of 0 means the request never got an answer.
    """
    import json as _json
    import urllib.error
    import urllib.request

    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, _json.loads(raw)
            except ValueError:
                return resp.status, None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, _json.loads(raw)
        except ValueError:
            return exc.code, None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return 0, {"error": {"message": str(exc)}}


def _error_text(body, fallback: str) -> str:
    """The message out of an error body, in either shape this API uses."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if err:
            return str(err)
    return fallback


def _primary_address() -> str:
    """This node's address as another node would reach it.

    A UDP connect to a public address picks the interface the default route uses
    without sending a packet; the hostname is the fallback when there is no route.
    """
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return socket.gethostname()


def cmd_cluster(args):
    """Cluster commands run on the box: mint and list join tokens."""
    from ainode.cluster.join import (
        DEFAULT_TTL_SECONDS,
        DEFAULT_WEB_PORT,
        JoinTokenStore,
        ensure_cluster_secret,
        join_command,
    )

    action = getattr(args, "cluster_action", None)

    if action == "token":
        config = NodeConfig.load()
        ttl = int(getattr(args, "ttl", None) or DEFAULT_TTL_SECONDS)
        try:
            minted = JoinTokenStore().mint(ttl_seconds=ttl)
        except ValueError as exc:
            console.print(f"  [red]{exc}[/red]")
            raise SystemExit(2)

        secret, generated = ensure_cluster_secret(config)
        host = (getattr(config, "master_address", None) or "").strip()
        if not host:
            port = int(getattr(config, "web_port", DEFAULT_WEB_PORT) or DEFAULT_WEB_PORT)
            host = f"{_primary_address()}:{port}"

        console.print()
        console.print("  [bold green]Join token minted.[/bold green]  "
                      f"[dim]valid {ttl / 60:.0f} min, one use, id {minted.token_id}[/dim]")
        console.print()
        console.print("  Run this on the joining node:")
        console.print()
        # soft_wrap: Rich would fold this at the terminal width, and a join
        # command broken across two lines is one an operator pastes wrong.
        console.print(f"    [bold]{join_command(host, minted.token)}[/bold]",
                      soft_wrap=True)
        console.print()
        if generated:
            console.print("  [yellow]This master had no cluster_secret, so one was "
                          "generated and saved.[/yellow]")
            console.print("  [dim]Every node in the cluster must end up with the same "
                          "value: signed discovery drops announcements it cannot "
                          "verify, so a peer that did not join through a token needs "
                          "the secret set by hand.[/dim]")
        elif not secret:
            console.print("  [yellow]This master has no cluster_secret: discovery on "
                          "this cluster is unauthenticated.[/yellow]")
        console.print(f"  [dim]Tokens live hashed in "
                      f"{AINODE_HOME / 'join-tokens.json'}. Lost one? Mint another.[/dim]")
        console.print("  Made in Texas")
        console.print()
        return

    if action == "tokens":
        live = JoinTokenStore().live()
        if not live:
            console.print("  No live join tokens. Mint one: ainode cluster token")
            return
        table = Table(title="Live join tokens", border_style="cyan")
        table.add_column("ID", style="bold cyan", no_wrap=True)
        table.add_column("Expires in")
        now = time.time()
        for record in live:
            left = max(0, int(float(record.get("expires_at") or 0) - now))
            table.add_row(str(record.get("id") or "?"), f"{left // 60}m {left % 60}s")
        console.print(table)
        return

    console.print("  Usage: ainode cluster {token|tokens}")


def cmd_join(args):
    """Join this node to a cluster using a token minted on the master."""
    from ainode.cluster.join import (
        apply_join,
        join_url,
        parse_host_port,
        version_refusal,
    )

    target = (getattr(args, "master", "") or "").strip()
    token = (getattr(args, "token", "") or "").strip()
    name = (getattr(args, "name", "") or "").strip()
    interface = (getattr(args, "interface", "") or "").strip()
    allow_mismatch = bool(getattr(args, "allow_version_mismatch", False))

    try:
        host, port = parse_host_port(target)
        url = join_url(target)
    except ValueError as exc:
        console.print(f"  [red]{exc}[/red]")
        console.print("  Usage: ainode join <master-host>:3000 <token>")
        raise SystemExit(2)

    console.print(f"\n  Joining the cluster at [bold]{host}:{port}[/bold] ...")
    status, body = _http_post_json(url, {"token": token, "node_name": name})

    if status == 0:
        console.print(f"  [red]Could not reach {url}[/red]")
        console.print(f"  [dim]{_error_text(body, 'no answer')}[/dim]")
        raise SystemExit(1)
    if status == 429:
        console.print("  [red]The master is rate limiting join attempts from this "
                      "address.[/red] Wait a minute and try again.")
        raise SystemExit(1)
    if status != 200 or not isinstance(body, dict):
        console.print(f"  [red]The master refused the join (HTTP {status}).[/red]")
        console.print(f"  [dim]{_error_text(body, 'no reason given')}[/dim]")
        raise SystemExit(1)

    refusal = version_refusal(__version__, body.get("ainode_version", ""),
                              allow_mismatch=allow_mismatch)
    if refusal:
        console.print("  [red]Refusing to join: version mismatch.[/red]")
        console.print(f"  {refusal}")
        raise SystemExit(1)

    written = apply_join(body, node_name=name, interface=interface)

    console.print("  [bold green]Joined.[/bold green]")
    console.print(f"    cluster_id        {written.get('cluster_id')}")
    console.print(f"    master_address    {written.get('master_address', 'unchanged')}")
    console.print(f"    discovery_port    {written.get('discovery_port', 'unchanged')}")
    console.print("    cluster_role      worker  [dim](distributed_mode=member)[/dim]")
    if written.get("cluster_secret"):
        console.print("    cluster_secret    [dim]set (hidden)[/dim]")
    else:
        console.print("    cluster_secret    [yellow]not set by the master: discovery "
                      "on this cluster is unauthenticated[/yellow]")
    if written.get("cluster_interface"):
        console.print(f"    cluster_interface {written['cluster_interface']}")
    console.print(f"  [dim]Written to {AINODE_HOME / 'config.json'}. "
                  "No other key was touched.[/dim]")
    console.print()

    # A freshly joined node is the one place a restart from here is right: the
    # config it just wrote is only read at startup, and this node is not serving
    # anything yet. Everywhere else, restarting the service is the operator's
    # call, because it sweeps the engine containers with it.
    _restart_after_join()
    console.print("  Made in Texas")
    console.print()


def _restart_after_join() -> None:
    """Restart the service to apply the join, or print the exact command."""
    from ainode.service import systemd

    for user_mode in (False, True):
        if not systemd.is_installed(user_mode=user_mode):
            continue
        scope = "--user " if user_mode else ""
        try:
            systemd.restart_service(user_mode=user_mode)
        except Exception as exc:  # noqa: BLE001 - every systemctl failure gets the same advice
            console.print(f"  [yellow]Could not restart the service: {exc}[/yellow]")
            console.print(f"  Run it yourself:  sudo systemctl {scope}restart ainode")
            return
        console.print(f"  [green]Restarted ainode.service[/green] "
                      f"[dim](systemctl {scope}restart ainode)[/dim]")
        return
    # No unit here: the usual case, because this CLI runs inside the container
    # where there is no systemd bus. The host wrapper forwards the command.
    console.print("  Restart AINode on the HOST to apply:")
    console.print("    [bold]sudo systemctl restart ainode[/bold]")


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
    start_parser.add_argument(
        "--in-container",
        action="store_true",
        help="Signal that the CLI is running inside the AINode image (docker-entrypoint.sh sets this).",
    )
    start_parser.set_defaults(func=cmd_start)

    # role
    role_parser = subparsers.add_parser("role", help="Set or show this node's cluster role")
    role_parser.add_argument(
        "job",
        nargs="?",
        choices=["master", "worker", "solo"],
        help="master = head node (runs engine, manages cluster), "
             "worker = member node (no engine, waits for head), "
             "solo = standalone (default)",
    )
    role_parser.set_defaults(func=cmd_role)

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
    config_parser.add_argument(
        "--hf-token",
        dest="hf_token",
        metavar="TOKEN",
        help="Set Hugging Face token for gated models (Llama, Gemma, etc.). Pass empty string to clear.",
    )
    config_parser.set_defaults(func=cmd_config)

    # logs
    logs_parser = subparsers.add_parser("logs", help="Show vLLM logs")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Tail logs in real-time")
    logs_parser.add_argument("--lines", "-n", type=int, default=50, help="Number of lines to show (default: 50)")
    logs_parser.set_defaults(func=cmd_logs)

    # service
    service_parser = subparsers.add_parser("service", help="Manage AINode systemd service")
    service_parser.add_argument(
        "--user", action="store_true", help="Use user-level systemd (no sudo required)"
    )
    service_sub = service_parser.add_subparsers(dest="service_action")
    svc_install_parser = service_sub.add_parser(
        "install", help="Install, enable, and start AINode service"
    )
    svc_install_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render the unit file even if the service is already installed",
    )
    service_sub.add_parser("uninstall", help="Stop, disable, and remove AINode service")
    service_sub.add_parser("status", help="Show AINode service status")
    svc_logs_parser = service_sub.add_parser("logs", help="Show AINode service logs")
    svc_logs_parser.add_argument("-n", "--lines", type=int, default=50, help="Number of log lines")
    service_parser.set_defaults(func=cmd_service)

    # cluster: token minting, on the box. See the cmd_cluster block above for
    # why minting is deliberately not an API route.
    cluster_parser = subparsers.add_parser(
        "cluster", help="Cluster commands: mint a join token for a new node")
    cluster_sub = cluster_parser.add_subparsers(dest="cluster_action")
    token_parser = cluster_sub.add_parser(
        "token", help="Mint a single-use join token and print the command to paste")
    token_parser.add_argument(
        "--ttl", type=int, default=None, metavar="SECONDS",
        help="How long the token stays valid (default 1800, i.e. 30 minutes)")
    cluster_sub.add_parser("tokens", help="List join tokens that are still valid")
    cluster_parser.set_defaults(func=cmd_cluster)

    # join
    join_parser = subparsers.add_parser(
        "join", help="Join this node to a cluster with a token from the master")
    join_parser.add_argument("master", metavar="HOST[:PORT]",
                             help="The master's address, e.g. 10.0.0.1:3000")
    join_parser.add_argument("token", help="A join token minted on the master")
    join_parser.add_argument("--name", default="", help="Name this node announces")
    join_parser.add_argument(
        "--interface", default="",
        help="NIC the cluster fabric binds (empty = autodetect at startup)")
    join_parser.add_argument(
        "--allow-version-mismatch", action="store_true",
        dest="allow_version_mismatch",
        help="Join even when the master runs a different AINode release")
    join_parser.set_defaults(func=cmd_join)

    # auth
    auth_parser = subparsers.add_parser("auth", help="Manage API key authentication")
    auth_sub = auth_parser.add_subparsers(dest="auth_action")
    auth_sub.add_parser("enable", help="Enable API key auth")
    auth_sub.add_parser("disable", help="Disable API key auth")
    auth_sub.add_parser("status", help="Show auth status")
    auth_sub.add_parser("new-key", help="Generate a new API key")
    auth_parser.set_defaults(func=cmd_auth)

    # prune-images: reclaim the images an update replaced (#184)
    prune_parser = subparsers.add_parser(
        "prune-images",
        help="Remove AINode images older than the running release (keeps 1 rollback)",
    )
    prune_parser.add_argument(
        "--keep-images", type=int, default=1, metavar="N",
        help="Rollback generations to keep below the running release (default 1)")
    prune_parser.add_argument(
        "--current", metavar="IMAGE",
        help="The image this node runs (default: AINODE_HOME/image.env, then "
             "$AINODE_IMAGE, then this version's ghcr tag)")
    prune_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the decision and remove nothing")
    prune_parser.add_argument(
        "--images-from", metavar="FILE",
        help="Decide against a saved `docker images` listing instead of this "
             "host's. Implies --dry-run.")
    prune_parser.add_argument(
        "--verbose", action="store_true",
        help="Also print every image that is kept, and why")
    prune_parser.set_defaults(func=cmd_prune_images)

    # doctor: node health report. Exits non-zero on any FAIL so it can gate a
    # script; see ainode/cli/doctor.py for what each check means.
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check this node: config, docker, GPUs, disk, image, service, ports, peers",
    )
    doctor_parser.add_argument(
        "--peer",
        metavar="HOST",
        help="Run the same report on a peer over SSH and print its answer",
    )
    doctor_parser.add_argument(
        "--json", action="store_true",
        help="Emit the report as JSON instead of one line per check",
    )
    doctor_parser.add_argument(
        "--fix", action="store_true",
        help="Apply only the safe fixes (missing dirs, secrets-store mode, "
             "discovery port) and list the rest",
    )

    def _cmd_doctor(a):
        # Lazy import: the doctor reaches into the cluster, engine and secrets
        # modules, and `ainode --help` should not pay for any of them.
        from ainode.cli.doctor import cmd_doctor
        return cmd_doctor(a)

    doctor_parser.set_defaults(func=_cmd_doctor)

    args = parser.parse_args()

    if args.command is None:
        # No subcommand — default to start
        cmd_start(args)
    else:
        # A command that returns a code means it: `ainode prune-images` failing
        # inside `ainode update` has to be visible to the shell that called it.
        code = args.func(args)
        if code:
            sys.exit(int(code))


if __name__ == "__main__":
    main()
