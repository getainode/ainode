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

    # ADOPT BEFORE SWEEPING (#240). An engine container that is still running the
    # model this config names, on the right port, from the right image, and is
    # answering, is the engine this node is supposed to be serving: keeping it
    # turns a restart from a full model load (420 s on pollux, 834 s on castor)
    # into seconds. The decision has to be made here because both things that
    # destroy it are here: the sweep below and the boot engine's own launch.
    # Anything that does not match is left to the sweep, so a node whose model,
    # image or width changed reloads exactly as it did before.
    from ainode.engine import reconcile
    adoption = reconcile.adopt_boot_engines(config)
    for line in adoption.get("lines") or []:
        console.print(f"  [dim]{line}[/dim]")
    if adoption.get("lines"):
        console.print()

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

    # The engine this boot ADOPTED is already up and answering, so there is
    # nothing to launch: hand the same backend handle to the server (its
    # instance_id token names the running container, so stop() and is_running()
    # both reach it) with the container's own start time as its launch stamp, so a
    # bind wait or the UI reports the engine's real age and not this process's.
    adopted_primary = adoption.get("primary")
    if adopted_primary is not None:
        reconcile.attach_adopted_backend(engine, adopted_primary)
        console.print(
            f"  [dim]Engine adopted, no reload: {adopted_primary['model']} is "
            f"serving on :{adopted_primary['api_port']}.[/dim]\n")
        console.print("  [bold green]Open http://localhost:3000 to get started.[/bold green]\n")
    else:
        # Start the engine in the background, and do NOT block the web server.
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

        console.print("  [dim]Engine starting in background, the web UI is ready now.[/dim]\n")
        console.print("  [bold green]Open http://localhost:3000 to get started.[/bold green]\n")

    # Run the API/web server immediately — it handles the engine's loading
    # state and surfaces it to the browser (spinner, status endpoint, etc.).
    from ainode.api.server import run_server
    try:
        run_server(config=config, engine=engine)
    except KeyboardInterrupt:
        console.print("\n  [yellow]Shutting down...[/yellow]")
    finally:
        # A still-running engine container is LEFT running (#240): it is a sibling
        # spawned through docker.sock, every stacked engine already outlives this
        # process, and stopping it here is what made the next boot reload the model
        # from scratch. The next start adopts it; the line says how to free the GPU
        # instead. A container that is not running is still stopped through the
        # backend, which reaps a corpse and a head's peer containers.
        left = reconcile.keep_engines_on_shutdown(engine, config)
        if left:
            console.print(f"  [dim]{left}[/dim]")
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
        # The cluster secret is what a joined node signs discovery with, and the
        # HF token can write to the operator's own repositories. Both are scrubbed
        # from GET /api/config (api/server.py::SCRUBBED_CONFIG_KEYS) for the same
        # reason they are masked here: `ainode config` output gets pasted into
        # issues and chat.
        if key in ("cluster_secret", "hf_token"):
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


#: Printed after every change to auth.json, because the change is live: the
#: server stats the file per request (auth/middleware.py::reload_if_changed), so
#: an `ainode auth ...` on a running node takes effect on the next request rather
#: than at the next restart. It used to take effect at the next restart and say
#: nothing, so `ainode auth enable` looked like it had done nothing.
AUTH_LIVE_NOTE = ("  In effect now: the running node re-reads auth.json, "
                  "no restart needed.")


def _print_new_key(entry: dict) -> None:
    """The one place a plaintext key is printed, and what to do with it.

    It exists only in this output: the store keeps a SHA-256, so a key that is
    not copied now is gone and the fix is another key, never a lookup.
    """
    console.print(f"  API key: {entry['key']}")
    console.print(f"  Key ID:  {entry['id']}")
    if entry.get("name"):
        console.print(f"  Name:    {entry['name']}")
    console.print()
    console.print("  Shown once, stored hashed. Copy it now.")
    console.print("  Use: Authorization: Bearer <key>")
    console.print("  Dashboard: paste the key under Config > API access.")


def cmd_auth_key(args):
    """``ainode auth key {create|list|revoke}``: the operator's key handling.

    Separate from ``auth enable`` / ``disable`` on purpose: whether this node
    requires a key and which clients hold one are two different decisions, and the
    second one is the one an operator returns to (a laptop replaced, an
    integration retired). The fleet key is not in here and never will be: peers
    authenticate with a key derived from ``cluster_secret``
    (``ainode/auth/fleet.py``), so there is nothing to mint, list or revoke for
    them, and revoking the fleet means changing that secret.
    """
    from ainode.auth.middleware import AuthConfig

    action = getattr(args, "key_action", None)
    auth_cfg = AuthConfig.load()

    if action == "create":
        entry = auth_cfg.generate_key(getattr(args, "name", "") or "")
        console.print("  [green]New API key created.[/green]")
        _print_new_key(entry)
        if not auth_cfg.enabled:
            console.print()
            console.print("  [yellow]This node does not require a key yet[/yellow], so "
                          "the key works but nothing is refused without it.")
            console.print("  Require one: ainode auth enable")
        console.print(AUTH_LIVE_NOTE)
        console.print("  Made in Texas")

    elif action == "list":
        keys = auth_cfg.key_ids()
        state = "[green]required[/green]" if auth_cfg.enabled else "[dim]not required[/dim]"
        console.print(f"  API key: {state}")
        if not keys:
            console.print("  No keys on this node. Create one: "
                          "ainode auth key create --name <client>")
            console.print("  Made in Texas")
            return
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Created")
        for key in keys:
            table.add_row(key["id"], key["name"] or "[dim](unnamed)[/dim]",
                          key["created_at"] or "[dim]unknown[/dim]")
        console.print(table)
        console.print()
        console.print("  Keys are stored hashed and cannot be shown again.")
        console.print("  Revoke one: ainode auth key revoke <id>")
        console.print("  Made in Texas")

    elif action == "revoke":
        key_id = getattr(args, "key_id", "") or ""
        # Named by id, never by name: two clients may share a name, and a revoke
        # that guessed which one to remove is a revoke nobody can trust.
        if auth_cfg.revoke_key(key_id):
            console.print(f"  [green]Revoked key {key_id}.[/green]")
            console.print(f"  Keys left: {len(auth_cfg.api_keys)}")
            if auth_cfg.enabled and not auth_cfg.api_keys:
                console.print("  [yellow]That was the last key, and this node still "
                              "requires one.[/yellow] Nothing can call it until you "
                              "run: ainode auth key create --name <client>")
        else:
            console.print(f"  [yellow]No key with id '{key_id}' on this node.[/yellow]")
            console.print("  List them: ainode auth key list")
            console.print("  Made in Texas")
            return
        console.print(AUTH_LIVE_NOTE)
        console.print("  Made in Texas")

    else:
        console.print("  Usage: ainode auth key {create [--name NAME]|list|revoke <id>}")


# =============================================================================
#  Dashboard accounts: `ainode auth user ...` and `ainode auth session ...`
# =============================================================================
#
# The dashboard login (#261) is a second credential on the node: a named account
# with a password, in `<AINODE_HOME>/users.json`. These commands are how an
# operator creates the first one, because there is no sign-up page and there must
# not be: a route that mints the first admin over HTTP is a route anybody on the
# LAN can call before the operator does.
#
# The store is written HERE, on the box, and the running server picks the file up
# through `reload_if_changed` exactly as `ainode auth enable` is live. On the
# cluster's MASTER the change is then pushed to every peer
# (ainode/auth/replication.py); on a worker it is written and the operator is told
# it will be overwritten, because the master is the authority for accounts.
#
# Sessions are per node and are never replicated, so `ainode auth session ...`
# answers about this node only. That is the honest shape: a session is one
# browser's cookie against one node, and copying them would hand every node in the
# fleet a credential it never issued.

#: Same promise AUTH_LIVE_NOTE makes about auth.json, for the account store.
USERS_LIVE_NOTE = ("  In effect now: the running node re-reads users.json, "
                   "no restart needed.")

#: Printed on a node that is not the account authority, once, and then the change
#: is made anyway: refusing would leave an operator who typed the command on the
#: wrong box with no accounts and no explanation.
NOT_THE_MASTER_NOTE = (
    "  [yellow]Accounts are managed on the master[/yellow], and this node will be "
    "overwritten by the next sync from it. Run the same command there to keep it.")


def _users_store():
    """The account store on this node, or None with a printed reason.

    One guarded import for the whole CLI (``auth/replication.py::open_store``), so
    a release without the account store says so instead of raising an ImportError
    at the operator.
    """
    from ainode.auth.replication import open_store

    store = open_store()
    if store is None:
        console.print("  [yellow]This build has no account store[/yellow], so there "
                      "are no dashboard logins to manage on it.")
        console.print("  API keys still work: ainode auth key create --name <client>")
    return store


def _save_store(store) -> None:
    """Persist the store, if persisting is a separate step on it.

    The store's mutators are modelled on ``AuthConfig``'s, which save themselves,
    so this is normally a second write of identical bytes (temp-then-replace, so
    it costs nothing and cannot half-apply). It is here because the alternative
    failure is an account an operator was told about that is not on disk.
    """
    save = getattr(store, "save", None)
    if not callable(save):
        return
    try:
        save()
    except Exception as exc:  # pragma: no cover - a store that cannot write
        console.print(f"  [red]Could not write the account store: {exc}[/red]")
        raise SystemExit(1)


def _read_new_password(args, label: str = "Password") -> str:
    """The new password: one read of stdin, or two prompts that must agree.

    ``--password-stdin`` exists because ``getpass`` needs a TTY, and a script, a
    unit file or ``ssh host ainode ...`` has none. The installer's host wrapper
    forwards a piped command without ``docker exec -t`` for the same reason, so
    the pipe survives the container boundary. Only the newline the shell added is
    stripped, so a password may contain spaces.
    """
    from ainode.auth.replication import min_password_length

    minimum = min_password_length()
    if getattr(args, "password_stdin", False):
        password = sys.stdin.read().rstrip("\r\n")
    else:
        import getpass

        try:
            password = getpass.getpass(f"  {label}: ")
            again = getpass.getpass("  Again: ")
        except (EOFError, OSError):
            console.print("  [red]No terminal to ask for a password on.[/red]")
            console.print("  Pipe it instead: echo 'the password' | ainode auth user "
                          "add <name> --password-stdin")
            raise SystemExit(2)
        if password != again:
            console.print("  [red]The two passwords do not match.[/red] "
                          "Nothing was changed.")
            raise SystemExit(2)
    if len(password) < minimum:
        console.print(f"  [red]That password is too short.[/red] It needs at least "
                      f"{minimum} characters.")
        raise SystemExit(2)
    return password


def _account_rows(store) -> list:
    """``list_users()`` as a list of dicts, whatever it hands back."""
    try:
        rows = store.list_users()
    except Exception as exc:
        console.print(f"  [red]Could not read the accounts: {exc}[/red]")
        raise SystemExit(1)
    out = []
    for row in rows or []:
        if isinstance(row, dict):
            out.append(row)
        else:  # a store that lists names only
            out.append({"name": str(row)})
    return out


def _find_account(rows: list, name: str):
    for row in rows:
        if str(row.get("name") or "") == name:
            return row
    return None


def _account_enabled(row: dict) -> bool:
    """Is this account allowed to sign in? Both spellings are read."""
    if row.get("disabled") is True:
        return False
    return row.get("enabled") is not False


def _admin_count(store, rows: list) -> int:
    try:
        return int(store.admin_count())
    except Exception:
        return sum(1 for row in rows if str(row.get("role") or "") == "admin")


def _refuse_if_last_admin(store, rows: list, name: str, action: str) -> None:
    """Refuse an action that would leave the dashboard with no admin.

    With auth on and no admin account, the dashboard can only be opened with an
    API key, which is the state ``ainode doctor``'s Login check FAILs on. Refusing
    here is cheaper than explaining it there.
    """
    row = _find_account(rows, name)
    if row is None or str(row.get("role") or "") != "admin":
        return
    if _admin_count(store, rows) > 1:
        return
    console.print(f"  [red]{name} is the only admin account[/red], so it cannot be "
                  f"{action}.")
    console.print("  Add another admin first: ainode auth user add <name> --admin")
    raise SystemExit(2)


def _set_account_enabled(store, name: str, enabled: bool) -> bool:
    """Disable or enable one account, however this store spells it.

    The account contract names the two ROUTES (``/disable`` and ``/enable``) and
    leaves the store's own method name open, so the three spellings a store of
    this shape would use are tried in order. Raises ``NotImplementedError`` when
    it has none: the CLI says so and points at the route, rather than inventing a
    field name inside somebody else's file.
    """
    setter = getattr(store, "set_enabled", None)
    if callable(setter):
        return setter(name, enabled) is not False
    setter = getattr(store, "set_disabled", None)
    if callable(setter):
        return setter(name, not enabled) is not False
    setter = getattr(store, "enable_user" if enabled else "disable_user", None)
    if callable(setter):
        return setter(name) is not False
    raise NotImplementedError("this account store cannot disable an account")


def _session_rows(store, name: str) -> list:
    """``sessions_for(name)`` as a list of dicts, whatever it hands back."""
    try:
        rows = store.sessions_for(name)
    except Exception as exc:
        console.print(f"  [yellow]Could not read {name}'s sessions: {exc}[/yellow]")
        return []
    out = []
    for row in rows or []:
        out.append(dict(row) if isinstance(row, dict) else {"id": str(row)})
    return out


def _account_counts(store) -> dict:
    """Users, admins and sessions on this node, for ``ainode auth status``."""
    rows = _account_rows(store)
    sessions = 0
    for row in rows:
        sessions += len(_session_rows(store, str(row.get("name") or "")))
    return {"users": len(rows), "admins": _admin_count(store, rows),
            "sessions": sessions}


def _replicate_after_change(config, store) -> None:
    """Push a just-written account change to the peers, or say why it was not.

    Three answers, and each of them is a line an operator needs: a solo node has
    nobody to tell, a worker has just written a change the master will overwrite,
    and a master pushes now instead of leaving the peers to catch up on the
    server's own 60 second retry.
    """
    from ainode.auth.replication import (
        cluster_info,
        local_role,
        replicate_from_cli,
    )

    info = cluster_info(config)
    role = local_role(config, info=info)
    if role == "solo":
        return
    if role == "worker":
        console.print(NOT_THE_MASTER_NOTE)
        return

    try:
        users = store.export_users()
    except Exception as exc:
        console.print(f"  [yellow]Could not read the accounts to replicate: {exc}"
                      f"[/yellow]")
        return
    result = replicate_from_cli(config, users, info=info)
    pushed, failed = result.get("pushed") or [], result.get("failed") or []
    if result.get("reason"):
        console.print(f"  [yellow]Not replicated: {result['reason']}[/yellow]")
        return
    if pushed:
        console.print(f"  Replicated to {len(pushed)} peer(s): {', '.join(pushed)}")
    for label, detail in failed:
        console.print(f"  [yellow]{label} did not take the update ({detail})[/yellow]; "
                      "this node retries it every 60s while it is behind.")
    if not pushed and not failed:
        console.print("  No peers to replicate to yet.")


def cmd_auth_user(args):
    """``ainode auth user {add|list|remove|passwd|disable|enable}``."""
    action = getattr(args, "user_action", None)
    if action is None:
        console.print("  Usage: ainode auth user {add <name> [--admin] "
                      "[--password-stdin]|list|remove <name>|passwd <name>|"
                      "disable <name>|enable <name>}")
        return

    store = _users_store()
    if store is None:
        raise SystemExit(2)
    config = NodeConfig.load()
    # Account names are lowercase by contract ([a-z0-9._-]), so what an operator
    # typed with a capital is folded rather than refused. The name is echoed back
    # in every line below, so the fold is visible.
    name = str(getattr(args, "name", "") or "").strip().lower()

    if action == "list":
        rows = _account_rows(store)
        if not rows:
            console.print("  No dashboard accounts on this node.")
            console.print("  Add the first one: ainode auth user add <name> --admin")
            console.print("  Made in Texas")
            return
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Name")
        table.add_column("Role")
        table.add_column("State")
        table.add_column("Sessions")
        for row in sorted(rows, key=lambda r: str(r.get("name") or "")):
            account = str(row.get("name") or "")
            table.add_row(
                account,
                str(row.get("role") or "member"),
                "enabled" if _account_enabled(row) else "[yellow]disabled[/yellow]",
                str(len(_session_rows(store, account))),
            )
        console.print(table)
        console.print()
        console.print(f"  {_admin_count(store, rows)} admin(s) of {len(rows)} "
                      "account(s). Sessions are per node and are not replicated.")
        console.print("  Made in Texas")
        return

    if not name:
        console.print(f"  [red]Which account?[/red] ainode auth user {action} <name>")
        raise SystemExit(2)

    if action == "add":
        role = "admin" if getattr(args, "admin", False) else "member"
        password = _read_new_password(args)
        try:
            store.add_user(name, password, role)
        except ValueError as exc:
            console.print(f"  [red]{exc}[/red]")
            raise SystemExit(2)
        except Exception as exc:
            console.print(f"  [red]Could not add {name}: {exc}[/red]")
            raise SystemExit(1)
        _save_store(store)
        console.print(f"  [green]Account {name} added[/green] as {role}.")
        console.print("  Sign in at the dashboard with that name and password.")
        console.print(USERS_LIVE_NOTE)
        _replicate_after_change(config, store)
        console.print("  Made in Texas")
        return

    rows = _account_rows(store)
    if _find_account(rows, name) is None:
        console.print(f"  [yellow]No account named '{name}' on this node.[/yellow]")
        console.print("  List them: ainode auth user list")
        raise SystemExit(2)

    if action == "remove":
        _refuse_if_last_admin(store, rows, name, "removed")
        try:
            removed = store.remove_user(name)
        except Exception as exc:
            console.print(f"  [red]Could not remove {name}: {exc}[/red]")
            raise SystemExit(1)
        if removed is False:
            console.print(f"  [yellow]No account named '{name}' on this node.[/yellow]")
            raise SystemExit(2)
        _save_store(store)
        console.print(f"  [green]Account {name} removed.[/green] Its sessions on this "
                      "node are gone with it.")
        console.print(USERS_LIVE_NOTE)
        _replicate_after_change(config, store)
        console.print("  Made in Texas")
        return

    if action == "passwd":
        password = _read_new_password(args, label=f"New password for {name}")
        try:
            store.set_password(name, password)
        except ValueError as exc:
            console.print(f"  [red]{exc}[/red]")
            raise SystemExit(2)
        except Exception as exc:
            console.print(f"  [red]Could not set {name}'s password: {exc}[/red]")
            raise SystemExit(1)
        _save_store(store)
        console.print(f"  [green]Password changed for {name}.[/green]")
        console.print("  Existing sessions are untouched: sign them out with "
                      f"ainode auth session clear --user {name}")
        console.print(USERS_LIVE_NOTE)
        _replicate_after_change(config, store)
        console.print("  Made in Texas")
        return

    if action in ("disable", "enable"):
        enabled = action == "enable"
        if not enabled:
            _refuse_if_last_admin(store, rows, name, "disabled")
        try:
            _set_account_enabled(store, name, enabled)
        except NotImplementedError:
            console.print("  [yellow]This account store has no disable of its own."
                          "[/yellow] Use the API route instead, on this node's own "
                          "port, with an operator key:")
            console.print(f"    POST /api/auth/users/{name}/{action}")
            raise SystemExit(2)
        except Exception as exc:
            console.print(f"  [red]Could not {action} {name}: {exc}[/red]")
            raise SystemExit(1)
        _save_store(store)
        if enabled:
            console.print(f"  [green]Account {name} enabled.[/green]")
        else:
            console.print(f"  [yellow]Account {name} disabled.[/yellow] It cannot "
                          "sign in, and its sessions are no longer accepted.")
        console.print(USERS_LIVE_NOTE)
        _replicate_after_change(config, store)
        console.print("  Made in Texas")
        return

    console.print(f"  [red]Unknown account command '{action}'.[/red]")
    raise SystemExit(2)


def cmd_auth_session(args):
    """``ainode auth session {list|revoke|clear}``: this node's sign-ins.

    Per node on purpose (see the block comment above): a session is one browser's
    cookie against one node, so there is nothing here to fan out and nothing to
    replicate.
    """
    action = getattr(args, "session_action", None)
    if action is None:
        console.print("  Usage: ainode auth session {list [--user NAME]|"
                      "revoke <id>|clear [--user NAME]}")
        return

    store = _users_store()
    if store is None:
        raise SystemExit(2)
    only = str(getattr(args, "user", "") or "").strip().lower()
    rows = _account_rows(store)
    if only and _find_account(rows, only) is None:
        console.print(f"  [yellow]No account named '{only}' on this node.[/yellow]")
        raise SystemExit(2)
    names = [only] if only else [str(r.get("name") or "") for r in rows]

    if action == "list":
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Session")
        table.add_column("User")
        table.add_column("Created")
        table.add_column("Last seen")
        total = 0
        for account in sorted(names):
            for session in _session_rows(store, account):
                total += 1
                table.add_row(
                    str(session.get("id") or ""),
                    account,
                    str(session.get("created_at") or "[dim]unknown[/dim]"),
                    str(session.get("last_seen")
                        or session.get("last_used") or "[dim]unknown[/dim]"),
                )
        if not total:
            console.print("  No sessions on this node"
                          + (f" for {only}." if only else "."))
            console.print("  Made in Texas")
            return
        console.print(table)
        console.print()
        console.print(f"  {total} session(s) on this node. Sessions are per node: "
                      "signing out here does not sign out anywhere else.")
        console.print("  Made in Texas")
        return

    if action == "revoke":
        session_id = str(getattr(args, "session_id", "") or "").strip()
        if not session_id:
            console.print("  [red]Which session?[/red] ainode auth session revoke <id>")
            raise SystemExit(2)
        try:
            gone = store.revoke_session(session_id)
        except Exception as exc:
            console.print(f"  [red]Could not revoke that session: {exc}[/red]")
            raise SystemExit(1)
        if gone is False:
            console.print(f"  [yellow]No session '{session_id}' on this node."
                          "[/yellow] List them: ainode auth session list")
            raise SystemExit(2)
        console.print(f"  [green]Session {session_id} revoked.[/green]")
        console.print("  Made in Texas")
        return

    if action == "clear":
        revoked = 0
        for account in names:
            for session in _session_rows(store, account):
                session_id = str(session.get("id") or "")
                if not session_id:
                    continue
                try:
                    if store.revoke_session(session_id, user=account) is not False:
                        revoked += 1
                except Exception as exc:
                    console.print(f"  [yellow]{account}: {exc}[/yellow]")
        who = f"{only}" if only else "everybody"
        console.print(f"  [green]Signed out {who}[/green]: {revoked} session(s) "
                      "revoked on this node.")
        console.print("  Made in Texas")
        return

    console.print(f"  [red]Unknown session command '{action}'.[/red]")
    raise SystemExit(2)


def cmd_auth(args):
    """Manage API key authentication."""
    from ainode.auth.middleware import AuthConfig

    action = getattr(args, "auth_action", None)
    auth_cfg = AuthConfig.load()

    if action == "key":
        return cmd_auth_key(args)

    if action == "user":
        return cmd_auth_user(args)

    if action == "session":
        return cmd_auth_session(args)

    if action == "enable":
        entry = auth_cfg.enable(getattr(args, "name", "") or "first key")
        console.print("  [green]Auth enabled.[/green]")
        if entry["key"]:
            _print_new_key({**entry,
                            "name": getattr(args, "name", "") or "first key"})
        else:
            # Keys are stored hashed, so an existing one cannot be printed again.
            console.print(f"  Using the {len(auth_cfg.api_keys)} key(s) this node "
                          "already has (stored hashed, so not shown again).")
            console.print("  Lost it? ainode auth key create --name <client>")
            console.print()
            console.print("  Use: Authorization: Bearer <key>")
            console.print("  Dashboard: paste the key under Config > API access.")
        console.print()
        console.print("  Peers need no key of their own: a node-to-node call "
                      "authenticates with the key derived from cluster_secret.")
        console.print(AUTH_LIVE_NOTE)
        console.print("  Made in Texas")

    elif action == "disable":
        auth_cfg.disable()
        console.print("  [yellow]Auth disabled.[/yellow] All requests allowed.")
        console.print("  The keys are still on the node; nothing is refused "
                      "without one.")
        console.print(AUTH_LIVE_NOTE)
        console.print("  Made in Texas")

    elif action == "status":
        state = "[green]enabled[/green]" if auth_cfg.enabled else "[dim]disabled[/dim]"
        console.print(f"  Auth: {state}")
        console.print(f"  Keys: {len(auth_cfg.api_keys)}")
        if not auth_cfg.enabled:
            console.print("  API open, no key set" if not auth_cfg.api_keys
                          else "  API open, key set but not required")
        # The accounts half of the same question. Auth on with no admin account is
        # a dashboard only a key can open, which is what ainode doctor FAILs on, so
        # the number belongs beside the key count rather than behind another
        # command.
        from ainode.auth.replication import open_store

        store = open_store()
        if store is not None:
            counts = _account_counts(store)
            console.print(f"  Users: {counts['users']} "
                          f"({counts['admins']} admin)")
            console.print(f"  Sessions: {counts['sessions']} (this node only, never "
                          "replicated)")
            if auth_cfg.enabled and not counts["users"]:
                console.print("  [yellow]No dashboard account[/yellow], so the UI can "
                              "only be opened with an API key.")
                console.print("  Fix: ainode auth user add <name> --admin")
            elif auth_cfg.enabled and not counts["admins"]:
                console.print("  [yellow]No admin account[/yellow]: nobody can manage "
                              "users from the dashboard.")
                console.print("  Fix: ainode auth user add <name> --admin")
        # Whether this node can still talk to its own cluster with auth on. The
        # doctor says the same thing as a FAIL (cli/doctor.py::check_auth); this
        # is the version an operator sees while turning auth on.
        config = NodeConfig.load()
        if auth_cfg.enabled and not (config.cluster_secret or "").strip():
            console.print("  [yellow]No cluster_secret on this node[/yellow]: if it has "
                          "peers, their node-to-node calls answer 401.")
            console.print("  Fix: ainode join <master> <token>, or set the cluster's "
                          "shared cluster_secret in config.json")
        elif auth_cfg.enabled:
            console.print("  Peers authenticate with the key derived from cluster_secret")
        console.print("  Made in Texas")

    elif action == "new-key":
        # Kept working, and pointed at its replacement: `auth key create` is the
        # same act with a name on it.
        entry = auth_cfg.generate_key(getattr(args, "name", "") or "")
        console.print("  [green]New API key generated.[/green]")
        _print_new_key(entry)
        console.print("  (ainode auth key create --name <client> names it.)")
        console.print(AUTH_LIVE_NOTE)
        console.print("  Made in Texas")

    else:
        console.print("  Usage: ainode auth {enable|disable|status|key|user|session"
                      "|new-key}")
        console.print("         ainode auth key {create [--name NAME]|list|revoke <id>}")
        console.print("         ainode auth user {add <name> [--admin] "
                      "[--password-stdin]|list|remove <name>|passwd <name>|"
                      "disable <name>|enable <name>}")
        console.print("         ainode auth session {list [--user NAME]|revoke <id>|"
                      "clear [--user NAME]}")


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

#: What to run on the HOST when `--tailscale` is asked for inside the container
#: and the pair is not there yet.
#:
#: The image ships no tailscale binary and the daemon socket is not mounted, so
#: the cert step happens on the host. The installer's host wrapper does exactly
#: that and then calls back in here, so an operator who typed `ainode tls enable
#: --tailscale` on the host never sees this message; it is for a node whose
#: wrapper predates that (or someone running the command inside the container by
#: hand), and the manual commands it prints are the same two the wrapper runs.
TAILSCALE_ON_THE_HOST = """\
  This is the AINode container, which has no tailscale binary and no access to
  the tailnet daemon, and no tailnet pair is on disk yet.

  Run it on the HOST, where the wrapper does the cert step for you:

    ainode tls enable --tailscale

  If this node's wrapper is older than that, the two commands it would run are:

    sudo tailscale cert --cert-file {host_cert} --key-file {host_key} {name}
    sudo chmod 600 {host_key}

  ({host_dir} is the same directory as {cert_dir} in here, so the
  server reads what tailscale writes.) Then, back in the container:

    ainode tls enable --tailscale

  `tailscale status --json` on the host names the node: look for Self.DNSName."""

#: The one-line fix when `tailscale cert` is refused because the tailnet has no
#: HTTPS support turned on. Printed by the CLI and by the wrapper, same words.
TAILNET_HTTPS_FIX = ("HTTPS must be enabled for the tailnet: Tailscale admin "
                     "console > DNS > HTTPS Certificates > Enable.")


def cmd_tls(args):
    """Manage TLS on the API port."""
    from ainode.tls import (
        TLSConfig,
        cert_paths,
        certificate_info,
        generate_self_signed,
        install_pair,
        load_tls_config,
        renewal_decision,
        save_tls_config,
        tailnet_dns_name,
        tailnet_pair_paths,
        tailscale_cert,
        tls_dir,
    )
    from ainode.tls.certs import tailscale_binary
    from ainode.tls.config import DEFAULT_TLS_PORT, ensure_tls_dir, harden_key

    action = getattr(args, "tls_action", None)
    config = NodeConfig.load()
    tls = load_tls_config(config)

    if action == "enable":
        cert_arg = getattr(args, "cert", None)
        key_arg = getattr(args, "key", None)
        if bool(cert_arg) != bool(key_arg):
            console.print("  [red]--cert and --key go together.[/red] "
                          "Pass both, or neither to generate a self-signed pair.")
            sys.exit(1)
        port = int(getattr(args, "port", None) or tls.port or DEFAULT_TLS_PORT)
        if port == int(config.web_port):
            console.print(f"  [red]Port {port} is already the HTTP port.[/red] "
                          "TLS needs its own port; 3443 is the default.")
            sys.exit(1)
        ensure_tls_dir()
        default_cert, default_key = cert_paths()

        if getattr(args, "tailscale", False):
            # The name first, because both branches below need it and it is the
            # one thing findable from either side of the container boundary.
            name = tailnet_dns_name()
            if tailscale_binary():
                # A host, or a source install outside the container: do the cert
                # step right here.
                if not name:
                    console.print("  [red]Could not read this node's MagicDNS "
                                  "name[/red] from `tailscale status --json`. "
                                  "Is tailscaled up?")
                    sys.exit(1)
                default_cert, default_key = tailnet_pair_paths(name)
                try:
                    tailscale_cert(name, default_cert, default_key)
                except RuntimeError as exc:
                    console.print(f"  [red]{exc}[/red]")
                    console.print(f"  {TAILNET_HTTPS_FIX}")
                    sys.exit(1)
                console.print(f"  [green]Installed a Let's Encrypt certificate for "
                              f"{name}.[/green]")
            else:
                # The container. The installer's host wrapper runs the cert step
                # on the host and then calls this, so the pair is already in the
                # bind mount under the name it was issued for: adopt it rather
                # than fail on a binary that is never going to be in this image.
                pair = tailnet_pair_paths(name) if name else None
                if pair and pair[0].is_file() and pair[1].is_file():
                    default_cert, default_key = pair
                    harden_key(default_key)
                    console.print(f"  [green]Adopting the tailnet certificate for "
                                  f"{name}.[/green]")
                    console.print("  Obtained on the host by `tailscale cert`; this "
                                  "directory is the bind mount both sides share.")
                else:
                    # AINODE_HOST_HOME is set by the systemd unit the installer
                    # renders, so the message can name the path as the OPERATOR
                    # sees it rather than the container's /root/.ainode.
                    host_tls = os.path.join(
                        os.environ.get("AINODE_HOST_HOME") or str(AINODE_HOME), "tls")
                    shown = name or "<your-node>.<tailnet>.ts.net"
                    console.print("  [yellow]No tailscale binary here, and no "
                                  "tailnet pair on disk.[/yellow]")
                    console.print(TAILSCALE_ON_THE_HOST.format(
                        host_cert=os.path.join(host_tls, f"{shown}.crt"),
                        host_key=os.path.join(host_tls, f"{shown}.key"),
                        host_dir=host_tls, cert_dir=tls_dir(), name=shown))
                    sys.exit(1)
        elif cert_arg:
            try:
                installed_cert, installed_key = install_pair(cert_arg, key_arg)
            except FileNotFoundError as exc:
                console.print(f"  [red]No such file: {exc}[/red]")
                sys.exit(1)
            default_cert, default_key = installed_cert, installed_key
            console.print(f"  [green]Installed the pair into {tls_dir()}.[/green]")
            console.print("  Copied rather than referenced, because the server runs "
                          "in a container that only sees this directory.")
        else:
            try:
                made = generate_self_signed(default_cert, default_key)
            except RuntimeError as exc:
                console.print(f"  [red]{exc}[/red]")
                sys.exit(1)
            console.print("  [green]Generated a self-signed certificate.[/green]")
            console.print(f"  Valid {made['days']} days, via {made['tool']}.")
            console.print(f"  Names: {', '.join(made['sans'])}")

        new = TLSConfig(enabled=True, cert_file=str(default_cert),
                        key_file=str(default_key), port=port)
        written = save_tls_config(new)
        console.print()
        console.print(f"  Cert: {new.cert_file}")
        console.print(f"  Key:  {new.key_file} (mode 0600)")
        console.print(f"  Wrote the tls block to {written}")
        console.print()
        console.print(f"  [bold]Restart the node to open port {port}.[/bold] "
                      "The listener is built at boot.")
        console.print("    sudo systemctl restart ainode   (or `docker restart ainode`)")
        console.print(f"  HTTP on {config.web_port} is untouched: every client in a "
                      "fleet keeps using it.")
        if new.cert_file and certificate_info(new.cert_file).get("self_signed"):
            console.print("  A self-signed certificate needs `curl -k`, or the cert "
                          "added to the client's trust store.")
        console.print("  Made in Texas")

    elif action == "disable":
        save_tls_config(TLSConfig(enabled=False, cert_file=tls.cert_file,
                                  key_file=tls.key_file, port=tls.port))
        console.print("  [yellow]TLS disabled.[/yellow] The certificate is left in "
                      f"{tls_dir()}; nothing is deleted.")
        console.print("  Restart the node to close the port.")
        console.print("  Made in Texas")

    elif action == "renew":
        # The DECISION lives in ainode.tls.renew and the ACTING lives on the
        # host: tailscale is not in this image, and the new pair is only served
        # after a restart, which is a host operation too. `--check` is the
        # machine-readable half the installer's renewal timer reads through the
        # wrapper; without it this prints the same answer for a human.
        info = certificate_info(tls.cert_file) if tls.cert_file else {}
        decision = renewal_decision(tls, info, tailnet_name=tailnet_dns_name())
        if getattr(args, "check", False):
            for line in decision.as_lines():
                console.print(line, highlight=False, markup=False, soft_wrap=True)
            # Basenames, not paths: the renewal writes into the files the CONFIG
            # points at (which may be cert.pem from an earlier manual run, not
            # <name>.crt), and the host has to name them under ITS own
            # <AINODE_HOME>/tls rather than the container's.
            console.print(f"cert_name={os.path.basename(tls.cert_file or '')}",
                          highlight=False, markup=False, soft_wrap=True)
            console.print(f"key_name={os.path.basename(tls.key_file or '')}",
                          highlight=False, markup=False, soft_wrap=True)
            # 0 means "renew now", 10 means "decided, nothing to do": a timer can
            # act on the code alone and stay quiet the other 75 days.
            sys.exit(0 if decision.renew else 10)
        if decision.renew:
            console.print("  [yellow]This certificate is due for renewal.[/yellow]")
        else:
            console.print("  [green]No renewal needed.[/green]")
        console.print(f"  {decision.reason}")
        if decision.days_left is not None:
            console.print(f"  Days left: {decision.days_left}")
        console.print()
        console.print("  Renewal runs on the HOST: `tailscale cert` needs the "
                      "tailnet daemon, which this container has no path to, and "
                      "the new pair is only served after a restart.")
        console.print("    [bold]ainode tls renew[/bold]   (on the host)")
        console.print("  The installer's ainode-tls-renew.timer already runs that "
                      "daily, so an unattended node renews itself.")
        console.print("  Made in Texas")

    elif action == "status":
        state = "[green]enabled[/green]" if tls.enabled else "[dim]disabled[/dim]"
        console.print(f"  TLS:  {state}")
        console.print(f"  Port: {tls.port}  (HTTP stays on {config.web_port})")
        if not tls.cert_file:
            console.print("  Cert: none configured")
        else:
            info = certificate_info(tls.cert_file)
            console.print(f"  Cert: {tls.cert_file}")
            if not info.get("exists"):
                console.print("        [red]missing[/red]")
            elif info.get("error"):
                console.print(f"        [yellow]{info['error']}[/yellow]")
            else:
                kind = "self-signed" if info.get("self_signed") else "CA-issued"
                console.print(f"        {kind}, expires {info['expires']} "
                              f"({info['days_left']} days left)")
                if info.get("sans"):
                    console.print(f"        names: {', '.join(info['sans'])}")
        if not tls.enabled:
            name = tailnet_dns_name()
            if name:
                console.print(f"  Tailnet name: {name}")
                console.print("  Enable it: [bold]ainode tls enable --tailscale[/bold] "
                              "(a real certificate for that name)")
            else:
                console.print("  Enable it: ainode tls enable")
        console.print("  Made in Texas")

    else:
        console.print("  Usage: ainode tls {enable|disable|status|renew}")
        console.print("  enable [--cert PATH --key PATH] [--tailscale] [--port N]")
        console.print("  renew [--check]")


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
    auth_enable = auth_sub.add_parser("enable", help="Require an API key on /api and /v1")
    auth_enable.add_argument(
        "--name", default="",
        help="Name the key this mints, if there is no key yet (default: first key)")
    auth_sub.add_parser("disable", help="Stop requiring an API key")
    auth_sub.add_parser("status", help="Show auth status")
    auth_new_key = auth_sub.add_parser(
        "new-key", help="Generate a new API key (same as: auth key create)")
    auth_new_key.add_argument("--name", default="", help="Which client this key is for")
    # `auth key ...` is the operator's key handling: create one per client, see
    # which clients hold one, take one away.
    auth_key = auth_sub.add_parser("key", help="Create, list and revoke API keys")
    auth_key_sub = auth_key.add_subparsers(dest="key_action")
    auth_key_create = auth_key_sub.add_parser(
        "create", help="Mint a key and print it once")
    auth_key_create.add_argument(
        "--name", default="",
        help="Which client this key is for (shown by: ainode auth key list)")
    auth_key_sub.add_parser("list", help="The keys on this node, by id and name")
    auth_key_revoke = auth_key_sub.add_parser("revoke", help="Revoke a key by id")
    auth_key_revoke.add_argument("key_id", help="The key id from: ainode auth key list")

    # `auth user ...` is the DASHBOARD login (#261): a name and a password, which
    # is a different credential from an API key and managed on the cluster's
    # master. --password-stdin is the answer for anything with no terminal: getpass
    # needs a TTY, so the prompt works from a shell and not from a script, a unit
    # or `ssh host ainode ...`. The installer's wrapper allocates a TTY only when
    # it has one, so a piped password reaches the container.
    auth_user = auth_sub.add_parser(
        "user", help="Dashboard accounts: add, list, remove, change a password")
    auth_user_sub = auth_user.add_subparsers(dest="user_action")
    auth_user_add = auth_user_sub.add_parser(
        "add", help="Create a dashboard account (prompts for the password twice)")
    auth_user_add.add_argument("name", help="Account name (a-z, 0-9, dot, dash, "
                                           "underscore)")
    auth_user_add.add_argument(
        "--admin", action="store_true",
        help="Make it an admin: it may manage users and sessions")
    auth_user_add.add_argument(
        "--password-stdin", action="store_true", dest="password_stdin",
        help="Read the password from stdin instead of prompting. Use this with no "
             "TTY: echo 'the password' | ainode auth user add jason --admin "
             "--password-stdin")
    auth_user_sub.add_parser("list", help="The accounts on this node and their roles")
    auth_user_remove = auth_user_sub.add_parser(
        "remove", help="Delete an account (refused for the last admin)")
    auth_user_remove.add_argument("name", help="The account to delete")
    auth_user_passwd = auth_user_sub.add_parser(
        "passwd", help="Change an account's password")
    auth_user_passwd.add_argument("name", help="The account to change")
    auth_user_passwd.add_argument(
        "--password-stdin", action="store_true", dest="password_stdin",
        help="Read the new password from stdin instead of prompting (no TTY needed)")
    auth_user_disable = auth_user_sub.add_parser(
        "disable", help="Stop an account signing in, keeping it (refused for the "
                        "last admin)")
    auth_user_disable.add_argument("name", help="The account to disable")
    auth_user_enable = auth_user_sub.add_parser(
        "enable", help="Let an account sign in again")
    auth_user_enable.add_argument("name", help="The account to enable")

    # `auth session ...` is this node's sign-ins. Never a fan-out: a session is one
    # browser's cookie against one node, so there is nothing to replicate.
    auth_session = auth_sub.add_parser(
        "session", help="Sign-ins on THIS node: list, revoke one, sign everybody out")
    auth_session_sub = auth_session.add_subparsers(dest="session_action")
    auth_session_list = auth_session_sub.add_parser(
        "list", help="The sessions on this node")
    auth_session_list.add_argument("--user", default="", help="Only this account's")
    auth_session_revoke = auth_session_sub.add_parser(
        "revoke", help="Revoke one session by id")
    auth_session_revoke.add_argument(
        "session_id", help="The session id from: ainode auth session list")
    auth_session_clear = auth_session_sub.add_parser(
        "clear", help="Sign everybody out of this node")
    auth_session_clear.add_argument(
        "--user", default="", help="Sign out only this account")
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

    # tls
    tls_parser = subparsers.add_parser(
        "tls", help="Serve HTTPS on a second port (HTTP keeps its own)")
    tls_sub = tls_parser.add_subparsers(dest="tls_action")
    tls_enable = tls_sub.add_parser(
        "enable",
        help="Turn TLS on, generating a self-signed certificate when none is given",
    )
    tls_enable.add_argument("--cert", metavar="PATH",
                            help="Use this certificate (copied into <AINODE_HOME>/tls/)")
    tls_enable.add_argument("--key", metavar="PATH",
                            help="Use this private key (copied into <AINODE_HOME>/tls/)")
    tls_enable.add_argument(
        "--tailscale", action="store_true",
        help="Get a real Let's Encrypt certificate for this node's MagicDNS name "
             "via `tailscale cert` (what a Mac client with App Transport Security "
             "needs)",
    )
    tls_enable.add_argument("--port", type=int,
                            help="TLS port (default 3443; must not be the HTTP port)")
    tls_sub.add_parser("disable", help="Stop serving HTTPS (the pair is kept)")
    tls_sub.add_parser("status", help="Show TLS state, certificate and expiry")
    tls_renew = tls_sub.add_parser(
        "renew", help="Whether the tailnet certificate is due for replacement")
    tls_renew.add_argument(
        "--check", action="store_true",
        help="Print the decision as key=value lines and exit 0 when a renewal is "
             "due, 10 when it is not (what the host's renewal timer reads)")
    tls_parser.set_defaults(func=cmd_tls)

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
