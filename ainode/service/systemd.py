"""Systemd service management for AINode.

As of v0.4.0 the unit runs the AINode container via ``docker run`` rather
than a host-venv ``ainode start``. The unit template mounts ``~/.ainode``
(config + model cache), the host docker socket (so the head container can
launch peer workers via eugr's launcher), and ``~/.ssh`` (read-only, for
passwordless SSH from head to workers).

Provides install, uninstall, enable, disable, start, stop, restart,
status, and is_installed operations.
"""

import os
import shutil
import subprocess
from pathlib import Path


SERVICE_NAME = "ainode.service"

SYSTEM_UNIT_DIR = Path("/etc/systemd/system")
USER_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"

# Image tag bumped when we cut a release and republish to GHCR.
AINODE_IMAGE_TAG = "0.4.0"
AINODE_IMAGE = f"ghcr.io/getainode/ainode:{AINODE_IMAGE_TAG}"

UNIT_FILE_TEMPLATE = """\
[Unit]
Description=AINode — Local AI inference platform
Documentation=https://ainode.dev
After=network.target docker.service nvidia-persistenced.service
Wants=docker.service nvidia-persistenced.service
Requires=docker.service

[Service]
Type=simple
# docker run in foreground so systemd tracks the container lifecycle.
ExecStartPre=-/usr/bin/docker rm -f ainode
ExecStart={exec_start}
ExecStop=/usr/bin/docker stop -t 30 ainode
Restart=on-failure
RestartSec=10
TimeoutStartSec=600
TimeoutStopSec=45

# GPU environment (docker run also passes --gpus all).
Environment=NVIDIA_VISIBLE_DEVICES=all
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=AINODE_HOME={ainode_home}

# No host-filesystem-protection stanza intentionally — see ainode/service/
# systemd.py source for rationale (docker socket + peer worker launch).

[Install]
WantedBy={wanted_by}
"""

DOCKER_RUN_CMD = (
    "/usr/bin/docker run --rm --name ainode"
    " --network=host --gpus all --ipc=host --shm-size=64g"
    " -v {ainode_home}:/root/.ainode"
    " -v /var/run/docker.sock:/var/run/docker.sock"
    " -v {home}/.ssh:/root/.ssh:ro"
    " {image}"
)


def _ainode_bin() -> str:
    """Return the path to the ainode executable."""
    return shutil.which("ainode") or "ainode"


def _ainode_home() -> str:
    """Return the AINODE_HOME directory."""
    return os.environ.get("AINODE_HOME", str(Path.home() / ".ainode"))


def _unit_dir(user_mode: bool) -> Path:
    if user_mode:
        return USER_UNIT_DIR
    return SYSTEM_UNIT_DIR


def _unit_path(user_mode: bool) -> Path:
    return _unit_dir(user_mode) / SERVICE_NAME


def _systemctl(args: list[str], user_mode: bool = False, capture: bool = False):
    """Run a systemctl command."""
    cmd = ["systemctl"]
    if user_mode:
        cmd.append("--user")
    cmd.extend(args)
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result
    subprocess.run(cmd, check=True, timeout=30)


def generate_unit_file(user_mode: bool = False) -> str:
    """Generate the systemd unit file content.

    Renders ``DOCKER_RUN_CMD`` as the ExecStart so the service is literally
    ``docker run ... ghcr.io/getainode/ainode:<ver>``. No more host venv.
    """
    wanted_by = "default.target" if user_mode else "multi-user.target"
    ainode_home = _ainode_home()
    home = str(Path.home())
    exec_start = DOCKER_RUN_CMD.format(
        ainode_home=ainode_home,
        home=home,
        image=AINODE_IMAGE,
    )
    return UNIT_FILE_TEMPLATE.format(
        exec_start=exec_start,
        ainode_home=ainode_home,
        home=home,
        wanted_by=wanted_by,
    )


def is_installed(user_mode: bool = False) -> bool:
    """Check if the AINode service unit file exists."""
    return _unit_path(user_mode).exists()


def install_service(user_mode: bool = False, reload: bool = True) -> None:
    """Write the unit file and optionally reload the systemd daemon.

    When called from inside a container (e.g. during install.sh), pass
    ``reload=False`` — the container has no systemd bus, so daemon-reload
    must be run by the host after the docker run completes.
    """
    unit_dir = _unit_dir(user_mode)
    unit_dir.mkdir(parents=True, exist_ok=True)

    unit_path = _unit_path(user_mode)
    unit_content = generate_unit_file(user_mode=user_mode)
    unit_path.write_text(unit_content)

    if reload:
        _systemctl(["daemon-reload"], user_mode=user_mode)


def uninstall_service(user_mode: bool = False) -> None:
    """Stop, disable, and remove the unit file."""
    if not is_installed(user_mode):
        return

    # Best-effort stop and disable before removal
    try:
        stop_service(user_mode=user_mode)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        disable_service(user_mode=user_mode)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    _unit_path(user_mode).unlink(missing_ok=True)
    _systemctl(["daemon-reload"], user_mode=user_mode)


def enable_service(user_mode: bool = False) -> None:
    """Enable AINode to start on boot."""
    _systemctl(["enable", SERVICE_NAME], user_mode=user_mode)


def disable_service(user_mode: bool = False) -> None:
    """Disable AINode from starting on boot."""
    _systemctl(["disable", SERVICE_NAME], user_mode=user_mode)


def start_service(user_mode: bool = False) -> None:
    """Start the AINode service."""
    _systemctl(["start", SERVICE_NAME], user_mode=user_mode)


def stop_service(user_mode: bool = False) -> None:
    """Stop the AINode service."""
    _systemctl(["stop", SERVICE_NAME], user_mode=user_mode)


def restart_service(user_mode: bool = False) -> None:
    """Restart the AINode service."""
    _systemctl(["restart", SERVICE_NAME], user_mode=user_mode)


def status_service(user_mode: bool = False) -> dict:
    """Return service status and recent journal lines.

    Returns dict with keys: state, enabled, journal_lines.
    """
    result = {"state": "unknown", "enabled": False, "journal_lines": []}

    # Active state
    r = _systemctl(["is-active", SERVICE_NAME], user_mode=user_mode, capture=True)
    result["state"] = r.stdout.strip() or "unknown"

    # Enabled state
    r = _systemctl(["is-enabled", SERVICE_NAME], user_mode=user_mode, capture=True)
    result["enabled"] = r.stdout.strip() == "enabled"

    # Recent journal lines
    result["journal_lines"] = get_journal_lines(user_mode=user_mode, lines=20)

    return result


def get_journal_lines(
    user_mode: bool = False,
    lines: int = 50,
    follow: bool = False,
) -> list[str]:
    """Fetch recent journal lines for the AINode service."""
    cmd = ["journalctl"]
    if user_mode:
        cmd.append("--user")
    cmd.extend(["-u", SERVICE_NAME, "-n", str(lines), "--no-pager"])
    if follow:
        cmd.append("-f")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines() if r.stdout else []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
