"""A node installed by scripts/install.sh must be able to load a model.

Issue #164. Up to 0.5.25 it could not, and the reasons were spread across four
files that never contradicted each other loudly enough to notice:

* ``NodeConfig.engine_backend`` defaulted to ``"eugr"``, a backend that execs a
  ``vllm`` binary the shipped image does not contain, and nothing anywhere ever
  set ``"nvidia"``, so every model load on a fresh node answered HTTP 500.
* The installer wrote ``gpu_memory_utilization: 0.9``, which the stacked-load
  guard reads as "no room for a second model" on a brand-new node.
* The installer pre-pulled ``nvcr.io/nvidia/vllm`` (~15 GB behind an NGC login),
  an image no code path has ever launched, while the engine image the backend
  does launch was pulled later, by the user's first Launch click.
* ``ainode logs`` tailed the eugr backend's ``vllm.log`` whatever the configured
  backend was, so the documented troubleshooting command read a dead file.
* ``sudo ainode update`` pinned the new image in ``/root/.ainode/image.env``
  while the unit reads the install user's, and reported success.

These tests pin each of those, and they run the REAL installer (``--dry-run``,
which renders config.json, the unit and the host wrapper into ``$AINODE_HOME``
and touches nothing else) rather than a transcription of it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ainode.core.config import DEFAULT_ENGINE_BACKEND, LOGS_DIR, NodeConfig
from ainode.engine.backends import EugrBackend, NvidiaBackend, get_backend

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


# ---------------------------------------------------------------------------
# 1. The config default
# ---------------------------------------------------------------------------

class TestEngineBackendDefault:
    def test_default_is_nvidia(self):
        """The one that decides whether a fresh install can serve anything."""
        assert DEFAULT_ENGINE_BACKEND == "nvidia"
        assert NodeConfig().engine_backend == "nvidia"
        assert isinstance(get_backend(NodeConfig()), NvidiaBackend)

    def test_eugr_is_still_available_when_asked_for_by_name(self):
        config = NodeConfig(engine_backend="eugr")
        assert isinstance(get_backend(config), EugrBackend)

    def test_missing_or_empty_value_lands_on_the_default(self):
        """An old config.json with no engine_backend key, and a blank one.

        Both used to reach a hardcoded ``or "eugr"`` in three separate call
        sites. They now resolve to whatever a fresh NodeConfig would use.
        """
        from_old_file = NodeConfig(**{"node_name": "n"})  # key absent entirely
        assert from_old_file.engine_backend == DEFAULT_ENGINE_BACKEND
        assert isinstance(get_backend(NodeConfig(engine_backend="")), NvidiaBackend)

    def test_unknown_backend_still_raises(self):
        with pytest.raises(ValueError, match="Unknown engine_backend"):
            get_backend(NodeConfig(engine_backend="bogus"))

    def test_eugr_guidance_names_the_config_file(self):
        """The 500 a user gets from the eugr backend must say which file to edit."""
        from ainode.core.config import CONFIG_FILE
        from ainode.engine.backends.eugr import NO_VLLM_MESSAGE

        assert "AINode runs as a container image" in NO_VLLM_MESSAGE
        assert str(CONFIG_FILE) in NO_VLLM_MESSAGE
        assert '"engine_backend": "nvidia"' in NO_VLLM_MESSAGE

    def test_the_named_config_file_is_the_one_on_the_HOST(self, monkeypatch):
        """In the container the config lives at a path the user cannot open.

        The unit bind-mounts <host>/.ainode at /root/.ainode and passes the host
        directory as AINODE_HOST_HOME, so that is the path to print.
        """
        import ainode.engine.backends.eugr as eugr

        monkeypatch.setattr(eugr, "AINODE_HOME", Path("/root/.ainode"))
        monkeypatch.setattr(eugr, "CONFIG_FILE", Path("/root/.ainode/config.json"))
        monkeypatch.setenv("AINODE_HOST_HOME", "/home/jason/.ainode")
        assert eugr._config_file_for_display() == "/home/jason/.ainode/config.json"

        monkeypatch.delenv("AINODE_HOST_HOME")
        assert eugr._config_file_for_display() == "/root/.ainode/config.json"


# ---------------------------------------------------------------------------
# 2. `ainode logs` follows the configured backend's real file
# ---------------------------------------------------------------------------

class TestLogsPathResolution:
    def test_nvidia_solo(self):
        from ainode.cli.main import _engine_log_file

        path = _engine_log_file(NodeConfig(engine_backend="nvidia"))
        assert path == LOGS_DIR / "nvidia-vllm.log"

    def test_nvidia_head_follows_the_distributed_log(self):
        from ainode.cli.main import _engine_log_file

        config = NodeConfig(engine_backend="nvidia", distributed_mode="head")
        assert _engine_log_file(config) == LOGS_DIR / "nvidia-distributed.log"

    def test_eugr_keeps_vllm_log(self):
        from ainode.cli.main import _engine_log_file

        path = _engine_log_file(NodeConfig(engine_backend="eugr"))
        assert path == LOGS_DIR / "vllm.log"

    def test_unknown_backend_falls_back_rather_than_raising(self):
        from ainode.cli.main import _engine_log_file

        assert _engine_log_file(NodeConfig(engine_backend="bogus")) == LOGS_DIR / "vllm.log"

    def test_default_config_does_not_resolve_to_the_dead_file(self):
        """The whole point: `ainode logs` on a fresh node reads a live file."""
        from ainode.cli.main import _engine_log_file

        assert _engine_log_file(NodeConfig()).name == "nvidia-vllm.log"


# ---------------------------------------------------------------------------
# 3. What scripts/install.sh actually writes
# ---------------------------------------------------------------------------

def _render_install(tmp_path: Path, *args: str, env_extra: dict | None = None):
    """Run the real installer in --dry-run against a throwaway HOME.

    --dry-run renders config.json, the systemd unit and the host wrapper into
    $AINODE_HOME and stops: no pulls, no systemd, no sudo, no GPU. AINODE_IMAGE
    is pinned so nothing reaches the network.
    """
    home = tmp_path / "home"
    ainode_home = home / ".ainode"
    home.mkdir(parents=True, exist_ok=True)
    sysfs = tmp_path / "sys-class-net"  # empty: no cluster interface detected
    sysfs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        HOME=str(home),
        AINODE_HOME=str(ainode_home),
        AINODE_IMAGE="ghcr.io/getainode/ainode:9.9.9",
        SYS_CLASS_NET=str(sysfs),
    )
    env.pop("AINODE_PEERS", None)
    env.pop("HF_TOKEN", None)
    env.update(env_extra or {})
    proc = subprocess.run(
        ["bash", str(INSTALL_SH), "--dry-run", *args],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return ainode_home, proc


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestInstallerConfig:
    def test_config_json_says_nvidia_and_leaves_room_to_stack(self, tmp_path):
        ainode_home, _ = _render_install(tmp_path)
        data = json.loads((ainode_home / "config.json").read_text())

        # The two values this whole file exists for.
        assert data["engine_backend"] == "nvidia"
        assert data["gpu_memory_utilization"] == 0.6
        # 0.6 has to leave room for one stacked model under the guard in
        # models/api_routes.py, which refuses a load totalling over 0.90.
        assert data["gpu_memory_utilization"] + 0.3 <= 0.9

    def test_no_model_is_configured_so_boot_launches_nothing(self, tmp_path):
        """The installer's own comment says the user picks a model afterwards.

        Leaving the key out inherited NodeConfig's default, so a fresh node booted
        into a launch of the gated meta-llama/Llama-3.2-3B-Instruct: observed on
        Spark-3 failing with a 401 nobody asked for (#164).
        """
        ainode_home, _ = _render_install(tmp_path)
        data = json.loads((ainode_home / "config.json").read_text())
        assert "model" in data, "an absent key inherits the NodeConfig default"
        assert data["model"] is None

    def test_every_key_written_is_a_real_config_field(self, tmp_path):
        """A typo here is silent: NodeConfig.load drops keys it does not know."""
        ainode_home, _ = _render_install(tmp_path)
        data = json.loads((ainode_home / "config.json").read_text())
        unknown = set(data) - set(NodeConfig.__dataclass_fields__)
        assert not unknown, f"install.sh writes unknown config keys: {unknown}"

    def test_the_rendered_config_loads_as_a_nodeconfig(self, tmp_path, monkeypatch):
        ainode_home, _ = _render_install(tmp_path)
        import ainode.core.config as cfgmod

        monkeypatch.setattr(cfgmod, "CONFIG_FILE", ainode_home / "config.json")
        config = cfgmod.NodeConfig.load()
        assert config.engine_backend == "nvidia"
        assert isinstance(get_backend(config), NvidiaBackend)

    def test_worker_and_master_jobs_still_get_the_backend(self, tmp_path):
        for job, mode in (("worker", "member"), ("master", "head")):
            ainode_home, _ = _render_install(tmp_path / job, "--job", job)
            data = json.loads((ainode_home / "config.json").read_text())
            assert data["distributed_mode"] == mode
            assert data["engine_backend"] == "nvidia"

    def test_installer_pre_pulls_the_image_the_backend_launches(self):
        """Not nvcr.io, and not a second copy of the tag: the code's own value."""
        text = INSTALL_SH.read_text()
        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        # nvcr.io survives only in the comment explaining what it used to do.
        assert not [ln for ln in code if "nvcr.io" in ln], "the unused NGC pre-pull is back"
        assert not [ln for ln in code if "NGC_API_KEY" in ln]
        assert "from ainode.engine.backends.nvidia import NVIDIA_VLLM_IMAGE" in text
        # And that import is the real one, so a rename breaks this test and not
        # only somebody's install.
        from ainode.engine.backends.nvidia import NVIDIA_VLLM_IMAGE

        assert NVIDIA_VLLM_IMAGE
        # The engine image is never spelled out in the installer.
        assert NVIDIA_VLLM_IMAGE not in text

    def test_pre_pull_is_still_skippable(self):
        text = INSTALL_SH.read_text()
        assert 'AINODE_NVIDIA_IMAGE="${AINODE_NVIDIA_IMAGE:-}"' in text
        assert '[ "$AINODE_NVIDIA_IMAGE" = "skip" ]' in text


# ---------------------------------------------------------------------------
# 4. `sudo ainode update` pins the image where the unit reads it
# ---------------------------------------------------------------------------

def _stub_bin(dirpath: Path, name: str, body: str) -> None:
    path = dirpath / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


def _sudo_env(tmp_path: Path, *, sudo_user: str | None, passwd_home: Path | None,
              fake_root_home: Path, wrapper_home: str | None = None) -> dict:
    """Environment for a wrapper run that looks like `sudo ainode update`.

    ``id -u`` is stubbed to 0 and HOME to root's, which is exactly what sudo
    hands the wrapper; ``getent`` answers for the invoking user.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    _stub_bin(bindir, "id", 'echo 0\n')
    _stub_bin(bindir, "docker", 'if [ "$1" = "exec" ]; then exit 1; fi\nexit 0\n')
    _stub_bin(bindir, "systemctl", 'exit 1\n')
    if passwd_home is not None and sudo_user:
        _stub_bin(
            bindir, "getent",
            f'[ "$2" = "{sudo_user}" ] && echo "{sudo_user}:x:1000:1000::{passwd_home}:/bin/bash"\n'
            "exit 0\n",
        )
    else:
        _stub_bin(bindir, "getent", "exit 2\n")
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["HOME"] = str(fake_root_home)
    env.pop("AINODE_HOME", None)
    if sudo_user:
        env["SUDO_USER"] = sudo_user
    else:
        env.pop("SUDO_USER", None)
    if wrapper_home:
        env["AINODE_HOME"] = wrapper_home
    return env


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestSudoUpdateHomeResolution:
    @pytest.fixture
    def wrapper(self, tmp_path):
        ainode_home, _ = _render_install(tmp_path)
        path = ainode_home / "ainode-wrapper"
        assert path.exists(), "the installer no longer renders a host wrapper"
        return path

    def _run(self, wrapper: Path, env: dict):
        return subprocess.run(
            ["bash", str(wrapper), "update", "9.9.9"],
            capture_output=True, text=True, timeout=60, env=env,
        )

    def test_sudo_pins_the_invoking_users_image_env(self, tmp_path, wrapper):
        root_home = tmp_path / "rootfake"
        user_home = tmp_path / "home" / "installer"
        root_home.mkdir(parents=True)
        user_home.mkdir(parents=True)

        proc = self._run(
            wrapper,
            _sudo_env(tmp_path, sudo_user="installer", passwd_home=user_home,
                      fake_root_home=root_home),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        pinned = user_home / ".ainode" / "image.env"
        assert pinned.read_text().strip() == "AINODE_IMAGE=ghcr.io/getainode/ainode:9.9.9"
        # The bug: root's copy, which the unit never reads.
        assert not (root_home / ".ainode" / "image.env").exists()
        # And it says where it wrote, so a wrong guess is visible.
        assert str(pinned) in proc.stdout

    def test_the_unit_file_wins_over_every_guess(self, tmp_path, wrapper):
        """The unit is the only thing that decides which image.env systemd reads."""
        root_home = tmp_path / "rootfake"
        user_home = tmp_path / "home" / "installer"
        unit_home = tmp_path / "somewhere" / "else" / ".ainode"
        (root_home / ".config" / "systemd" / "user").mkdir(parents=True)
        user_home.mkdir(parents=True)
        (root_home / ".config" / "systemd" / "user" / "ainode.service").write_text(
            "[Service]\nEnvironment=AINODE_HOME=%s\n" % unit_home
        )

        proc = self._run(
            wrapper,
            _sudo_env(tmp_path, sudo_user="installer", passwd_home=user_home,
                      fake_root_home=root_home),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (unit_home / "image.env").exists()
        assert not (user_home / ".ainode" / "image.env").exists()

    def test_it_refuses_rather_than_writing_the_wrong_file(self, tmp_path, wrapper):
        """Under sudo, with no unit and an invoking user nothing can resolve.

        The one case where guessing would mean pinning /root/.ainode and
        reporting success, which is the 0.5.25 behaviour this fixes.
        """
        root_home = tmp_path / "rootfake"
        root_home.mkdir(parents=True)

        proc = self._run(
            wrapper,
            _sudo_env(tmp_path, sudo_user="ghostuser", passwd_home=None,
                      fake_root_home=root_home),
        )
        assert proc.returncode != 0
        assert "Cannot tell which .ainode" in proc.stderr
        assert "AINODE_HOME=" in proc.stderr
        assert not (root_home / ".ainode" / "image.env").exists()
        # It refuses BEFORE pulling: nothing is done that it cannot finish.
        assert "Pulling" not in proc.stdout

    def test_explicit_ainode_home_wins(self, tmp_path, wrapper):
        root_home = tmp_path / "rootfake"
        chosen = tmp_path / "chosen"
        root_home.mkdir(parents=True)

        proc = self._run(
            wrapper,
            _sudo_env(tmp_path, sudo_user="installer", passwd_home=root_home,
                      fake_root_home=root_home, wrapper_home=str(chosen)),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (chosen / "image.env").exists()

    def test_a_plain_non_root_update_is_unchanged(self, tmp_path, wrapper):
        """No sudo, no unit: still $HOME/.ainode."""
        home = tmp_path / "plainuser"
        home.mkdir(parents=True)
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        _stub_bin(bindir, "id", "echo 1000\n")
        _stub_bin(bindir, "docker", 'if [ "$1" = "exec" ]; then exit 1; fi\nexit 0\n')
        _stub_bin(bindir, "systemctl", "exit 1\n")
        env = dict(os.environ)
        env["PATH"] = f"{bindir}:{env['PATH']}"
        env["HOME"] = str(home)
        env.pop("AINODE_HOME", None)
        env.pop("SUDO_USER", None)

        proc = self._run(wrapper, env)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (home / ".ainode" / "image.env").exists()
