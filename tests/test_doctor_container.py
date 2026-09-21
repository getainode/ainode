"""The doctor inside the container, and the node setting behind a hung API.

Two issues, one file, because both are about the doctor telling the truth about
where it is standing:

* **#225.** ``docker exec ainode ainode doctor`` is the documented way to run this
  (the installer's host wrapper does exactly that), and from in there the systemd
  unit, the host's docker and which image the container runs are unanswerable. All
  three used to answer WARN, so every node in the fleet carried a permanent yellow
  line about something nobody standing in the container could fix. They are INFO
  naming the host command now, and the wrapper hands the real unit state in.
* **#238.** Persistence mode off on a node with discrete GPUs is what made every
  NVML sample pay a full GPU init, which is what hung castor's whole API. The
  doctor WARNs with the one command that fixes it. The product half of that issue
  (the read off the event loop) is tests/test_nvml_sampling.py.

Plus the two facts the prune list and the prose have to agree on: the Docker Hub
repository is spelled ``argentaios/ainode``, and ``argentos/ainode`` is a local
tag from the old mirror step that still has to be prunable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ainode.cli import doctor as doc
from ainode.cli.doctor import FAIL, INFO, OK, WARN
from ainode.core.config import NodeConfig

INSTALL_SH = Path(__file__).resolve().parents[1] / "scripts" / "install.sh"


def _one(checks):
    assert len(checks) == 1, [c.name for c in checks]
    return checks[0]


# ---------------------------------------------------------------- am I inside?

def test_the_image_env_var_is_the_cheapest_answer():
    assert doc.running_in_container(env={"AINODE_IN_CONTAINER": "1"}) is True


def test_the_docker_marker_file_counts(tmp_path):
    marker = tmp_path / ".dockerenv"
    marker.write_text("")
    assert doc.running_in_container(env={}, cgroup_path=str(tmp_path / "nope"),
                                   dockerenv=str(marker)) is True


@pytest.mark.parametrize("line", [
    "0::/docker/2f3c8a",
    "0::/system.slice/containerd.service",
    "11:memory:/kubepods/besteffort/podabc",
])
def test_pid_ones_cgroup_naming_a_runtime_counts(tmp_path, line):
    cgroup = tmp_path / "cgroup"
    cgroup.write_text(line + "\n")
    assert doc.running_in_container(env={}, cgroup_path=str(cgroup),
                                   dockerenv=str(tmp_path / "nope")) is True


def test_a_plain_host_is_a_host(tmp_path):
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/init.scope\n")
    assert doc.running_in_container(env={}, cgroup_path=str(cgroup),
                                   dockerenv=str(tmp_path / "nope")) is False


def test_nothing_readable_at_all_is_a_host(tmp_path):
    assert doc.running_in_container(env={}, cgroup_path=str(tmp_path / "a"),
                                   dockerenv=str(tmp_path / "b")) is False


def test_the_host_state_comes_from_the_wrappers_env_var():
    assert doc.host_service_state(env={"AINODE_HOST_SERVICE_STATE": " active "}) == "active"
    assert doc.host_service_state(env={}) == ""


# --------------------------------------------------- the service check (#225)

def test_in_the_container_with_no_host_state_it_is_info_not_warn():
    """The permanent WARN itself: 21 OK, 1 WARN on every fleet node, and the WARN
    was "there is no systemd in a container"."""
    check = _one(doc.check_service(in_container=True, host_state=""))
    assert check.status == INFO
    assert "no systemd to ask" in check.detail
    assert "on the host" in check.fix


def test_the_host_state_the_wrapper_passes_in_is_reported_as_the_answer():
    check = _one(doc.check_service(in_container=True, host_state="active"))
    assert check.status == OK
    assert "host wrapper" in check.detail
    assert check.data["state"] == "active"


def test_a_dead_unit_under_a_live_container_is_a_real_warn():
    """Worth saying out loud: the node is serving, and a reboot will not bring it
    back."""
    check = _one(doc.check_service(in_container=True, host_state="failed"))
    assert check.status == WARN
    assert "reboot" in check.detail
    assert "systemctl enable --now" in check.fix


def test_a_unit_mid_restart_is_info():
    check = _one(doc.check_service(in_container=True, host_state="activating"))
    assert check.status == INFO


def test_on_a_host_the_check_is_unchanged(monkeypatch):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (0, "active"))
    check = _one(doc.check_service(in_container=False, host_state=""))
    assert check.status == OK
    assert check.data["in_container"] is False


# ---------------------------------------------------- the docker checks (#225)

def _docker_says(monkeypatch, code, out=""):
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (code, out))


def test_an_unreachable_docker_in_the_container_is_info_naming_both_fixes(monkeypatch):
    _docker_says(monkeypatch, 1, "Cannot connect to the Docker daemon")
    check = _one(doc.check_docker("vllm/vllm-openai:v0.27.1", in_container=True))
    assert check.status == INFO
    assert "docker info" in check.fix
    assert "docker.sock" in check.fix
    assert check.data["reachable"] is False


def test_an_unreachable_docker_on_a_host_is_still_a_fail(monkeypatch):
    _docker_says(monkeypatch, 1, "Cannot connect to the Docker daemon")
    assert _one(doc.check_docker("", in_container=False)).status == FAIL


def test_a_reachable_docker_in_the_container_is_ok(monkeypatch):
    _docker_says(monkeypatch, 0, "27.1.1")
    check = _one(doc.check_docker("", in_container=True))
    assert check.status == OK


def test_the_image_pin_is_info_when_there_is_no_docker_to_ask(tmp_path):
    (tmp_path / "image.env").write_text("AINODE_IMAGE=ghcr.io/getainode/ainode:0.5.29\n")
    checks = doc.check_image_pin(tmp_path, "0.5.29", docker_ok=False, in_container=True)
    pin = [c for c in checks if c.name == "image.pin"]
    assert _one(pin).status == INFO
    assert "docker ps" in _one(pin).fix


def test_the_image_pin_still_warns_on_a_host_with_no_container(tmp_path, monkeypatch):
    (tmp_path / "image.env").write_text("AINODE_IMAGE=ghcr.io/getainode/ainode:0.5.29\n")
    monkeypatch.setattr(doc, "latest_image_tag", lambda: "0.5.29")
    checks = doc.check_image_pin(tmp_path, "0.5.29", docker_ok=False, in_container=False)
    assert [c for c in checks if c.name == "image.pin"][0].status == WARN


def test_the_engine_backend_is_info_when_docker_cannot_be_asked_from_here():
    config = NodeConfig(engine_backend="nvidia")
    check = _one(doc.check_engine_backend(config, "/x/config.json", docker_ok=False,
                                          image_present=None, in_container=True))
    assert check.status == INFO
    assert "on the host" in check.fix


def test_the_engine_backend_still_warns_on_a_host():
    config = NodeConfig(engine_backend="nvidia")
    check = _one(doc.check_engine_backend(config, "/x/config.json", docker_ok=False,
                                          image_present=None, in_container=False))
    assert check.status == WARN


def test_a_containerized_node_with_no_docker_socket_has_no_warn(tmp_path, monkeypatch):
    """The whole point of #225, asserted over the real report: none of the three
    host-only checks may be a WARN when the doctor is standing in the container."""
    monkeypatch.setattr(doc, "running_in_container", lambda *a, **k: True)
    monkeypatch.setattr(doc, "host_service_state", lambda *a, **k: "")
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (1, "no daemon"))
    monkeypatch.setattr(doc, "probe_gpus", lambda: [])
    monkeypatch.setattr(doc, "latest_image_tag", lambda: None)
    monkeypatch.setattr(doc, "tcp_listening", lambda *a, **k: False)
    monkeypatch.setattr(doc, "udp_listeners", lambda: set())
    monkeypatch.setattr(doc, "check_peers", lambda c, version="": [
        doc.Check("cluster.peers", OK, "fake", data={"seen": 0})])

    checks = {c.name: c for c in doc.run_checks(tmp_path, tmp_path / "config.json")}
    for name in ("service.unit", "docker.daemon", "image.pin", "config.engine_backend"):
        assert checks[name].status == INFO, f"{name} is {checks[name].status}"


# ------------------------------------------------ persistence mode (#238)

def _gpu(index, name="Tesla V100-SXM2-32GB", unified=False, persistence=None):
    return {"index": index, "name": name, "memory_total_mb": 32768,
            "memory_free_mb": 32768, "unified_memory": unified,
            "persistence_mode": persistence}


def test_persistence_off_on_discrete_cards_warns_with_the_one_command():
    gpus = [_gpu(0, persistence=False), _gpu(1, persistence=False),
            _gpu(2, persistence=True), _gpu(3, persistence=True)]
    check = _one(doc.check_persistence_mode(gpus))
    assert check.status == WARN
    assert "nvidia-smi -pm 1" in check.fix
    assert check.data["disabled"] == [0, 1]
    assert check.data["discrete"] == 4


def test_persistence_on_everywhere_is_ok():
    gpus = [_gpu(i, persistence=True) for i in range(4)]
    check = _one(doc.check_persistence_mode(gpus))
    assert check.status == OK
    assert check.data["disabled"] == []


def test_a_unified_memory_node_is_info_and_never_a_warn():
    """A GB10 does not pay the init and NVML does not report the setting, so a
    WARN there would be another yellow line nobody can clear."""
    check = _one(doc.check_persistence_mode([_gpu(0, name="NVIDIA GB10", unified=True)]))
    assert check.status == INFO
    assert "unified memory" in check.detail


def test_a_driver_that_will_not_say_is_info():
    check = _one(doc.check_persistence_mode([_gpu(0), _gpu(1)]))
    assert check.status == INFO
    assert check.data["unknown"] == [0, 1]


def test_no_gpu_at_all_is_info():
    check = _one(doc.check_persistence_mode([]))
    assert check.status == INFO


def test_probe_gpus_reports_the_setting(monkeypatch):
    """The seam reads it from NVML, and reports None where NVML refuses."""
    class _Nvml:
        NVML_TEMPERATURE_GPU = 0

        def nvmlInit(self):
            return None

        def nvmlShutdown(self):
            return None

        def nvmlDeviceGetCount(self):
            return 2

        def nvmlDeviceGetHandleByIndex(self, index):
            return index

        def nvmlDeviceGetName(self, handle):
            return "Tesla V100-SXM2-32GB"

        def nvmlDeviceGetMemoryInfo(self, handle):
            return SimpleNamespace(total=32 * 1024 ** 3, free=32 * 1024 ** 3)

        def nvmlDeviceGetPersistenceMode(self, handle):
            if handle == 0:
                return 0
            raise RuntimeError("Not Supported")

    monkeypatch.setitem(__import__("sys").modules, "pynvml", _Nvml())
    found = doc.probe_gpus()
    assert [g["persistence_mode"] for g in found] == [False, None]


def test_the_persistence_check_is_in_the_real_report(tmp_path, monkeypatch):
    """A check nobody calls from run_checks is a check nobody sees."""
    monkeypatch.setattr(doc, "probe_gpus",
                        lambda: [_gpu(0, persistence=False)])
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (0, "27.1.1"))
    monkeypatch.setattr(doc, "latest_image_tag", lambda: None)
    monkeypatch.setattr(doc, "tcp_listening", lambda *a, **k: False)
    monkeypatch.setattr(doc, "udp_listeners", lambda: set())
    monkeypatch.setattr(doc, "check_peers", lambda c, version="": [
        doc.Check("cluster.peers", OK, "fake", data={"seen": 0})])

    checks = {c.name: c for c in doc.run_checks(tmp_path, tmp_path / "config.json")}
    assert checks["gpu.persistence"].status == WARN


def test_the_gpu_probe_runs_once_for_both_gpu_checks(tmp_path, monkeypatch):
    """NVML is the expensive read on a node with persistence mode off, which is
    exactly the node this check exists for. Asking twice per report would be the
    bug in miniature."""
    calls = []

    def _probe():
        calls.append(1)
        return [_gpu(0, persistence=True)]

    monkeypatch.setattr(doc, "probe_gpus", _probe)
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (0, "27.1.1"))
    monkeypatch.setattr(doc, "latest_image_tag", lambda: None)
    monkeypatch.setattr(doc, "tcp_listening", lambda *a, **k: False)
    monkeypatch.setattr(doc, "udp_listeners", lambda: set())
    monkeypatch.setattr(doc, "check_peers", lambda c, version="": [
        doc.Check("cluster.peers", OK, "fake", data={"seen": 0})])

    doc.run_checks(tmp_path, tmp_path / "config.json")
    assert len(calls) == 1


# ------------------------------------------------------------- --fix --peer

def test_fix_with_peer_is_refused_rather_than_silently_dropped(capsys):
    """It used to accept both and apply nothing anywhere."""
    args = SimpleNamespace(json=False, peer="spark3-remote", fix=True)
    with pytest.raises(SystemExit) as exc:
        doc.cmd_doctor(args)
    assert exc.value.code == 2
    said = capsys.readouterr().err
    assert "--fix is not applied over --peer" in said
    assert "spark3-remote" in said


def test_a_peer_report_without_fix_still_runs(monkeypatch):
    monkeypatch.setattr(doc, "peer_checks",
                        lambda peer: ([doc.Check("x", OK, "fine")], None))
    args = SimpleNamespace(json=True, peer="spark3-remote", fix=False)
    with pytest.raises(SystemExit) as exc:
        doc.cmd_doctor(args)
    assert exc.value.code == 0


# -------------------------------------------------- the host wrapper (#225)
#
# The wrapper's `doctor` case reads the unit state on the host and exports it, then
# forwards through `forward_to_container`, the same helper the `tls` case uses. The
# value therefore travels in the environment (docker's `-e NAME` pass-through form)
# and not in the argv, so the fake docker below prints both.

@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestWrapperHandsTheUnitStateIn:
    @pytest.fixture
    def wrapper(self, tmp_path):
        home = tmp_path / "home"
        ainode_home = home / ".ainode"
        sysfs = tmp_path / "sys-class-net"
        home.mkdir(parents=True)
        sysfs.mkdir(parents=True)
        env = dict(os.environ)
        env.update(HOME=str(home), AINODE_HOME=str(ainode_home),
                   AINODE_IMAGE="ghcr.io/getainode/ainode:9.9.9",
                   SYS_CLASS_NET=str(sysfs))
        env.pop("AINODE_PEERS", None)
        env.pop("HF_TOKEN", None)
        proc = subprocess.run(["bash", str(INSTALL_SH), "--dry-run"],
                              capture_output=True, text=True, timeout=120, env=env)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        path = ainode_home / "ainode-wrapper"
        assert path.exists(), "the installer no longer renders a host wrapper"
        return path, tmp_path, env

    def _fakes(self, tmp_path, unit_state, container_up=True):
        """A docker that prints its argv and a systemctl that answers ``unit_state``."""
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        (bindir / "docker").write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "exec" ] && [ "$3" = "true" ]; then exit '
            f'{0 if container_up else 1}; fi\n'
            'echo "docker $*"\n'
            # The state travels in the environment, through the `-e NAME`
            # pass-through form, so print what this process inherited.
            'echo "env AINODE_HOST_SERVICE_STATE=${AINODE_HOST_SERVICE_STATE-unset}"\n')
        (bindir / "systemctl").write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "--user" ]; then exit 1; fi\n'
            f'echo "{unit_state}"\n')
        for name in ("docker", "systemctl"):
            (bindir / name).chmod(0o755)
        return bindir

    def _run(self, wrapper, bindir, env, *args):
        run_env = dict(env)
        run_env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
        return subprocess.run(["bash", str(wrapper), "doctor", *args],
                              capture_output=True, text=True, timeout=60, env=run_env)

    def test_the_active_unit_state_reaches_the_container(self, wrapper):
        path, tmp_path, env = wrapper
        proc = self._run(path, self._fakes(tmp_path, "active"), env, "--json")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "-e AINODE_HOST_SERVICE_STATE" in proc.stdout
        assert "env AINODE_HOST_SERVICE_STATE=active" in proc.stdout
        assert "ainode doctor --json" in proc.stdout

    def test_a_failed_unit_state_reaches_it_too(self, wrapper):
        path, tmp_path, env = wrapper
        proc = self._run(path, self._fakes(tmp_path, "failed"), env)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "env AINODE_HOST_SERVICE_STATE=failed" in proc.stdout

    def test_with_no_container_the_one_shot_run_carries_it(self, wrapper):
        path, tmp_path, env = wrapper
        proc = self._run(path, self._fakes(tmp_path, "active", container_up=False), env)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "docker run" in proc.stdout
        assert "-e AINODE_HOST_SERVICE_STATE" in proc.stdout
        assert "env AINODE_HOST_SERVICE_STATE=active" in proc.stdout

    def test_the_update_path_fails_when_no_version_can_be_resolved(self, wrapper):
        """"exit 0" is what a roll script reads as "this node is updated", and a
        pull of a floating tag that nothing verified is not that."""
        path, tmp_path, env = wrapper
        bindir = self._fakes(tmp_path, "active")
        # No GHCR, no /api/status: nothing can resolve or verify a version.
        (bindir / "curl").write_text("#!/usr/bin/env bash\nexit 1\n")
        # The restart is real in this path, so sudo has to be a pass-through
        # rather than a password prompt on the machine running the suite.
        (bindir / "sudo").write_text('#!/usr/bin/env bash\nexec "$@"\n')
        (bindir / "curl").chmod(0o755)
        (bindir / "sudo").chmod(0o755)
        run_env = dict(env)
        run_env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
        proc = subprocess.run(["bash", str(path), "update"], capture_output=True,
                              text=True, timeout=60, env=run_env)
        assert proc.returncode != 0, proc.stdout + proc.stderr
        assert "nothing was verified" in proc.stderr


# ------------------------------------- the Docker Hub name, in one spelling

def test_the_prune_list_carries_the_docker_hub_repo_that_exists():
    """``argentaios/ainode`` is the repository on Docker Hub (stale at 0.4.7) and
    the spelling the README, CLAUDE.md and scripts/uninstall.sh use.
    ``argentos/ainode`` has no repository there at all, and is in the list only
    because the old mirror step tagged it locally: Spark-1 carried every release
    under it, and a prune list without it leaves those tags holding the bytes
    (#184)."""
    from ainode.core.image_prune import AINODE_IMAGE_REPOS, DOCKER_HUB_REPO

    assert DOCKER_HUB_REPO == "argentaios/ainode"
    assert DOCKER_HUB_REPO in AINODE_IMAGE_REPOS
    assert "ghcr.io/getainode/ainode" in AINODE_IMAGE_REPOS
    assert "argentos/ainode" in AINODE_IMAGE_REPOS
    assert "ainode" in AINODE_IMAGE_REPOS


def test_the_documented_spelling_is_the_one_the_uninstaller_uses():
    from ainode.core.image_prune import DOCKER_HUB_REPO

    uninstall = (Path(__file__).resolve().parents[1] / "scripts" / "uninstall.sh").read_text()
    assert DOCKER_HUB_REPO in uninstall
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert DOCKER_HUB_REPO in readme


def test_the_readme_does_not_quote_a_check_count():
    """It said 21, the report emits more than that and the number moves with
    every release."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "21 checks" not in readme


def test_json_is_still_json_with_the_new_check(tmp_path, monkeypatch):
    monkeypatch.setattr(doc, "probe_gpus", lambda: [_gpu(0, persistence=False)])
    monkeypatch.setattr(doc, "run_command", lambda argv, timeout=10.0: (0, "27.1.1"))
    monkeypatch.setattr(doc, "latest_image_tag", lambda: None)
    monkeypatch.setattr(doc, "tcp_listening", lambda *a, **k: False)
    monkeypatch.setattr(doc, "udp_listeners", lambda: set())
    monkeypatch.setattr(doc, "check_peers", lambda c, version="": [
        doc.Check("cluster.peers", OK, "fake", data={"seen": 0})])

    payload = doc.report_payload(doc.run_checks(tmp_path, tmp_path / "config.json"))
    reparsed = json.loads(json.dumps(payload))
    names = [c["name"] for c in reparsed["checks"]]
    assert "gpu.persistence" in names
