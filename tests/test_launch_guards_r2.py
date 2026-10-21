"""Original caller authority, local storage policy, and peer namespace checks."""
import json
import shlex
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ainode.auth.accounts import may_control_host, may_supply_launch_fields
from ainode.core.config import NodeConfig
from ainode.core.storage_policy import sanitize_storage, validate_storage_root
from ainode.engine.launch_validation import request_volume
from ainode.models.api_routes import parse_launch_overrides


class Request(dict):
    def __init__(self, body=None, *, identity=None, remote="198.51.100.3", enabled=True):
        super().__init__(identity or {})
        self.body = body or {}
        self.remote = remote
        self.headers = {}
        self.app = {"auth_config": SimpleNamespace(enabled=enabled), "config": NodeConfig()}

    async def json(self):
        return self.body


@pytest.mark.parametrize("remote,allowed", [("127.0.0.1", True), ("::1", True),
                                            ("198.51.100.3", False), (None, False)])
def test_open_host_authority(remote, allowed):
    request = Request(remote=remote, enabled=False)
    request.headers["X-Forwarded-For"] = "127.0.0.1"
    assert may_control_host(request) is allowed


@pytest.mark.parametrize("field,value", [("engine_image", "example/image:1"),
    ("extra_volumes", ["/usr:/host-usr:rw"]), ("trust_remote_code", True),
    ("extra_vllm_args", ["--trust-remote-code"]), ("extra_env", {"PYTHONPATH": "/host"})])
@pytest.mark.parametrize("node", ["local", "remote"])
@pytest.mark.asyncio
async def test_cluster_checks_original_member(field, value, node):
    from ainode.api.server import _cluster_dispatch
    request = Request({"node_id": node, "model": "example/model", field: value},
                      identity={"authenticated": True, "user": "member", "user_role": "member"})
    request.app["config"].node_id = "local"
    request.app["client_session"] = Mock()
    response = await _cluster_dispatch(request, "/api/models/load")
    assert response.status == 403
    request.app["client_session"].post.assert_not_called()


def test_fleet_transport_is_not_original_caller():
    request = Request(identity={"authenticated": True, "api_key_id": "fleet"})
    assert not may_supply_launch_fields(request)
    request.headers["X-AINode-Launch-Authorized"] = "true"
    assert may_supply_launch_fields(request)
    ordinary = Request(identity={"authenticated": True, "user": "member", "user_role": "member"})
    ordinary.headers.update(request.headers)
    assert not may_supply_launch_fields(ordinary)


@pytest.mark.parametrize("field", ["engine_image", "extra_volumes", "trust_remote_code", "extra_env", "extra_vllm_args"])
def test_raw_host_controls_require_authority(field):
    _, error = parse_launch_overrides({field: True}, authenticated=True, privileged=False)
    assert "admin or operator" in error


@pytest.mark.parametrize("path", ["/", "/etc", "/usr", "/var", "/root", "/proc", "/sys", "/dev", "/boot", "/home"])
def test_system_roots(path):
    with pytest.raises(ValueError):
        validate_storage_root(path)


@pytest.mark.parametrize("remote", ["198.51.100.3", "127.0.0.1"])
@pytest.mark.asyncio
async def test_open_patch_root(remote, tmp_path):
    from ainode.api.server import handle_patch_config
    root = tmp_path / "data"
    request = Request({"datasets_dir": "/", "models_dir": str(root)}, remote=remote, enabled=False)
    request.app["config"].save = Mock()
    response = await handle_patch_config(request)
    data = json.loads(response.text)
    assert "datasets_dir" in data["rejected"]
    assert ("models_dir" in data["applied"]) == (remote == "127.0.0.1")
    assert request.app["config"].datasets_dir is None



@pytest.mark.parametrize("remote", ["198.51.100.3", "127.0.0.1"])
@pytest.mark.asyncio
async def test_open_patch_identity_fields(remote):
    """master_address is where a worker sends its fleet key: only host authority sets it."""
    from ainode.api.server import handle_patch_config
    request = Request({"master_address": "198.51.100.66:8000", "node_name": "Renamed"}, remote=remote, enabled=False)
    request.app["config"].save = Mock()
    before = request.app["config"].master_address
    response = await handle_patch_config(request)
    data = json.loads(response.text)
    local = remote == "127.0.0.1"
    assert ("master_address" in data["applied"]) == local and ("node_name" in data["applied"]) == local
    if not local:
        assert request.app["config"].master_address == before


def test_bad_persisted_root_falls_back(tmp_path, monkeypatch, caplog):
    from ainode.core import config
    file = tmp_path / "config.json"
    file.write_text(json.dumps({"datasets_dir": "/", "models_dir": "/usr", "training_dir": "/etc"}))
    monkeypatch.setattr(config, "CONFIG_FILE", file)
    loaded = NodeConfig.load()
    assert loaded.datasets_dir is None and loaded.models_dir == NodeConfig().models_dir
    assert loaded.training_dir is None
    assert "using the default storage root" in caplog.text
    loaded.hf_cache_dir = "/proc"
    sanitize_storage(loaded)
    assert loaded.hf_cache_dir is None


@pytest.mark.parametrize("source", ["/usr", "/boot"])
def test_system_mounts(source, tmp_path):
    with pytest.raises(ValueError):
        request_volume(f"{source}:/host:rw", config=NodeConfig(models_dir=str(tmp_path)))


def test_custom_socket_and_regular_data(tmp_path):
    policy = NodeConfig(models_dir=str(tmp_path))
    source = tmp_path / "engine-control"
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(source))
        with pytest.raises(ValueError):
            request_volume(f"{source}:/control:rw", config=policy)
    source.unlink()
    source.write_text("data")
    assert request_volume(f"{source}:/data", config=policy).endswith(":ro")
    assert request_volume(f"{source}:/data:rw", config=policy).endswith(":rw")


def test_peer_rechecks_its_own_symlink_and_socket(tmp_path, monkeypatch):
    from ainode.engine.backends.nvidia import NvidiaBackend
    head = tmp_path / "head"
    peer = tmp_path / "peer"
    head.mkdir()
    peer.mkdir()
    source = peer / "mount"
    source.mkdir()
    backend = NvidiaBackend(NodeConfig(models_dir=str(head), extra_volumes=[f"{source}:/data:rw"]))
    guard = backend._peer_mount_guard(str(peer))
    monkeypatch.setenv("AINODE_HOME", str(tmp_path / "node-home"))
    assert subprocess.run(shlex.split(guard), capture_output=True).returncode == 0
    source.rmdir()
    source.symlink_to("/etc")
    assert subprocess.run(shlex.split(guard), capture_output=True).returncode != 0
    source.unlink()
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(source))
        assert subprocess.run(shlex.split(guard), capture_output=True).returncode != 0


@pytest.mark.parametrize("handler_name", ["handle_engine_update", "handle_cluster_update_all"])
@pytest.mark.asyncio
async def test_updates_refuse_before_work(handler_name, monkeypatch):
    from ainode.api import server
    monkeypatch.setattr(server, "_fetch_latest_ghcr_tag", lambda: None)
    request = Request(enabled=False)
    request.json = AsyncMock(side_effect=AssertionError("must reject before reading or updating"))
    assert (await getattr(server, handler_name)(request)).status == 403


def test_backend_rechecks_type_after_parse(tmp_path):
    from ainode.engine.backends.nvidia import NvidiaBackend
    source = tmp_path / "data"
    source.write_text("ordinary data")
    config = NodeConfig(models_dir=str(tmp_path))
    spec = request_volume(f"{source}:/data", config=config)
    source.unlink()
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(source))
        config.extra_volumes = [spec]
        with pytest.raises(ValueError):
            NvidiaBackend(config)._volume_args()


def test_ssh_command_checks_peer_before_docker(tmp_path, monkeypatch):
    from ainode.engine.backends.nvidia import NvidiaBackend
    import ainode.engine.backends.nvidia as module
    source = tmp_path / "data"
    source.mkdir()
    backend = NvidiaBackend(NodeConfig(models_dir=str(tmp_path), extra_volumes=[f"{source}:/data"]))
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(module.subprocess, "run", run)
    backend._ssh_run_worker_container(peer_ip="198.51.100.2", worker_name="worker",
                                      docker_cmd=["docker", "run", "image"],
                                      peer_hf_cache=str(tmp_path), peer_models_dir=str(tmp_path))
    command = run.call_args.args[0][-1]
    assert "python3 -c" in command
    assert command.index("python3 -c") < command.index("&& docker run")


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [{"authenticated": True, "user": "owner", "user_role": "admin"},
                                      {"authenticated": True, "api_key_id": "operator"}])
async def test_remote_authorized_launch_carries_approval(identity):
    from ainode.api.server import _cluster_dispatch
    request = Request({"node_id": "remote", "model": "example/model", "engine_image": "example/image:1"},
                      identity=identity)
    request.app["config"].node_id = "local"
    request.app["config"].cluster_secret = "test-fleet-secret"
    peer = SimpleNamespace(web_port=3000, ip="198.51.100.2")
    request.app["cluster_state"] = Mock(get_node=Mock(return_value=peer))
    response = SimpleNamespace(status=200, headers={}, read=AsyncMock(return_value=b"{}"))
    context = Mock(__aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock())
    session = Mock(post=Mock(return_value=context))
    request.app["client_session"] = session
    from unittest.mock import patch
    with patch("ainode.api.server.best_host", return_value=peer.ip):
        result = await _cluster_dispatch(request, "/api/models/load")
    assert result.status == 200
    assert session.post.call_args.kwargs["headers"]["X-AINode-Launch-Authorized"] == "true"


def test_mount_outside_approved_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("data")
    with pytest.raises(ValueError):
        request_volume(f"{outside}:/data:rw", config=NodeConfig(models_dir=str(root)))


@pytest.mark.parametrize("container_config", [False, True])
def test_peer_default_model_store(tmp_path, monkeypatch, container_config):
    from ainode.engine.backends.nvidia import NvidiaBackend
    home = tmp_path / "node"
    source = home / "models" / "data"
    source.mkdir(parents=True)
    if container_config:
        (home / "config.json").write_text(json.dumps({"models_dir": "/root/.ainode/models"}))
    monkeypatch.setenv("AINODE_HOME", str(home))
    backend = NvidiaBackend(NodeConfig(extra_volumes=[f"{source}:/data"]))
    guard = backend._peer_mount_guard(str(tmp_path / "peer-models"))
    assert subprocess.run(shlex.split(guard), capture_output=True).returncode == 0


def test_storage_root_cannot_contain_node_secret_home(tmp_path, monkeypatch):
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path / "node-home")
    with pytest.raises(ValueError):
        validate_storage_root(str(tmp_path))
    monkeypatch.setenv("AINODE_HOST_HOME", str(tmp_path / "host-home"))
    with pytest.raises(ValueError):
        validate_storage_root(str(tmp_path / "host-home"))


@pytest.mark.asyncio
async def test_operator_sets_mount_roots_and_member_cannot(tmp_path):
    from ainode.api.server import handle_patch_config
    root = tmp_path / "cache"
    root.mkdir()
    for role in ("member", "admin"):
        request = Request({"allowed_mount_roots": [str(root)]}, identity={
            "authenticated": True, "user": role, "user_role": role})
        request.app["config"].save = Mock()
        result = json.loads((await handle_patch_config(request)).text)
        assert ("allowed_mount_roots" in result["applied"]) == (role == "admin")
    assert request_volume(f"{root}:/cache:rw", config=request.app["config"]).endswith(":rw")


def test_container_mount_requires_visible_direct_source(tmp_path, monkeypatch):
    from ainode.engine.backends.nvidia import NvidiaBackend
    home = tmp_path / "node-home"
    models = home / "models"
    models.mkdir(parents=True)
    data = models / "data"
    data.mkdir()
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", home)
    monkeypatch.setenv("AINODE_HOST_HOME", str(tmp_path / "host-node"))
    backend = NvidiaBackend(NodeConfig(models_dir=str(models), extra_volumes=[f"{data}:/data"]))
    assert str(tmp_path / "host-node" / "models" / "data") in backend._volume_args()[1]
    alias = models / "alias"
    alias.symlink_to(data)
    backend.config.extra_volumes = [f"{alias}:/data"]
    with pytest.raises(ValueError):
        backend._volume_args()
    outside = tmp_path / "outside"
    outside.mkdir()
    backend.config.allowed_mount_roots = [str(outside)]
    backend.config.extra_volumes = [f"{outside}:/data"]
    with pytest.raises(ValueError):
        backend._volume_args()
