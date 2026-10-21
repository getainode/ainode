"""Host capabilities require operator authority and bounded filesystem access."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.auth.accounts import may_use_server_paths
from ainode.datasets.api_routes import setup_dataset_routes
from ainode.datasets.manager import DatasetManager
from ainode.engine.launch_validation import request_image, request_volume, safe_host_path
from ainode.models.api_routes import catalog_recipe, parse_launch_overrides


@pytest.mark.parametrize("identity,allowed", [
    ({}, False),
    ({"authenticated": True, "user": "reader", "user_role": "member"}, False),
    ({"authenticated": True, "user": "owner", "user_role": "admin"}, True),
    ({"authenticated": True, "api_key_id": "operator"}, True),
])
def test_launch_roles(identity, allowed, tmp_path):
    class Request(dict):
        app = {"auth_config": SimpleNamespace(enabled=True)}
    request = Request(identity)
    for body in ({"engine_image": "registry.example/team/engine:v1"},
                 {"extra_volumes": [f"{tmp_path}:/data"]}):
        overrides, error = parse_launch_overrides(
            body, authenticated=identity.get("authenticated", False),
            privileged=may_use_server_paths(request),
            config=SimpleNamespace(models_dir=str(tmp_path)))
        assert (error is None) == allowed
        assert bool(overrides) == allowed


@pytest.mark.parametrize("image", ["--privileged", "evil\nimage", "evil\timage", "", None,
                                    "evil;command", "repo:tag --volume=/:/host", "UPPER/repo"])
def test_hostile_images(image):
    with pytest.raises(ValueError):
        request_image(image)


@pytest.mark.parametrize("image", ["vllm/vllm-openai:v0.27.1", "localhost:5000/team/engine:v1",
                                    "registry.example/repo@sha256:" + "a" * 64])
def test_valid_images(image):
    assert request_image(image) == image


@pytest.mark.parametrize("host", ["/", "/tmp/..", "/etc", "/etc/passwd", "/root",
                                   "/var/run/docker.sock", "/run", "/var/run", "/proc", "/sys", "/dev", "/home/operator/.ssh/id_rsa"])
def test_sensitive_mounts(host):
    with pytest.raises(ValueError):
        request_volume(f"{host}:/data")


def test_mount_symlink_and_secrets(tmp_path, monkeypatch):
    config = tmp_path / "secret-home"
    config.mkdir()
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", config)
    for target in (Path("/etc"), config, Path("/")):
        link = tmp_path / "alias"
        link.symlink_to(target)
        with pytest.raises(ValueError):
            safe_host_path(str(link))
        link.unlink()
    data = tmp_path / "data"
    data.mkdir()
    policy = SimpleNamespace(models_dir=str(data))
    assert request_volume(f"{data}:/data", config=policy).endswith(":ro")
    assert request_volume(f"{data}:/data:rw", config=policy).endswith(":rw")


def test_catalog_not_reparsed():
    from ainode.models.registry import CURATED_CLUSTER_MODELS
    model = next(m.hf_repo for m in CURATED_CLUSTER_MODELS.values() if m.engine_image)
    recipe = catalog_recipe(model)
    assert recipe["engine_image"]
    overrides, error = parse_launch_overrides({}, privileged=False)
    assert error is None and overrides == {}
    assert catalog_recipe(model) == recipe


def test_dataset_containment_and_preview(tmp_path):
    root = tmp_path / "datasets"
    mgr = DatasetManager(root)
    outside = tmp_path / "config.json"
    outside.write_text('{"cluster_secret":"sentinel"}')
    link = root / "escape.json"
    link.symlink_to(outside)
    for path in (outside, link):
        with pytest.raises(ValueError, match="approved dataset root"):
            mgr.add_local(str(path))
    data = root / "data.jsonl"
    data.write_text('{"text":"hello"}\n')
    ds = mgr.add_local(str(data))
    assert mgr.preview(ds.id)["samples"] == [{"text": "hello"}]
    data.unlink()
    data.symlink_to(outside)
    with pytest.raises(ValueError):
        mgr.preview(ds.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("identity,enabled,status", [
    ({}, True, 403),
    ({"authenticated": True, "user": "member", "user_role": "member"}, True, 403),
    ({"authenticated": True, "user": "admin", "user_role": "admin"}, True, 201),
    ({"authenticated": True}, True, 201),
    ({}, False, 201),
])
async def test_local_dataset_http_roles_and_paths(tmp_path, identity, enabled, status):
    @web.middleware
    async def identify(request, handler):
        request.update(identity)
        return await handler(request)
    app = web.Application(middlewares=[identify])
    app["auth_config"] = SimpleNamespace(enabled=enabled)
    mgr = DatasetManager(tmp_path)
    setup_dataset_routes(app, mgr)
    data = tmp_path / "normal.jsonl"
    data.write_text('{"text":"hello"}\n')
    async with TestClient(TestServer(app), headers={"X-AINode-Client": "dashboard"}) as client:
        response = await client.post("/api/datasets", json={"source": "local", "path": str(data)})
        assert response.status == status
        payload = await response.json()
        assert str(tmp_path) not in str(payload)
        if status == 201:
            for url in ("/api/datasets", f"/api/datasets/{payload['id']}"):
                response = await client.get(url)
                assert str(tmp_path) not in await response.text()
            response = await client.post("/api/datasets", json={
                "source": "local", "path": str(tmp_path.parent / 'config.json')})
            assert response.status == 400
            assert str(tmp_path.parent) not in await response.text()


def test_backend_rechecks_mount_symlink(tmp_path):
    from ainode.core.config import NodeConfig
    from ainode.engine.backends.nvidia import NvidiaBackend
    link = tmp_path / "mount"
    link.symlink_to("/etc")
    backend = NvidiaBackend(NodeConfig(extra_volumes=[f"{link}:/data"]))
    with pytest.raises(ValueError):
        backend._volume_args()


@pytest.mark.parametrize("filename", ["config.json", "auth.json", "users.json", "secrets.json"])
def test_config_denied_even_with_broad_dataset_root(tmp_path, monkeypatch, filename):
    from ainode.datasets import manager
    monkeypatch.setattr(manager, "AINODE_HOME", tmp_path)
    secret = tmp_path / filename
    secret.write_text('{"cluster_secret":"sentinel"}')
    mgr = DatasetManager(tmp_path)
    with pytest.raises(ValueError):
        mgr.add_local(str(secret))


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["load", "sharding"])
@pytest.mark.parametrize("identity", [{}, {"authenticated": True, "user": "member",
                                          "user_role": "member"}])
async def test_launch_handlers_refuse_before_side_effects(route, identity):
    from unittest.mock import Mock
    from ainode.models.api_routes import handle_model_load
    from ainode.engine.sharding_routes import handle_sharding_launch

    class Request(dict):
        app = {"auth_config": SimpleNamespace(enabled=True), "engine": Mock()}

        async def json(self):
            return {"model": "example/model", "engine_image": "example/image:1"}
    request = Request(identity)
    response = await (handle_model_load(request) if route == "load"
                      else handle_sharding_launch(request))
    assert response.status == 400
    assert "admin or operator" in response.text
    request.app["engine"].stop.assert_not_called()


def test_mount_of_home_with_ssh_is_denied(tmp_path):
    (tmp_path / ".ssh").mkdir()
    with pytest.raises(ValueError):
        request_volume(f"{tmp_path}:/home")
