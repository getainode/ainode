"""Quantized outputs cannot escape the model store or borrow a publishing identity."""

import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
import pytest_asyncio

from ainode.models import api_routes as model_routes, hf_upload, registry
from ainode.training import engine, _run_quant, _run_training
from ainode.training.api_routes import setup_training_routes


BAD_SLUGS = ["../", "..", ".", "/etc", "a/b", "a\\b", "a\x00b", "", " x", "é", "a" * 129, 1, []]
BAD_REPOS = ["../private", "/etc", "a/b/c", "a\\b", "a\x00b", "a--b", "a..b", ".name", "name.git", "", 7]


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr(engine, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(engine, "JOBS_DIR", tmp_path / "training" / "jobs")
    monkeypatch.setitem(engine._DIR_OVERRIDES, "training", None)
    monkeypatch.setitem(engine._DIR_OVERRIDES, "datasets", None)
    monkeypatch.setattr(engine, "_image_present", lambda image: False)
    monkeypatch.setattr(hf_upload, "_whoami", MagicMock(return_value={"name": "owner"}))
    monkeypatch.setenv("AINODE_HOST_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


@pytest_asyncio.fixture
async def client(monkeypatch):
    @web.middleware
    async def caller(request, handler):
        request._cache["remote"] = request.app["caller"].remote
        request["authenticated"] = request.app["caller"].authenticated
        request["user"] = request.app["caller"].user
        request["user_role"] = request.app["caller"].role
        return await handler(request)

    app = web.Application(middlewares=[caller])
    app["caller"] = SimpleNamespace(remote="198.51.100.9", authenticated=False, user="", role="")
    app["auth_config"] = SimpleNamespace(enabled=False)
    app["config"] = SimpleNamespace(hf_token="node-token", model=None)
    app["secrets_manager"] = {"huggingface_write_token": "stored-write-token"}
    manager = engine.TrainingManager()
    monkeypatch.setattr(manager, "start_next", AsyncMock())
    setup_training_routes(app, manager)
    async with TestClient(TestServer(app)) as result:
        yield result


def quant_config(**values):
    return engine.TrainingConfig(base_model="org/model", method="quantize", scheme="awq", **values)


@pytest.mark.asyncio
@pytest.mark.parametrize("slug", BAD_SLUGS)
async def test_submit_refuses_unsafe_slug_before_token_lookup_or_queue(client, slug):
    response = await client.post("/api/training/jobs", json={
        "base_model": "org/model", "method": "quantize", "scheme": "awq",
        "out_slug": slug, "push_to_hf": True, "hf_token": "caller-token",
    })
    assert response.status == 400
    assert "out_slug" in (await response.json())["error"]
    assert client.app["training_manager"].list_jobs() == []
    hf_upload._whoami.assert_not_called()


@pytest.mark.parametrize("slug", BAD_SLUGS)
def test_persisted_config_validation_refuses_unsafe_slug(slug):
    config = engine.TrainingConfig.from_dict(quant_config(out_slug=slug).to_dict())
    assert any("out_slug" in error for error in config.validate())


@pytest.mark.parametrize("slug", BAD_SLUGS)
def test_quantize_command_rechecks_slug(slug):
    job = engine.TrainingJob(quant_config(out_slug=slug))
    with pytest.raises(RuntimeError, match="out_slug"):
        job._build_quant_command(job._job_dir / "config.json")


@pytest.mark.parametrize("slug", BAD_SLUGS)
def test_container_output_path_rechecks_slug(tmp_path, slug):
    with pytest.raises(ValueError, match="out_slug"):
        _run_quant.output_directory(quant_config(out_slug=slug).to_dict(), tmp_path / "models")


@pytest.mark.asyncio
@pytest.mark.parametrize("repo", BAD_REPOS)
async def test_submit_refuses_invalid_hf_repo_before_token_lookup(client, repo):
    response = await client.post("/api/training/jobs", json={
        "base_model": "m", "method": "quantize", "scheme": "awq", "out_slug": "m-awq",
        "hf_repo": repo, "push_to_hf": True, "hf_token": "caller-token",
    })
    assert response.status == 400
    assert "hf_repo" in (await response.json())["error"]
    hf_upload._whoami.assert_not_called()
    assert any("hf_repo" in error for error in quant_config(hf_repo=repo).validate())


@pytest.mark.asyncio
@pytest.mark.parametrize("slug, expected", [(None, "org--model-awq"), ("model-awq", "model-awq")])
async def test_open_quantize_keeps_explicit_and_default_slugs(client, tmp_path, slug, expected):
    body = {"base_model": "org/model", "method": "quantize", "scheme": "awq"}
    if slug is not None:
        body["out_slug"] = slug
    response = await client.post("/api/training/jobs", json=body)
    assert response.status == 201
    config = (await response.json())["config"]
    assert config["out_slug"] == expected
    assert _run_quant.output_directory(config, tmp_path / "models") == tmp_path / "models" / expected


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["open-remote", "loopback", "member", "operator", "admin"])
@pytest.mark.parametrize("own_token", [False, True])
async def test_push_identity_is_authorized_on_original_submit(client, authority, own_token):
    caller = client.app["caller"]
    if authority == "loopback":
        caller.remote = "127.0.0.1"
    elif authority in ("member", "operator", "admin"):
        client.app["auth_config"].enabled = True
        caller.authenticated = True
        caller.user = "" if authority == "operator" else authority
        caller.role = "admin" if authority == "admin" else "member"
    privileged = authority in ("loopback", "operator", "admin")
    body = {"base_model": "m", "method": "quantize", "scheme": "awq", "out_slug": "m-awq",
            "push_to_hf": True, "hf_repo": "mine/checkpoint"}
    if own_token:
        body["hf_token"] = "caller-token"
    headers = {"X-AINode-Client": "dashboard"} if authority == "loopback" else {}
    response = await client.post("/api/training/jobs", json=body, headers=headers)
    if own_token or privileged:
        assert response.status == 201
        job = client.app["training_manager"].get_job((await response.json())["job_id"])
        assert job.config.hf_token == ("caller-token" if own_token else "stored-write-token")
        assert job.config._node_hf_token_authorized is privileged
        hf_upload._whoami.assert_called_once_with(job.config.hf_token)
    else:
        assert response.status == 403
        hf_upload._whoami.assert_not_called()
        assert client.app["training_manager"].list_jobs() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["jobs", "estimate"])
async def test_public_schema_cannot_grant_node_token_authority(client, route):
    response = await client.post(f"/api/training/{route}", json={
        "base_model": "m", "dataset_path": "d.jsonl", "_node_hf_token_authorized": True,
    })
    assert response.status == 400
    assert "_node_hf_token_authorized" in (await response.json())["error"]


def checkpoint(tmp_path):
    directory = tmp_path / "models" / "m-awq"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    return directory


@pytest.fixture
def hub(monkeypatch):
    api = MagicMock()
    monkeypatch.setattr("huggingface_hub.HfApi", MagicMock(return_value=api))
    return api


@pytest.mark.parametrize("kind", ["outside", "prefix", "root", "top-link", "inner-link", "directory-link",
                                  "root-link", "fifo", "socket", "top-file", "missing", "empty"])
def test_upload_refuses_unsafe_tree_before_any_hub_call(tmp_path, monkeypatch, hub, kind):
    directory = checkpoint(tmp_path)
    root = tmp_path / "models"
    connection = None
    if kind in ("outside", "prefix"):
        directory = tmp_path / ("models-other" if kind == "prefix" else "outside")
        directory.mkdir()
        (directory / "secret").write_text("private")
    elif kind == "root":
        directory = root
    elif kind == "top-link":
        link = root / "link"
        link.symlink_to(directory, target_is_directory=True)
        directory = link
    elif kind in ("inner-link", "directory-link"):
        (directory / "link").symlink_to(tmp_path if kind == "directory-link" else directory / "config.json")
    elif kind == "root-link":
        home = tmp_path / "linked-home"
        home.mkdir()
        root = home / "models"
        root.symlink_to(tmp_path / "models", target_is_directory=True)
        monkeypatch.setattr("ainode.core.config.AINODE_HOME", home)
        directory = root / "m-awq"
    elif kind == "fifo":
        os.mkfifo(directory / "fifo")
    elif kind == "socket":
        connection = socket.socket(socket.AF_UNIX)
        connection.bind(str(directory / "sock"))
    elif kind == "top-file":
        directory = directory / "config.json"
    elif kind == "missing":
        directory = root / "missing"
    elif kind == "empty":
        (directory / "config.json").unlink()
    try:
        with pytest.raises(ValueError, match="Unsafe checkpoint"):
            hf_upload.upload_checkpoint(str(directory), "mine/model", "caller-token")
        hf_upload._whoami.assert_not_called()
        assert hub.mock_calls == []
    finally:
        if connection is not None:
            connection.close()


def test_upload_rechecks_files_after_hub_round_trip(tmp_path, hub):
    directory = checkpoint(tmp_path)
    hub.create_repo.side_effect = lambda *a, **kw: (directory / "link").symlink_to(directory / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        hf_upload.upload_checkpoint(str(directory), "mine/model", "caller-token")
    hub.upload_folder.assert_not_called()


@pytest.mark.asyncio
async def test_caller_token_uploads_only_checkpoint_and_remains_private(tmp_path, hub):
    directory = checkpoint(tmp_path)
    job = engine.TrainingJob(quant_config(out_slug="m-awq", hf_repo="mine/model", hf_token="caller-token"))
    await job._push_to_hf()
    hf_upload._whoami.assert_called_once_with("caller-token")
    hub.create_repo.assert_called_once_with("mine/model", repo_type="model", private=True, exist_ok=True)
    assert hub.upload_folder.call_args.kwargs["folder_path"] == str(directory)
    assert any("uploaded to" in line for line in job.logs)


@pytest.mark.asyncio
async def test_persisted_push_cannot_borrow_environment_token(tmp_path, monkeypatch, hub):
    checkpoint(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "node-env-token")
    job = engine.TrainingJob(quant_config(out_slug="m-awq", hf_repo="mine/model"))
    await job._push_to_hf()
    hf_upload._whoami.assert_not_called()
    hub.upload_folder.assert_not_called()


@pytest.mark.parametrize("slug", ["../", "/etc"])
def test_legacy_checkpoint_discovery_does_not_escape_store(tmp_path, slug):
    # The old traversal lookup mistook the node's own config for model output.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "models").mkdir()
    config = quant_config(out_slug=slug)
    assert engine._completed_artifact(tmp_path / "job", config) is None


@pytest.mark.parametrize("name", ["..", "../outside", "/etc", "a/../../outside", "a\\b"])
def test_model_lookup_rejects_path_names(tmp_path, name):
    models = tmp_path / "models"
    models.mkdir()
    assert registry.snapshot_dir_for(name, models) is None
    assert registry.find_model_dir(models, name) is None
    with pytest.raises(ValueError, match="Invalid model"):
        registry.ModelManager._repo_to_dirname(name)


def test_model_lookup_rejects_directory_symlink_escape(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    (models / "org--model").symlink_to(outside, target_is_directory=True)
    assert registry.snapshot_dir_for("org/model", models) is None
    assert registry.find_model_dir(models, "org/model") is None


def test_forged_dataset_id_does_not_authorize_an_absolute_path(tmp_path):
    with pytest.raises(ValueError, match="registered dataset"):
        engine.TrainingManager().submit_job(engine.TrainingConfig(
            base_model="m", dataset_id="unknown", dataset_path=str(tmp_path / "auth.json"),
        ))


@pytest.mark.asyncio
async def test_default_derived_slug_can_push_with_callers_token(client, tmp_path, hub):
    response = await client.post("/api/training/jobs", json={
        "base_model": "org/model", "method": "quantize", "scheme": "awq",
        "push_to_hf": True, "hf_token": "caller-token",
    })
    assert response.status == 201
    job = client.app["training_manager"].get_job((await response.json())["job_id"])
    directory = tmp_path / "models" / "org--model-awq"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    await job._push_to_hf()
    hub.create_repo.assert_called_once_with("owner/org-model-awq", repo_type="model", private=True, exist_ok=True)
    assert hub.upload_folder.call_args.kwargs["folder_path"] == str(directory)


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [model_routes.handle_download_repo, model_routes.handle_delete_repo])
@pytest.mark.parametrize("repo", ["../private", "a/b/c", "a\\b/c", "org/../model"])
async def test_model_repo_paths_validate_before_lookup_or_download(tmp_path, monkeypatch, handler, repo):
    fit_check = AsyncMock(return_value=web.json_response({}, status=418))
    monkeypatch.setattr(model_routes, "_refuse_if_it_will_not_fit", fit_check)
    request = SimpleNamespace(
        app={"model_manager": SimpleNamespace(models_dir=tmp_path / "models")},
        json=AsyncMock(return_value={"hf_repo": repo}),
    )
    response = await handler(request)
    assert response.status == 400
    fit_check.assert_not_called()


@pytest.mark.parametrize("repo", BAD_REPOS)
def test_upload_revalidates_repo_before_hub_call(tmp_path, hub, repo):
    directory = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="hf_repo"):
        hf_upload.upload_checkpoint(str(directory), repo, "caller-token")
    hf_upload._whoami.assert_not_called()
    hub.create_repo.assert_not_called()


@pytest.mark.asyncio
async def test_upload_runtime_cannot_escape_with_caller_token(tmp_path, hub):
    checkpoint(tmp_path)
    (tmp_path / "auth.json").write_text("private")
    job = engine.TrainingJob(quant_config(out_slug="../", hf_repo="mine/model", hf_token="caller-token"))
    await job._push_to_hf()
    hf_upload._whoami.assert_not_called()
    hub.upload_folder.assert_not_called()
    assert any("out_slug" in line for line in job.logs)


@pytest.mark.asyncio
async def test_unsafe_existing_output_answers_400_without_launching(client, tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (models / "link").symlink_to(outside, target_is_directory=True)
    manager = client.app["training_manager"]
    monkeypatch.setattr(manager, "start_next", engine.TrainingManager.start_next.__get__(manager))
    popen = MagicMock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(engine.subprocess, "Popen", popen)
    response = await client.post("/api/training/jobs", json={
        "base_model": "m", "method": "quantize", "scheme": "awq", "out_slug": "link",
    })
    assert response.status == 400
    assert "Unsafe" in (await response.json())["error"]
    popen.assert_not_called()
    assert manager.active_job is None


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_type", ["ordinary", "own-token", "operator"])
async def test_resume_rechecks_publishing_identity(client, caller_type):
    manager = client.app["training_manager"]
    config = quant_config(out_slug="m-awq", hf_repo="mine/model", push_to_hf=True, hf_token="node-token")
    config._node_hf_token_authorized = True
    source = manager.submit_job(config)
    source.status = engine.JobStatus.COMPLETED
    checkpoint_path = source._job_dir / "output" / "checkpoint-1"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / "trainer_state.json").write_text("{}")
    body = {}
    if caller_type == "own-token":
        body["hf_token"] = "caller-token"
    elif caller_type == "operator":
        client.app["auth_config"].enabled = True
        client.app["caller"].authenticated = True
    response = await client.post(f"/api/training/jobs/{source.job_id}/resume", json=body)
    if caller_type == "ordinary":
        assert response.status == 403
        assert len(manager.list_jobs()) == 1
    else:
        assert response.status == 201
        resumed = manager.get_job((await response.json())["resume_job_id"])
        assert resumed.config._node_hf_token_authorized is (caller_type == "operator")
        assert resumed.config.hf_token == ("caller-token" if caller_type == "own-token" else "stored-write-token")


@pytest.mark.asyncio
@pytest.mark.parametrize("source, token", [("config", "node-token"), ("secrets", "stored-write-token"),
                                          ("environment", "env-token")])
async def test_operator_can_push_with_each_node_token_source(client, monkeypatch, source, token):
    client.app["auth_config"].enabled = True
    client.app["caller"].authenticated = True
    if source != "secrets":
        client.app["secrets_manager"].clear()
    if source == "environment":
        client.app["config"].hf_token = None
        monkeypatch.setenv("HF_TOKEN", token)
    response = await client.post("/api/training/jobs", json={
        "base_model": "m", "method": "quantize", "scheme": "awq", "out_slug": "m-awq",
        "push_to_hf": True, "hf_repo": "mine/model",
    })
    assert response.status == 201
    job = client.app["training_manager"].get_job((await response.json())["job_id"])
    assert job.config.hf_token == token
    assert job.config._node_hf_token_authorized is True


def launch_token(command):
    if "--env-file" not in command:
        return ""
    return Path(command[command.index("--env-file") + 1]).read_text()


def probe_remote_code(monkeypatch, config_path, command, method, expected):
    """Reach the real runner's trust_remote_code loader with fake dependencies."""
    class Loaded(Exception):
        pass

    def from_pretrained(model, **kwargs):
        assert model == "attacker/model"
        assert kwargs["trust_remote_code"] is True
        assert {key: os.environ.get(key) for key in engine.SECRET_ENV_KEYS} == dict.fromkeys(
            engine.SECRET_ENV_KEYS, expected,
        )
        raise Loaded

    with monkeypatch.context() as child:
        for key in engine.SECRET_ENV_KEYS:
            child.delenv(key, raising=False)
        for line in launch_token(command).splitlines():
            key, value = line.split("=", 1)
            child.setenv(key, value)
        child.setenv("WORLD_SIZE", "1")
        loader = SimpleNamespace(from_pretrained=from_pretrained)
        child.setitem(sys.modules, "torch", SimpleNamespace(
            __version__="stub", cuda=SimpleNamespace(is_available=lambda: False),
        ))
        child.setitem(sys.modules, "transformers", SimpleNamespace(
            AutoConfig=loader, AutoTokenizer=loader, AutoModelForCausalLM=loader,
            DataCollatorForSeq2Seq=object, TrainingArguments=object, Trainer=object,
            TrainerCallback=object,
        ))
        child.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=object))
        child.setitem(sys.modules, "llmcompressor", SimpleNamespace(oneshot=object))
        child.setattr(sys, "argv", ["runner", "--config", str(config_path)])
        with pytest.raises(Loaded):
            (_run_quant if method == "quantize" else _run_training).main()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["lora", "quantize"])
@pytest.mark.parametrize("source", ["config", "secrets", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
@pytest.mark.parametrize("caller", ["open", "own-token", "operator"])
async def test_remote_code_gets_only_original_callers_token(client, monkeypatch, method, source, caller):
    node_token = "synthetic-node-token"
    client.app["config"].hf_token = node_token if source == "config" else None
    client.app["secrets_manager"].clear()
    if source == "secrets":
        client.app["secrets_manager"]["huggingface_token"] = node_token
    elif source in engine.SECRET_ENV_KEYS:
        monkeypatch.setenv(source, node_token)
    if caller == "operator":
        client.app["auth_config"].enabled = True
        client.app["caller"].authenticated = True
    body = {"base_model": "attacker/model", "dataset_path": "public/data", "method": method,
            "scheme": "awq"}
    if caller == "own-token":
        body["hf_token"] = "synthetic-caller-token"
    response = await client.post("/api/training/jobs", json=body)
    assert response.status == 201
    job = client.app["training_manager"].get_job((await response.json())["job_id"])
    command = (job._build_quant_command(job._job_dir / "config.json") if method == "quantize"
               else job._build_container_command())
    expected = {"open": None, "own-token": "synthetic-caller-token", "operator": node_token}[caller]
    assert job.config.hf_token == expected
    assert job.config._node_hf_token_authorized is (caller == "operator")
    assert not any(node_token in part or "synthetic-caller-token" in part for part in command)
    if caller != "operator":
        assert node_token not in launch_token(command)
    if expected is None:
        assert "--env-file" not in command
        assert not (job._job_dir / "hf.env").exists()
    config_path = job._job_dir / "config.container.json"
    if method == "quantize":
        config_path = job._job_dir / "config.json"
        config_path.write_text(json.dumps(engine._config_without_secrets(job.config)))
    probe_remote_code(monkeypatch, config_path, command, method, expected)


@pytest.mark.parametrize("builder", ["training", "quantize", "merge"])
@pytest.mark.parametrize("source", engine.SECRET_ENV_KEYS)
@pytest.mark.parametrize("authorized", [False, True])
def test_all_container_fallbacks_require_recorded_authority(monkeypatch, builder, source, authorized):
    monkeypatch.setenv(source, "synthetic-node-token")
    job = engine.TrainingJob(engine.TrainingConfig(base_model="m", dataset_path="d"))
    job.config._node_hf_token_authorized = authorized
    if builder == "merge":
        adapter = job._job_dir / "adapter"
        adapter.mkdir()
        command = engine.build_merge_command(job, "m", adapter, job._job_dir / "merged")
    elif builder == "training":
        command = job._build_container_command()
    else:
        job.config.method = "quantize"
        command = job._build_quant_command(job._job_dir / "config.json")
    assert ("synthetic-node-token" in launch_token(command)) is authorized


@pytest.mark.asyncio
@pytest.mark.parametrize("own_token", [False, True])
async def test_nonpublishing_resume_discards_source_credentials(client, own_token):
    manager = client.app["training_manager"]
    config = engine.TrainingConfig(base_model="m", dataset_path="d", hf_token="source-private-token")
    config._node_hf_token_authorized = True
    source = manager.submit_job(config)
    source.status = engine.JobStatus.COMPLETED
    (source._job_dir / "output" / "checkpoint-1").mkdir(parents=True)
    body = {"hf_token": "caller-token"} if own_token else {}
    response = await client.post(f"/api/training/jobs/{source.job_id}/resume", json=body)
    assert response.status == 201
    resumed = manager.get_job((await response.json())["resume_job_id"])
    assert resumed.config.hf_token == ("caller-token" if own_token else None)
    assert resumed.config._node_hf_token_authorized is False


@pytest.mark.asyncio
@pytest.mark.parametrize("container", [False, True])
@pytest.mark.parametrize("own_token", [False, True])
async def test_merge_uses_new_caller_credentials(client, monkeypatch, container, own_token):
    monkeypatch.setenv("HF_TOKEN", "synthetic-node-token")
    if container:
        monkeypatch.setenv("AINODE_IN_CONTAINER", "1")
    else:
        monkeypatch.delenv("AINODE_IN_CONTAINER", raising=False)
    monkeypatch.setattr(engine.TrainingJob, "_assert_job_image", lambda self: None)
    manager = client.app["training_manager"]
    source = manager.submit_job(engine.TrainingConfig(base_model="m", dataset_path="d",
                                                     hf_token="source-private-token"))
    source.config._node_hf_token_authorized = True
    source.status = engine.JobStatus.COMPLETED
    Path(source.config.output_dir).mkdir()
    process = SimpleNamespace(stdout=AsyncMock(), wait=AsyncMock(return_value=0))
    process.stdout = AsyncMock()
    process.stdout.__aiter__.return_value = []
    env_files = []
    async def spawn(*command, **kwargs):
        env_files.append(launch_token(command))
        return process
    launch = AsyncMock(side_effect=spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    body = {"hf_token": "caller-token"} if own_token else {}
    response = await client.post(f"/api/training/jobs/{source.job_id}/merge", json=body)
    assert response.status == 202
    merged = manager.get_job((await response.json())["merge_job_id"])
    for _ in range(100):
        if merged.status in (engine.JobStatus.COMPLETED, engine.JobStatus.FAILED):
            break
        await asyncio.sleep(0.01)
    assert merged.status == engine.JobStatus.COMPLETED
    assert merged.config.hf_token == ("caller-token" if own_token else None)
    assert merged.config._node_hf_token_authorized is False
    launch.assert_awaited_once()
    env = launch.call_args.kwargs["env"]
    for key in engine.SECRET_ENV_KEYS:
        assert env.get(key) == ("caller-token" if own_token else None)
    assert "synthetic-node-token" not in env_files[0]
    assert "source-private-token" not in env_files[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized, own_token", [(False, False), (False, True), (True, False)])
async def test_host_runner_does_not_inherit_node_token(monkeypatch, authorized, own_token):
    monkeypatch.setenv("HF_TOKEN", "synthetic-node-token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "synthetic-other-node-token")
    config = engine.TrainingConfig(base_model="m", dataset_path="d",
                                   hf_token="caller-token" if own_token else None)
    config._node_hf_token_authorized = authorized
    job = engine.TrainingJob(config)
    monkeypatch.setattr(job, "_build_command", lambda path: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(job, "_monitor", AsyncMock())
    popen = MagicMock()
    monkeypatch.setattr(engine.subprocess, "Popen", popen)
    await job.start()
    env = popen.call_args.kwargs["env"]
    expected = "caller-token" if own_token else "synthetic-node-token" if authorized else None
    assert {key: env.get(key) for key in engine.SECRET_ENV_KEYS} == dict.fromkeys(engine.SECRET_ENV_KEYS, expected)
    await job._monitor_task


@pytest.mark.parametrize("model", [".", "..", "escape"])
def test_engine_model_lookup_cannot_select_store_root_or_parent(tmp_path, model):
    from ainode.engine.backends.nvidia import NvidiaBackend
    root = tmp_path / "models"
    root.mkdir()
    (root / "model.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    # Avoid constructing a backend that creates the operator's log directory.
    backend = NvidiaBackend.__new__(NvidiaBackend)
    backend.config = SimpleNamespace(model=model, models_dir=str(root))
    assert backend._local_model_dir() is None


@pytest.mark.asyncio
async def test_artifact_download_refuses_a_symlink_to_node_secrets(client, tmp_path):
    manager = client.app["training_manager"]
    job = manager.submit_job(engine.TrainingConfig(base_model="m", dataset_path="d.jsonl"))
    output = job._job_dir / "output"
    output.mkdir()
    (tmp_path / "auth.json").write_text("private")
    (output / "auth.json").symlink_to(tmp_path / "auth.json")
    response = await client.get(f"/api/training/jobs/{job.job_id}/output/auth.json")
    assert response.status == 400
    assert "private" not in await response.text()
