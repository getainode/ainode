"""A restart keeps the engine that is already serving it (#240).

PR #227 taught a node to RECOGNISE a surviving distributed head. It still
reloaded every solo model on the way back up, because the two things that destroy
a running engine both happen in ``ainode start`` before anything is adopted: the
pre-launch sweep (`docker rm -f` every engine container this node owns) and the
boot engine's own launch, with ``engine.stop()`` on the way out having removed the
container in the first place. Measured cost of that: 420 s on pollux, 834 s on
castor, on every `systemctl restart` and every `ainode update`.

So the decision moves ahead of both, and it is a decision with three gates:
RUNNING, the same SHAPE the configured recipe renders, and ANSWERING. All three
pass and the container is kept, skipped by the sweep, not relaunched, and put in
the InstanceManager as ``adopted``. Any one of them fails and the boot reloads
exactly as it did before, with one line saying which.

Fakes only: no docker, no HTTP, no sleeps. The seams are
``reconcile.inspect_container``, ``list_engine_containers``, ``port_health``,
``served_models`` and ``port_serving``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import ainode.engine.reconcile as rec
import ainode.models.api_routes as mr
from ainode.cli import main as cli
from ainode.core.config import NodeConfig
from ainode.engine.backends.nvidia import NVIDIA_VLLM_IMAGE
from ainode.engine.instance_manager import InstanceManager

MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
IMAGE = "onecat-vllm:src-full"
OTHER = "cerebras/Ornith-1.5-35B-A3B-NVFP4"
PRIMARY = "ainode-vllm-node-solo"

# pollux's real argv, from `docker inspect ainode-vllm-node-solo` on the node on
# 2026-09-20 (entrypoint ["vllm"], the rest the Cmd). Kept verbatim so the gate is
# pinned against production output and not against what this file thinks a launch
# renders.
POLLUX_ARGV = [
    "vllm", "serve", "/ainode-models/nvidia--Qwen3.6-35B-A3B-NVFP4",
    "--host", "0.0.0.0", "--port", "8000",
    "--gpu-memory-utilization", "0.9",
    "--kv-cache-dtype", "auto",
    "--max-model-len", "65536",
    "--trust-remote-code",
    "--attention-backend", "FLASH_ATTN_V100",
    "--max-num-seqs", "8",
    "--enable-prefix-caching",
    "--reasoning-parser", "qwen3",
    "--tool-call-parser", "qwen3_coder",
    "--enable-auto-tool-choice",
    "--limit-mm-per-prompt", '{"image":0,"video":0}',
    "--served-model-name", "nvidia/Qwen3.6-35B-A3B-NVFP4",
]

POLLUX_EXTRA_ARGS = [
    "--attention-backend", "FLASH_ATTN_V100",
    "--max-num-seqs", "8",
    "--enable-prefix-caching",
    "--reasoning-parser", "qwen3",
    "--tool-call-parser", "qwen3_coder",
    "--enable-auto-tool-choice",
    "--limit-mm-per-prompt", '{"image":0,"video":0}',
]


# ------------------------------------------------------------------ harness --

def _inspect(argv, *, name=PRIMARY, running=True, cid="c0ffee123456",
             image=IMAGE, started="2026-09-20T11:22:26.758191156Z") -> dict:
    return {
        "Id": cid,
        "Name": "/" + name,
        "Config": {"Image": image, "Entrypoint": ["vllm"], "Cmd": list(argv[1:])},
        "State": {"Running": running, "Status": "running" if running else "exited",
                  "StartedAt": started},
    }


def _stacked_argv(port: int, model: str, gmu="0.35") -> list:
    return ["vllm", "serve", model, "--host", "0.0.0.0", "--port", str(port),
            "--gpu-memory-utilization", gmu, "--max-model-len", "32768",
            "--served-model-name", model]


def _config(**kw) -> NodeConfig:
    defaults = dict(node_id="d0901a4b", node_name="pollux", api_port=8000,
                    web_port=3000, model=MODEL, distributed_mode="solo",
                    engine_backend="nvidia", engine_strategy="pip",
                    engine_image=IMAGE, gpu_memory_utilization=0.9,
                    kv_cache_dtype="auto", kv_cache_dtype_explicit=True,
                    max_model_len=65536, trust_remote_code=True,
                    extra_vllm_args=list(POLLUX_EXTRA_ARGS), onboarded=True)
    defaults.update(kw)
    config = NodeConfig(**defaults)
    config.save = lambda *a, **k: None
    return config


class _FakeBackend:
    """A backend handle with no docker behind it."""

    built: list = []

    def __init__(self, config, on_ready=None, instance_id=""):
        self.config = config
        self.instance_id = instance_id
        self.started = 0
        self.stopped = 0
        self._launched_at = None
        _FakeBackend.built.append(self)

    def start(self):
        self.started += 1
        return True

    def stop(self):
        self.stopped += 1

    def is_running(self):
        return True


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """A clean AINODE_HOME, healthy probes by default, and no real docker."""
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(rec, "port_health", lambda port, timeout=3.0: True)
    monkeypatch.setattr(rec, "served_models",
                        lambda port, timeout=3.0: [MODEL, OTHER])
    monkeypatch.setattr(rec, "port_serving", lambda port, timeout=3.0: True)
    monkeypatch.setattr("ainode.engine.backends.get_backend", _FakeBackend)
    monkeypatch.setattr(InstanceManager, "_port_bindable",
                        staticmethod(lambda port, host="0.0.0.0": True))
    _FakeBackend.built = []
    rec.reset_state_for_tests()
    yield
    rec.reset_state_for_tests()


def _fake_docker(monkeypatch, containers: dict):
    """``{name: inspect dict}`` for both seams at once."""
    monkeypatch.setattr(rec, "inspect_container", lambda name: containers.get(name))
    monkeypatch.setattr(rec, "list_engine_containers", lambda: list(containers))


def _app(config=None, engine=None, manager=None) -> dict:
    config = config if config is not None else _config()
    return {"config": config, "engine": engine,
            "instances": manager if manager is not None
            else InstanceManager(base_port=config.api_port)}


# ----------------------------------------------------------- the three gates --

def test_a_healthy_primary_is_kept_and_named_in_one_line(monkeypatch):
    """RUNNING plus the configured shape plus answering: keep it."""
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})

    decision = rec.adopt_boot_engines(_config())

    entry = decision["primary"]
    assert entry is not None
    assert entry["model"] == MODEL
    assert entry["api_port"] == 8000
    assert entry["container"] == PRIMARY
    assert entry["image"] == IMAGE
    assert decision["stacked"] == []
    # The operator's line, and the shape the issue asked for.
    assert decision["lines"] == [
        f"adopted {MODEL} on :8000 (container {PRIMARY}, up {entry['uptime']})"]
    # And the sweep is told to leave it alone.
    assert rec.adopted_container_ids() == {"c0ffee123456"}


def test_the_real_pollux_container_passes_the_gate(monkeypatch):
    """The argv in this file is the one pollux is actually running.

    The parse and every comparison run against production output: the serve
    target is a mount path, the id comes from --served-model-name, the width is
    stated nowhere (a solo launch renders TP only from extra_vllm_args) and the
    image is a locally built Volta fork.
    """
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    config = _config()
    shape = rec.container_shape(_inspect(POLLUX_ARGV))
    expected = rec.expected_shape(config)

    assert shape["model"] == MODEL          # not the /ainode-models path
    assert shape["api_port"] == 8000
    assert shape["tensor_parallel_size"] == 0   # the flag is absent
    assert expected["tensor_parallel_size"] == 1
    assert expected["image"] == IMAGE
    assert rec.shape_mismatch(expected, shape) == ""
    assert rec.adopt_boot_engines(config)["primary"] is not None


def test_a_primary_serving_another_model_is_relaunched(monkeypatch):
    _fake_docker(monkeypatch, {PRIMARY: _inspect(_stacked_argv(8000, OTHER))})
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is None
    assert decision["lines"] == [
        f"relaunching {MODEL}: container {PRIMARY} is serving {OTHER}, "
        f"config says {MODEL}"]
    assert rec.adopted_container_ids() == set()


def test_a_primary_on_the_previous_engine_image_is_relaunched(monkeypatch):
    """The update case: a release moved the recipe's image, so it has to reload."""
    _fake_docker(monkeypatch,
                 {PRIMARY: _inspect(POLLUX_ARGV, image="onecat-vllm:1.4.0")})
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is None
    assert decision["lines"] == [
        f"relaunching {MODEL}: container {PRIMARY} runs onecat-vllm:1.4.0, "
        f"config says {IMAGE}"]


def test_a_primary_at_the_wrong_width_is_relaunched(monkeypatch):
    """castor's lane: the width lives in extra_vllm_args, so it is compared."""
    four_wide = POLLUX_ARGV + ["--tensor-parallel-size", "4"]
    _fake_docker(monkeypatch, {PRIMARY: _inspect(four_wide)})
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is None
    assert "is TP=4, config says TP=1" in decision["lines"][0]

    # And the other direction: a config that asks for four keeps a TP=4 container.
    rec.reset_state_for_tests()
    config = _config(extra_vllm_args=["--tensor-parallel-size", "4"])
    assert rec.adopt_boot_engines(config)["primary"] is not None


def test_a_stopped_primary_container_is_relaunched(monkeypatch):
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV, running=False)})
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is None
    assert decision["lines"] == [
        f"relaunching {MODEL}: container {PRIMARY} is exited"]


def test_no_primary_container_at_all_is_relaunched(monkeypatch):
    _fake_docker(monkeypatch, {})
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is None
    assert decision["lines"] == [
        f"relaunching {MODEL}: no container {PRIMARY} is running here"]


def test_a_primary_that_is_not_answering_yet_is_relaunched(monkeypatch):
    """Up but not bound is the mid-load case, and it reloads exactly as today.

    Nothing is lost by declining: the sweep frees it and the boot engine launches
    it, which is what would have happened without adoption at all.
    """
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    monkeypatch.setattr(rec, "port_health", lambda port, timeout=3.0: False)
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is None
    assert decision["lines"] == [
        f"relaunching {MODEL}: container {PRIMARY} is up but /health on :8000 "
        f"did not answer"]


def test_a_primary_whose_engine_names_another_model_is_relaunched(monkeypatch):
    """The container's argv can be right while the engine serves something else."""
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    monkeypatch.setattr(rec, "served_models", lambda port, timeout=3.0: [OTHER])
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is None
    assert f"/v1/models on :8000 names {OTHER}, not {MODEL}" in decision["lines"][0]


def test_an_engine_that_answers_nothing_at_all_is_relaunched(monkeypatch):
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    monkeypatch.setattr(rec, "served_models", lambda port, timeout=3.0: [])
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is None
    assert "/v1/models on :8000 named no model" in decision["lines"][0]


# ------------------------------------------------------- what is out of scope --

def test_start_clean_adopts_nothing(monkeypatch):
    """The knob exists to free a node, so it must not keep an engine alive."""
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    config = _config()
    config._skip_replay = True
    decision = rec.adopt_boot_engines(config)

    assert decision == {"primary": None, "stacked": [], "lines": []}
    assert rec.adopted_container_ids() == set()


def test_a_member_adopts_nothing(monkeypatch):
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    decision = rec.adopt_boot_engines(_config(distributed_mode="member"))
    assert decision["primary"] is None


def test_a_backend_that_runs_no_container_adopts_nothing(monkeypatch):
    """The eugr backend drives vLLM on the host: there is no container of ours."""
    asked: list = []
    monkeypatch.setattr(rec, "inspect_container",
                        lambda name: asked.append(name) or None)
    decision = rec.adopt_boot_engines(_config(engine_backend="eugr"))
    assert decision["primary"] is None
    assert asked == []


def test_a_node_with_no_model_configured_adopts_nothing(monkeypatch):
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    decision = rec.adopt_boot_engines(_config(model=None))
    assert decision["primary"] is None
    assert decision["lines"] == []


def test_the_head_container_is_kept_the_same_way(monkeypatch):
    """A head is the primary too, so the 834 s restart on a TP=4 lane is covered."""
    head_argv = ["vllm", "serve", MODEL, "--host", "0.0.0.0", "--port", "8000",
                 "--tensor-parallel-size", "2",
                 "--distributed-executor-backend", "mp",
                 "--nnodes", "2", "--node-rank", "0",
                 "--served-model-name", MODEL]
    _fake_docker(monkeypatch, {"ainode-vllm-head": _inspect(head_argv,
                                                           name="ainode-vllm-head")})
    config = _config(distributed_mode="head", peer_ips=["10.100.0.15"],
                     distributed_executor="mp", extra_vllm_args=[])
    decision = rec.adopt_boot_engines(config)

    assert decision["primary"] is not None
    assert decision["primary"]["container"] == "ainode-vllm-head"
    assert decision["primary"]["tensor_parallel_size"] == 2
    assert decision["primary"]["distributed_mode"] == "head"


def test_a_head_running_the_wrong_shape_is_relaunched(monkeypatch):
    """A ray head container states no serve target, so there is nothing to match."""
    _fake_docker(monkeypatch, {"ainode-vllm-head": _inspect(
        ["ray", "start", "--head", "--port", "6379", "--block"],
        name="ainode-vllm-head")})
    config = _config(distributed_mode="head", peer_ips=["10.100.0.15"],
                     distributed_executor="ray", extra_vllm_args=[])
    decision = rec.adopt_boot_engines(config)

    assert decision["primary"] is None
    assert "names no served model in its own argv" in decision["lines"][0]


# ------------------------------------------------------- the stacked instances --

def _write_manifest(entries: list) -> None:
    path = mr._manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"instances": entries}))


def test_a_recorded_stacked_instance_is_kept(monkeypatch):
    stacked = "ainode-vllm-node-solo-8001"
    _write_manifest([
        {"model": MODEL, "gpu_memory_utilization": 0.9},
        {"model": OTHER, "gpu_memory_utilization": 0.35,
         "max_model_len": 32768, "engine_image": IMAGE},
    ])
    _fake_docker(monkeypatch, {
        PRIMARY: _inspect(POLLUX_ARGV),
        stacked: _inspect(_stacked_argv(8001, OTHER), name=stacked, cid="stacked01"),
    })
    decision = rec.adopt_boot_engines(_config())

    assert decision["primary"] is not None
    assert [e["model"] for e in decision["stacked"]] == [OTHER]
    assert decision["stacked"][0]["api_port"] == 8001
    assert rec.adopted_container_ids() == {"c0ffee123456", "stacked01"}
    assert decision["lines"][1].startswith(f"adopted {OTHER} on :8001")


def test_a_stacked_instance_on_another_image_is_relaunched(monkeypatch):
    """The manifest entry is the recipe, so a moved image is caught here too."""
    stacked = "ainode-vllm-node-solo-8001"
    _write_manifest([{"model": OTHER, "gpu_memory_utilization": 0.35,
                      "engine_image": "vllm/vllm-openai:v0.28.0"}])
    _fake_docker(monkeypatch, {
        stacked: _inspect(_stacked_argv(8001, OTHER), name=stacked,
                          cid="stacked01", image="vllm/vllm-openai:v0.27.1"),
    })
    decision = rec.adopt_boot_engines(_config(model=None))

    assert decision["stacked"] == []
    assert decision["lines"] == [
        f"relaunching {OTHER}: container {stacked} runs vllm/vllm-openai:v0.27.1, "
        f"config says vllm/vllm-openai:v0.28.0"]


def test_a_stacked_instance_the_manifest_never_heard_of_is_left_to_adoption(
        monkeypatch):
    """No entry means no recipe to compare: #179's orphan path owns that case."""
    stacked = "ainode-vllm-node-solo-8001"
    _write_manifest([])
    _fake_docker(monkeypatch, {
        stacked: _inspect(_stacked_argv(8001, OTHER), name=stacked, cid="stacked01"),
    })
    decision = rec.adopt_boot_engines(_config(model=None))

    assert decision["stacked"] == []
    assert decision["lines"] == []
    assert rec.adopted_container_ids() == set()


def test_a_stacked_instance_with_no_image_of_its_own_compares_the_default(
        monkeypatch):
    """A manifest entry that states no image expects the node's default image."""
    stacked = "ainode-vllm-node-solo-8001"
    _write_manifest([{"model": OTHER, "gpu_memory_utilization": 0.35}])
    _fake_docker(monkeypatch, {
        stacked: _inspect(_stacked_argv(8001, OTHER), name=stacked,
                          cid="stacked01", image=NVIDIA_VLLM_IMAGE),
    })
    decision = rec.adopt_boot_engines(_config(model=None))

    assert [e["model"] for e in decision["stacked"]] == [OTHER]


# ----------------------------------------------------------------- the sweep --

def test_the_boot_sweep_neither_removes_nor_waits_for_an_adopted_container(
        monkeypatch):
    """Both halves: `docker rm -f` skips it AND the poll does not wait for it."""
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    rec.adopt_boot_engines(_config())
    assert rec.adopted_container_ids() == {"c0ffee123456"}

    removed: list = []

    class _Proc:
        stdout = "c0ffee123456\nsweepme02\n"

    def _run(argv, **kwargs):
        if argv[:3] == ["docker", "rm", "-f"]:
            removed.extend(argv[3:])
        return _Proc()

    monkeypatch.setattr("subprocess.run", _run)
    # The poll answers with the adopted container STILL there, which is the point:
    # it is deliberately still running, so the sweep must not spend its timeout
    # waiting for it to disappear.
    monkeypatch.setattr(mr, "_engine_container_ids",
                        lambda include_primary: ["c0ffee123456"])
    monkeypatch.setattr(mr, "_claim_boot_sweep", lambda: True)
    monkeypatch.setattr(mr.time, "sleep", lambda s: (_ for _ in ()).throw(
        AssertionError("the sweep waited for an adopted container")))

    assert mr.sweep_engines_before_boot() == ["sweepme02"]
    assert removed == ["sweepme02"]


# ----------------------------------------------------- into the InstanceManager --

def test_adoption_puts_the_kept_primary_in_the_manager(monkeypatch):
    """The record the dashboard and /api/status read, with adopted on it."""
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    config = _config()
    rec.adopt_boot_engines(config)
    app = _app(config, engine=_FakeBackend(config))

    adopted = asyncio.run(rec.adopt_running_engines(app))

    assert [a["kind"] for a in adopted] == ["primary"]
    record = app["instances"].by_port(8000).record
    assert record.model == MODEL
    assert record.adopted is True
    assert record.status == "serving"


def test_adoption_builds_a_handle_when_the_container_outlived_the_process(
        monkeypatch):
    """No boot engine (a server started directly): the record still gets a backend."""
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    config = _config()
    rec.adopt_boot_engines(config)
    app = _app(config, engine=None)

    asyncio.run(rec.adopt_running_engines(app))

    inst = app["instances"].by_port(8000)
    assert inst.backend is not None
    assert inst.backend.config.model == MODEL
    # Stamped from the container's own StartedAt, not from this process.
    assert inst.backend._launched_at is not None
    assert app["engine"] is inst.backend


def test_a_kept_stacked_instance_carries_its_manifest_recipe(monkeypatch):
    """Not the argv-derived snapshot: the entry is the full override set.

    A snapshot missing the entry's engine image and extra flags would be written
    back OVER that entry by the next manifest save, so the model would come back
    on the wrong image after the restart after this one.
    """
    stacked = "ainode-vllm-node-solo-8001"
    _write_manifest([{"model": OTHER, "gpu_memory_utilization": 0.35,
                      "max_model_len": 32768, "engine_image": IMAGE,
                      "extra_vllm_args": ["--enable-prefix-caching"]}])
    _fake_docker(monkeypatch, {
        stacked: _inspect(_stacked_argv(8001, OTHER), name=stacked, cid="stacked01"),
    })
    config = _config(model=None)
    rec.adopt_boot_engines(config)
    app = _app(config)

    asyncio.run(rec.adopt_running_engines(app))

    inst = app["instances"].by_port(8001)
    assert inst.record.model == OTHER
    assert inst.record.adopted is True
    assert inst.backend.config.engine_image == IMAGE
    assert inst.backend.config.extra_vllm_args == ["--enable-prefix-caching"]
    assert inst.backend.config.max_model_len == 32768
    assert inst.backend.config.gpu_memory_utilization == 0.35
    # The manifest is unchanged: it already described this instance.
    manifest = json.loads(mr._manifest_path().read_text())["instances"]
    assert [e["model"] for e in manifest] == [OTHER]


def test_a_stacked_load_beside_an_adopted_primary_still_stacks(monkeypatch):
    """The 0.5.25 guard: the adopted primary holds the node's port, so a second
    model is STACKED and never overwrites config.model."""
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    # 0.5 rather than pollux's 0.9, so the stacked load's own reservation fits
    # under the node's 0.90 overcommit cap and the test is about the PORT.
    config = _config(gpu_memory_utilization=0.5)
    rec.adopt_boot_engines(config)
    app = _app(config, engine=_FakeBackend(config))
    asyncio.run(rec.adopt_running_engines(app))

    result = mr.append_solo_instance(app, OTHER, 0.3, persist=False)

    assert result["ok"] is True
    assert result["stacked"] is True
    assert result["api_port"] == 8001
    assert config.model == MODEL           # the primary was not taken over
    assert app["instances"].by_port(8000).record.model == MODEL


def test_the_app_seeds_the_primary_record_as_adopted(monkeypatch, tmp_path):
    """What ``/api/status``, ``/api/nodes`` and the cluster graphic read.

    ``create_app`` seeds the boot engine as the primary instance; when the boot
    adopted that container the record has to SAY so, and say it is serving rather
    than starting, or the dashboard draws a loading card for a model that is
    answering requests.
    """
    from ainode.api import server

    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    config = _config(models_dir=str(tmp_path / "models"),
                     datasets_dir=str(tmp_path / "datasets"),
                     training_dir=str(tmp_path / "runs"))
    rec.adopt_boot_engines(config)
    # Set AFTER the decision: this flag is start-clean's, and start-clean
    # deliberately adopts nothing. Here it only keeps the background replay task
    # out of a unit test.
    config._skip_replay = True

    app = server.create_app(config=config, engine=_FakeBackend(config))

    record = app["instances"].by_port(8000).record
    assert record.adopted is True
    assert record.status == "serving"
    assert record.model == MODEL
    # And it travels: the announcement carries the flag to every node view.
    wire = server.announced_instances(config, [record], "solo", True)
    assert wire[0]["adopted"] is True


def test_the_app_seeds_a_launched_primary_as_starting(monkeypatch, tmp_path):
    """No adoption, no change: a launch this process made is starting, not adopted."""
    from ainode.api import server

    _fake_docker(monkeypatch, {})
    config = _config(models_dir=str(tmp_path / "models"),
                     datasets_dir=str(tmp_path / "datasets"),
                     training_dir=str(tmp_path / "runs"))
    rec.adopt_boot_engines(config)
    # Set AFTER the decision: this flag is start-clean's, and start-clean
    # deliberately adopts nothing. Here it only keeps the background replay task
    # out of a unit test.
    config._skip_replay = True

    app = server.create_app(config=config, engine=_FakeBackend(config))

    record = app["instances"].by_port(8000).record
    assert record.adopted is False
    assert record.status == "starting"


# ------------------------------------------------------------- the boot order --

def test_the_replay_neither_waits_on_nor_records_an_adopted_primary(monkeypatch):
    """A bind wait would write the container's whole life into the ledger.

    ``_wait_for_bind`` reports the CONTAINER's age, so an engine adopted after
    fourteen hours up would be recorded as a fourteen-hour load and every "how
    long does this model take here" answer would read it.
    """
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    config = _config()
    rec.adopt_boot_engines(config)
    engine = _FakeBackend(config)
    app = _app(config, engine=engine)

    waits: list = []

    async def _no_wait(*a, **kw):
        waits.append(a)
        return True

    monkeypatch.setattr(mr, "_ensure_serving", _no_wait)
    monkeypatch.setattr(mr, "_wait_port_ready", _no_wait)

    async def _serving(port):
        return True

    monkeypatch.setattr(mr, "_port_serving", _serving)
    asyncio.run(mr._replay_serialized(app, config, []))

    assert waits == [], "an adopted primary must not go through the bind wait"
    assert mr.read_launch_times() == []
    assert engine.started == 0


def test_the_replay_still_waits_on_a_primary_this_boot_launched(monkeypatch):
    """No adoption, no change: the boot primary is waited for and retried once."""
    _fake_docker(monkeypatch, {})
    config = _config()
    rec.adopt_boot_engines(config)
    engine = _FakeBackend(config)
    app = _app(config, engine=engine)

    waited: list = []

    async def _ensure(app_, port, relaunch, label, timeout=300.0, backend=None):
        waited.append((port, label))
        return True

    monkeypatch.setattr(mr, "_ensure_serving", _ensure)
    asyncio.run(mr._replay_serialized(app, config, []))

    assert waited == [(8000, f"boot primary {MODEL}")]


# ------------------------------------------------------------- `ainode start` --

@pytest.fixture
def start_harness(monkeypatch):
    """Drive cmd_start with no disk, no GPU, no server and a fake backend."""
    state = {"engines": [], "ran": 0, "removed_pid": 0}

    monkeypatch.setattr(cli, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "_write_pid", lambda: None)
    monkeypatch.setattr(cli, "_remove_pid",
                        lambda: state.__setitem__("removed_pid",
                                                  state["removed_pid"] + 1))
    monkeypatch.setattr("ainode.core.gpu.detect_gpu", lambda: None)
    monkeypatch.setattr("ainode.models.api_routes.consume_start_clean", lambda: False)
    swept: list = []
    monkeypatch.setattr("ainode.models.api_routes.sweep_engines_before_boot",
                        lambda: swept.append(rec.adopted_container_ids()) or [])
    state["swept"] = swept
    monkeypatch.delenv("AINODE_IN_CONTAINER", raising=False)

    def _backend(config, on_ready=None, instance_id=""):
        engine = _FakeBackend(config, instance_id=instance_id)
        state["engines"].append(engine)
        return engine

    monkeypatch.setattr("ainode.engine.backends.get_backend", _backend)

    def _run_server(config=None, engine=None):
        state["ran"] += 1
        state["served"] = engine

    monkeypatch.setattr("ainode.api.server.run_server", _run_server)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    return state


def _flat(text: str) -> str:
    """Console output with Rich's 80-column wrapping collapsed back to one line."""
    return " ".join(text.split())


def _start(monkeypatch, config):
    monkeypatch.setattr(cli.NodeConfig, "load", classmethod(lambda cls: config))
    cli.cmd_start(SimpleNamespace(model=None, port=None, in_container=False))


def test_ainode_start_does_not_relaunch_an_adopted_engine(
        start_harness, monkeypatch, capsys):
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    config = _config()

    _start(monkeypatch, config)

    engine = start_harness["served"]
    assert engine is not None
    assert engine.started == 0, "the engine was relaunched despite being adopted"
    # The handle carries the container's own start time, so is_running() and any
    # bind wait ask docker rather than a subprocess this process never had.
    assert engine._launched_at is not None
    # The sweep ran AFTER the decision, and knew to keep that container.
    assert start_harness["swept"] == [{"c0ffee123456"}]
    out = _flat(capsys.readouterr().out)
    assert f"adopted {MODEL} on :8000 (container {PRIMARY}, up " in out
    assert "Engine adopted, no reload" in out
    # Left running on the way out, so the NEXT boot can adopt it too.
    assert engine.stopped == 0


def test_ainode_start_relaunches_when_the_shape_moved(
        start_harness, monkeypatch, capsys):
    _fake_docker(monkeypatch,
                 {PRIMARY: _inspect(POLLUX_ARGV, image="onecat-vllm:1.4.0")})
    _start(monkeypatch, _config())

    assert start_harness["served"].started == 1
    out = _flat(capsys.readouterr().out)
    assert f"relaunching {MODEL}: container {PRIMARY} runs onecat-vllm:1.4.0" in out
    assert "Engine starting in background" in out


def test_ainode_start_relaunches_when_there_is_no_container(
        start_harness, monkeypatch):
    _fake_docker(monkeypatch, {})
    _start(monkeypatch, _config())
    assert start_harness["served"].started == 1


# ------------------------------------------------------------- the way out --

def test_a_running_engine_is_left_alone_on_shutdown(monkeypatch):
    """The other half of the fix: stopping it here is what forced the reload."""
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV)})
    config = _config()
    engine = _FakeBackend(config)

    line = rec.keep_engines_on_shutdown(engine, config)

    assert engine.stopped == 0
    assert line == (f"engine left serving {MODEL} on :8000 (container {PRIMARY}); "
                    f"the next start adopts it. To free the GPU: "
                    f"docker rm -f {PRIMARY}")


def test_an_engine_whose_container_is_gone_is_still_stopped(monkeypatch):
    """That call is what reaps a corpse and a head's peer containers."""
    _fake_docker(monkeypatch, {})
    config = _config()
    engine = _FakeBackend(config)

    assert rec.keep_engines_on_shutdown(engine, config) == ""
    assert engine.stopped == 1


def test_an_exited_container_is_stopped_through_the_backend(monkeypatch):
    _fake_docker(monkeypatch, {PRIMARY: _inspect(POLLUX_ARGV, running=False)})
    config = _config()
    engine = _FakeBackend(config)

    assert rec.keep_engines_on_shutdown(engine, config) == ""
    assert engine.stopped == 1


# ------------------------------------------------------------ small helpers --

def test_the_container_name_a_launch_would_have_used():
    assert rec.primary_container_name(_config()) == PRIMARY
    assert rec.primary_container_name(
        _config(distributed_mode="head")) == "ainode-vllm-head"
    assert rec.stacked_container_name(8001) == "ainode-vllm-node-solo-8001"


def test_the_expected_served_id_follows_the_backend():
    assert rec.expected_served_model(_config()) == MODEL
    assert rec.expected_served_model(_config(served_model_name=["Aegis-14B"])) \
        == "Aegis-14B"
    # A recipe that states the flag itself wins, same rule as every serve flag.
    assert rec.expected_served_model(
        _config(extra_vllm_args=["--served-model-name", "mine"])) == "mine"


def test_uptime_reads_for_a_human():
    assert rec.uptime_phrase(1000.0, now=1037.0) == "37s"
    assert rec.uptime_phrase(1000.0, now=1000.0 + 20 * 60) == "20m"
    assert rec.uptime_phrase(1000.0, now=1000.0 + 13 * 3600 + 41 * 60) == "13h 41m"
    assert rec.uptime_phrase(None) == "unknown"


def test_the_engine_image_has_one_home():
    """The reconciler and the backend must resolve the same image."""
    from ainode.engine.backends.nvidia import NvidiaBackend, resolve_engine_image

    config = _config()
    assert resolve_engine_image(config) == IMAGE
    assert NvidiaBackend(config)._engine_image() == IMAGE
    assert resolve_engine_image(_config(engine_image="")) == NVIDIA_VLLM_IMAGE
