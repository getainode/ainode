"""A restart on a distributed head does not lose the multi-node model (#179).

The three behaviours, in the order a boot uses them:

* ADOPTION: the engine containers survive an orchestrator restart, so the first
  thing a node does is ask docker what it is already running and put it back in
  the InstanceManager with the shape the container itself states. Nothing is
  relaunched, no peer is contacted, and an adopted record says so.
* THE RECORD: the distributed shape is written to ``distributed.json`` on a
  successful launch and removed on unload, so a restart knows what this node was
  serving even when the container is gone.
* THE REPLAY POLICY: record present and container gone relaunches ONLY when
  every peer answers the probe the launch depends on; otherwise the record is
  marked degraded with which peer said what, and that shows up in
  ``/api/status``, in ``ainode doctor`` and as a dashboard banner. One attempt
  per process, never a loop.

Fakes only: no docker, no ssh, no HTTP, no sleeps.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import ainode.engine.reconcile as rec
import ainode.models.api_routes as mr
from ainode.core.config import NodeConfig
from ainode.discovery.instance import InstanceRecord
from ainode.engine.instance_manager import InstanceManager

MODEL = "fraserprice/DeepSeek-V4-Flash-DSpark"
IMAGE = "vllm-dspark-runtime:dspark-nvfp4-stage-c"
PEER = "10.100.0.15"


# ------------------------------------------------------------------ harness --

def _mp_head_argv(port: int = 8000, tp: int = 2, model: str = MODEL) -> list:
    """The argv an mp head container actually carries (rank 0 of nnodes)."""
    return [
        "vllm", "serve", "/ainode-models/" + model.replace("/", "--"),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", "0.8",
        "--kv-cache-dtype", "nvfp4_ds_mla",
        "--max-model-len", "1048576",
        "--port", str(port),
        "--enforce-eager",
        "--served-model-name", model,
        "--nnodes", str(tp), "--node-rank", "0",
        "--master-addr", "10.100.0.13", "--master-port", "29500",
    ]


def _stacked_argv(port: int, model: str) -> list:
    return [
        "vllm", "serve", model,
        "--port", str(port),
        "--gpu-memory-utilization", "0.35",
        "--max-model-len", "32768",
    ]


def _inspect(argv, *, running=True, cid="c0ffee1234", image=IMAGE,
             started="2026-09-19T20:14:31.123456789Z") -> dict:
    return {
        "Id": cid,
        "Name": "/ainode-vllm-head",
        "Config": {"Image": image, "Entrypoint": None, "Cmd": list(argv)},
        "State": {"Running": running, "Status": "running" if running else "exited",
                  "StartedAt": started},
    }


class _FakeBackend:
    """A backend handle with no docker behind it."""

    built: list = []

    def __init__(self, config, on_ready=None, instance_id=""):
        self.config = config
        self.instance_id = instance_id
        self.stopped = False
        self.launched = False
        self._launched_at = None
        _FakeBackend.built.append(self)

    def is_running(self):
        return True

    def stop(self):
        self.stopped = True

    def start_distributed(self):
        self.launched = True
        return True

    def health_check(self):
        return {"api_responding": True, "models_loaded": [self.config.model]}


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Every test gets its own AINODE_HOME and a clean per-process state."""
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(rec, "port_serving", lambda port, timeout=3.0: True)
    monkeypatch.setattr("ainode.engine.backends.get_backend", _FakeBackend)
    _FakeBackend.built = []
    rec.reset_state_for_tests()
    yield
    rec.reset_state_for_tests()


def _config(**kw) -> NodeConfig:
    defaults = dict(node_id="25033c02", node_name="Spark-2-DGX", api_port=8000,
                    web_port=3000, model=MODEL, distributed_mode="head",
                    distributed_executor="mp", peer_ips=[PEER],
                    engine_image=IMAGE, gpu_memory_utilization=0.8,
                    ssh_user="sem")
    defaults.update(kw)
    config = NodeConfig(**defaults)
    config.save = lambda *a, **k: None
    return config


def _app(config=None, engine=None, manager=None) -> dict:
    config = config if config is not None else _config()
    return {"config": config, "engine": engine,
            "instances": manager if manager is not None
            else InstanceManager(base_port=config.api_port)}


# ----------------------------------------------------------------- adoption --

def test_a_running_head_container_is_adopted_with_its_real_shape(monkeypatch, caplog):
    """The manager gets the model, width, peers, port and executor, from docker."""
    monkeypatch.setattr(rec, "inspect_container",
                        lambda name: _inspect(_mp_head_argv()) if name == "ainode-vllm-head" else None)
    monkeypatch.setattr(rec, "list_engine_containers", lambda: ["ainode-vllm-head"])
    app = _app()
    with caplog.at_level("INFO"):
        adopted = asyncio.run(rec.adopt_running_engines(app))

    assert [a["kind"] for a in adopted] == ["head"]
    records = app["instances"].records()
    assert len(records) == 1
    record = records[0]
    assert record.model == MODEL
    assert record.api_port == 8000
    assert record.tensor_parallel_size == 2
    assert record.peer_ips == [PEER]
    assert record.distributed_executor == "mp"
    assert record.adopted is True
    assert record.status == "serving"
    # Nothing was launched: adoption is a read.
    assert all(not b.launched for b in _FakeBackend.built)
    assert "adopted the distributed head container ainode-vllm-head" in caplog.text


def test_adoption_writes_the_record_for_a_head_that_came_from_config(monkeypatch):
    """The mp head is a config state, so adoption is what first writes it down."""
    monkeypatch.setattr(rec, "inspect_container", lambda name: _inspect(_mp_head_argv()))
    monkeypatch.setattr(rec, "list_engine_containers", lambda: ["ainode-vllm-head"])
    assert rec.load_distributed_record() is None
    asyncio.run(rec.adopt_running_engines(_app()))

    record = rec.load_distributed_record()
    assert record["model"] == MODEL
    assert record["peer_ips"] == [PEER]
    assert record["tensor_parallel_size"] == 2
    assert record["distributed_executor"] == "mp"
    assert record["container"] == "ainode-vllm-head"
    assert record["status"] == rec.SERVING
    # The proven recipe rides along, so a relaunch renders the same flags.
    assert record["overrides"]["engine_image"] == IMAGE


def test_an_exited_head_container_is_not_adopted(monkeypatch, caplog):
    monkeypatch.setattr(rec, "inspect_container",
                        lambda name: _inspect(_mp_head_argv(), running=False))
    monkeypatch.setattr(rec, "list_engine_containers", lambda: ["ainode-vllm-head"])
    app = _app()
    with caplog.at_level("INFO"):
        assert asyncio.run(rec.adopt_running_engines(app)) == []
    assert app["instances"].is_empty()
    assert "not adopting it" in caplog.text


def test_a_solo_node_with_no_record_adopts_nothing(monkeypatch):
    """Adoption only asks about shapes this node is supposed to be running."""
    called: list = []
    monkeypatch.setattr(rec, "inspect_container",
                        lambda name: called.append(name) or None)
    monkeypatch.setattr(rec, "list_engine_containers", lambda: [])
    app = _app(_config(distributed_mode="solo", peer_ips=[]))
    assert asyncio.run(rec.adopt_running_engines(app)) == []
    assert called == []


def test_adoption_reuses_the_boot_engine_for_the_container_it_just_launched(monkeypatch):
    """A head that booted from config in THIS process keeps its own backend."""
    monkeypatch.setattr(rec, "inspect_container", lambda name: _inspect(_mp_head_argv()))
    monkeypatch.setattr(rec, "list_engine_containers", lambda: ["ainode-vllm-head"])
    config = _config()
    boot = _FakeBackend(config)
    _FakeBackend.built = []
    app = _app(config, engine=boot)
    asyncio.run(rec.adopt_running_engines(app))

    inst = app["instances"].by_port(8000)
    assert inst.backend is boot
    assert _FakeBackend.built == [], "no second backend was built for a live handle"


def test_adoption_points_the_back_compat_engine_at_the_live_container(monkeypatch):
    """A boot with no engine handle (the container outlived the process) gets one."""
    monkeypatch.setattr(rec, "inspect_container", lambda name: _inspect(_mp_head_argv()))
    monkeypatch.setattr(rec, "list_engine_containers", lambda: ["ainode-vllm-head"])
    app = _app(engine=None)
    asyncio.run(rec.adopt_running_engines(app))

    assert app["engine"] is not None
    assert app["engine"].config.model == MODEL
    assert app["engine"].config.distributed_mode == "head"
    assert app["engine"].config.peer_ips == [PEER]
    # Stamped from the container's own start time so is_running() asks docker.
    assert app["engine"]._launched_at is not None


def test_an_orphan_stacked_container_is_adopted_and_written_to_the_manifest(monkeypatch):
    """A crash between a stacked launch and the manifest write is repaired."""
    stacked = "ainode-vllm-node-solo-8001"
    other = "cerebras/Ornith-1.5-35B-A3B-NVFP4"

    def _inspect_by_name(name):
        if name == stacked:
            return _inspect(_stacked_argv(8001, other), cid="stacked01",
                            image="vllm/vllm-openai:v0.27.1")
        return None

    monkeypatch.setattr(rec, "inspect_container", _inspect_by_name)
    monkeypatch.setattr(rec, "list_engine_containers",
                        lambda: ["ainode-vllm-node-solo", stacked])
    app = _app(_config(distributed_mode="solo", peer_ips=[], model=None))
    adopted = asyncio.run(rec.adopt_running_engines(app))

    assert [a["kind"] for a in adopted] == ["stacked"]
    inst = app["instances"].by_port(8001)
    assert inst.record.model == other
    assert inst.record.adopted is True
    assert inst.record.tensor_parallel_size == 1
    # The gmu the container is actually running, not the node default.
    assert inst.backend.config.gpu_memory_utilization == 0.35
    # And it is in the manifest now, so the NEXT restart replays it.
    manifest = json.loads((mr._manifest_path()).read_text())
    assert [e["model"] for e in manifest["instances"]] == [other]


def test_an_adopted_container_is_not_swept(monkeypatch):
    """The startup sweep must not remove the engine adoption just kept (#179)."""
    monkeypatch.setattr(rec, "inspect_container",
                        lambda name: _inspect(_stacked_argv(8001, MODEL), cid="keepme01")
                        if name == "ainode-vllm-node-solo-8001" else None)
    monkeypatch.setattr(rec, "list_engine_containers",
                        lambda: ["ainode-vllm-node-solo-8001"])
    app = _app(_config(distributed_mode="solo", peer_ips=[], model=None))
    asyncio.run(rec.adopt_running_engines(app))
    assert rec.adopted_container_ids() == {"keepme01"}

    removed: list = []

    class _Proc:
        stdout = "keepme01\nsweepme02\n"

    def _run(argv, **kwargs):
        if argv[:3] == ["docker", "rm", "-f"]:
            removed.extend(argv[3:])
        return _Proc()

    monkeypatch.setattr("subprocess.run", _run)
    ids = mr._remove_engine_containers(include_primary=False)
    assert ids == ["sweepme02"]
    assert removed == ["sweepme02"]


def test_the_orphan_poll_does_not_wait_for_an_adopted_container(monkeypatch):
    monkeypatch.setattr(rec, "inspect_container",
                        lambda name: _inspect(_stacked_argv(8001, MODEL), cid="keepme01")
                        if name == "ainode-vllm-node-solo-8001" else None)
    monkeypatch.setattr(rec, "list_engine_containers",
                        lambda: ["ainode-vllm-node-solo-8001"])
    asyncio.run(rec.adopt_running_engines(
        _app(_config(distributed_mode="solo", peer_ips=[], model=None))))
    monkeypatch.setattr(mr, "_engine_container_ids",
                        lambda include_primary: ["keepme01"])
    assert mr._orphan_engine_ids() == []


# --------------------------------------------------------- the argv it reads --

def test_the_container_argv_is_what_names_the_shape():
    shape = rec.container_shape(_inspect(_mp_head_argv(port=8001, tp=4)))
    assert shape["running"] is True
    assert shape["model"] == MODEL            # --served-model-name, not the mount path
    assert shape["api_port"] == 8001
    assert shape["tensor_parallel_size"] == 4
    assert shape["nnodes"] == 4
    assert shape["distributed_executor"] == "mp"
    assert shape["gpu_memory_utilization"] == 0.8
    assert shape["max_model_len"] == 1048576
    assert shape["kv_cache_dtype"] == "nvfp4_ds_mla"
    assert shape["started_at"] is not None


def test_a_ray_head_container_is_not_read_as_mp():
    """The Ray shape runs vllm through a docker exec and states no rendezvous."""
    argv = ["ray", "start", "--head", "--port", "6379", "--block"]
    shape = rec.container_shape(_inspect(argv))
    assert "distributed_executor" not in shape
    assert shape["nnodes"] == 0


def test_an_unparseable_container_leaves_the_shape_empty():
    shape = rec.container_shape({})
    assert shape["running"] is False
    assert shape["model"] == ""
    assert shape["api_port"] == 0


# ------------------------------------------------------- the record lifecycle --

def test_the_record_round_trips_and_clears():
    config = _config()
    written = rec.write_distributed_record(
        config, model=MODEL, api_port=8000, peer_ips=[PEER],
        tensor_parallel_size=2, distributed_executor="mp",
        instance_id="25033c02:" + MODEL, container="ainode-vllm-head",
        overrides={"engine_image": IMAGE})
    assert written["status"] == rec.SERVING
    assert rec.load_distributed_record()["container"] == "ainode-vllm-head"
    assert rec.record_path().name == "distributed.json"

    rec.clear_distributed_record()
    assert rec.load_distributed_record() is None
    # Clearing something already gone is not an error.
    rec.clear_distributed_record()


def test_a_junk_record_reads_as_no_record():
    rec.record_path().write_text("{not json")
    assert rec.load_distributed_record() is None
    rec.record_path().write_text('{"peer_ips": []}')
    assert rec.load_distributed_record() is None


def test_unloading_the_distributed_instance_removes_the_record(monkeypatch):
    """The other half of the write: an unloaded shape is not replayed (#179)."""
    config = _config()
    rec.write_distributed_record(
        config, model=MODEL, api_port=8000, peer_ips=[PEER],
        tensor_parallel_size=2, distributed_executor="mp",
        instance_id="head:" + MODEL, container="ainode-vllm-head")
    manager = InstanceManager(base_port=8000)
    backend = _FakeBackend(config)
    manager.add(InstanceRecord(instance_id="head:" + MODEL, model=MODEL,
                               peer_ips=[PEER], api_port=8000,
                               tensor_parallel_size=2, distributed_executor="mp"),
                backend)
    app = _app(config, engine=backend, manager=manager)

    class _Req:
        def __init__(self):
            self.app = app
            self.query = {}

        async def json(self):
            return {"model": MODEL, "api_port": 8000}

    resp = asyncio.run(mr.handle_model_unload(_Req()))
    assert resp.status == 200
    assert backend.stopped is True
    assert rec.load_distributed_record() is None


def test_a_solo_load_of_the_heads_own_model_is_refused_not_a_downgrade(monkeypatch):
    """Adoption makes the head visible to the load path, so the downgrade is loud.

    Re-loading a model that is up replaces that instance, which for a multi-node
    engine would mean stopping it and bringing a frontier MoE back on one node.
    """
    monkeypatch.setattr(rec, "inspect_container", lambda name: _inspect(_mp_head_argv()))
    monkeypatch.setattr(rec, "list_engine_containers", lambda: ["ainode-vllm-head"])
    config = _config()
    app = _app(config)
    asyncio.run(rec.adopt_running_engines(app))

    result = mr.append_solo_instance(app, MODEL, 0.4, persist=False)
    assert result["ok"] is False
    assert result["status"] == 409
    assert "DISTRIBUTED" in result["error"]
    assert PEER in result["error"]
    # The head is still there, and nothing was stopped.
    inst = app["instances"].by_port(8000)
    assert inst is not None
    assert inst.backend.stopped is False


def test_unloading_a_neighbour_leaves_the_record_alone():
    config = _config()
    rec.write_distributed_record(
        config, model=MODEL, api_port=8000, peer_ips=[PEER],
        tensor_parallel_size=2, distributed_executor="mp",
        instance_id="head:" + MODEL, container="ainode-vllm-head")
    assert mr.forget_distributed_instance("other/model", 8001) is False
    assert mr.forget_distributed_instance(MODEL, 8001) is False
    assert rec.load_distributed_record() is not None
    assert mr.forget_distributed_instance(MODEL, 8000) is True
    assert rec.load_distributed_record() is None


# -------------------------------------------------------- the replay policy --

def _record(config, **kw) -> dict:
    fields = dict(model=MODEL, api_port=8000, peer_ips=[PEER],
                  tensor_parallel_size=2, distributed_executor="mp",
                  instance_id="head:" + MODEL, container="ainode-vllm-head",
                  overrides={"engine_image": IMAGE})
    fields.update(kw)
    return rec.write_distributed_record(config, **fields)


def test_replay_relaunches_the_recorded_shape_when_every_peer_answers(monkeypatch):
    config = _config()
    _record(config)
    probed: list = []
    monkeypatch.setattr(rec, "probe_peer",
                        lambda ip, user, timeout=20.0: probed.append((ip, user))
                        or (True, "28.0.1"))
    app = _app(config)
    outcome = asyncio.run(rec.replay_distributed_if_needed(app))

    assert outcome["action"] == "relaunched"
    assert probed == [(PEER, "sem")]
    inst = app["instances"].by_port(8000)
    assert inst.record.model == MODEL
    assert inst.record.tensor_parallel_size == 2
    assert inst.record.adopted is False, "this process launched it"
    assert inst.backend.launched is True
    # The recorded recipe reaches the relaunch.
    assert inst.backend.config.engine_image == IMAGE
    assert inst.backend.config.distributed_executor == "mp"
    assert inst.backend.config.peer_ips == [PEER]
    assert rec.load_distributed_record()["status"] == rec.SERVING


def test_replay_marks_the_record_degraded_when_a_peer_is_down(monkeypatch, caplog):
    config = _config()
    _record(config)
    monkeypatch.setattr(
        rec, "probe_peer",
        lambda ip, user, timeout=20.0: (False, "ssh: connect to host "
                                              f"{ip} port 22: No route to host"))
    app = _app(config)
    with caplog.at_level("WARNING"):
        outcome = asyncio.run(rec.replay_distributed_if_needed(app))

    assert outcome["action"] == "degraded"
    assert app["instances"].is_empty(), "nothing was launched"
    record = rec.load_distributed_record()
    assert record["status"] == rec.DEGRADED
    assert record["degraded_peers"][0]["peer_ip"] == PEER
    assert "No route to host" in record["degraded_reason"]
    assert PEER in record["degraded_reason"]
    assert "DEGRADED" in caplog.text
    # And it is reportable.
    surfaced = rec.degraded_instances()
    assert surfaced[0]["model"] == MODEL
    assert surfaced[0]["tensor_parallel_size"] == 2
    assert surfaced[0]["peers_unreachable"][0]["peer_ip"] == PEER


def test_a_relaunch_that_fails_is_degraded_not_retried(monkeypatch):
    config = _config()
    _record(config)
    monkeypatch.setattr(rec, "probe_peer", lambda ip, user, timeout=20.0: (True, "28.0.1"))

    class _Dead(_FakeBackend):
        def start_distributed(self):
            raise RuntimeError("no fabric IP on enP2p1s0f1np1")

    monkeypatch.setattr("ainode.engine.backends.get_backend", _Dead)
    outcome = asyncio.run(rec.replay_distributed_if_needed(_app(config)))
    assert outcome["action"] == "degraded"
    assert "no fabric IP" in rec.load_distributed_record()["degraded_reason"]


def test_replay_is_one_attempt_per_process(monkeypatch):
    """Never a loop: a shape that needs a human is not fixed by retrying."""
    config = _config()
    _record(config)
    calls: list = []
    monkeypatch.setattr(rec, "probe_peer",
                        lambda ip, user, timeout=20.0: calls.append(ip) or (False, "down"))
    app = _app(config)
    assert asyncio.run(rec.replay_distributed_if_needed(app))["action"] == "degraded"
    assert asyncio.run(rec.replay_distributed_if_needed(app))["action"] == "none"
    assert calls == [PEER]


def test_an_adopted_container_makes_the_replay_a_no_op(monkeypatch):
    """The container is alive, so there is nothing to launch and no peer to ask."""
    config = _config()
    _record(config)
    monkeypatch.setattr(rec, "inspect_container", lambda name: _inspect(_mp_head_argv()))
    monkeypatch.setattr(rec, "list_engine_containers", lambda: ["ainode-vllm-head"])
    probed: list = []
    monkeypatch.setattr(rec, "probe_peer",
                        lambda ip, user, timeout=20.0: probed.append(ip) or (True, "ok"))
    app = _app(config)
    asyncio.run(rec.adopt_running_engines(app))
    outcome = asyncio.run(rec.replay_distributed_if_needed(app))

    assert outcome["action"] == "adopted"
    assert probed == []
    assert all(not b.launched for b in _FakeBackend.built)


def test_a_degraded_record_clears_once_the_shape_is_back(monkeypatch):
    config = _config()
    record = _record(config)
    rec.mark_record_degraded(record, "peer down", [{"peer_ip": PEER, "answer": "down"}])
    assert rec.degraded_instances()[0]["reason"] == "peer down"
    monkeypatch.setattr(rec, "inspect_container", lambda name: _inspect(_mp_head_argv()))
    monkeypatch.setattr(rec, "list_engine_containers", lambda: ["ainode-vllm-head"])
    app = _app(config)
    asyncio.run(rec.adopt_running_engines(app))
    asyncio.run(rec.replay_distributed_if_needed(app))
    assert rec.load_distributed_record()["status"] == rec.SERVING
    assert rec.degraded_instances() == []


def test_a_member_node_adopts_nothing_and_replays_nothing(monkeypatch, caplog):
    """A member's containers belong to whichever head placed them over ssh."""
    config = _config(distributed_mode="member")
    _record(_config())
    asked: list = []
    monkeypatch.setattr(rec, "inspect_container",
                        lambda name: asked.append(name) or None)
    monkeypatch.setattr(rec, "list_engine_containers", lambda: [])
    monkeypatch.setattr(rec, "probe_peer",
                        lambda ip, user, timeout=20.0: (True, "28.0.1"))
    app = _app(config)
    assert asyncio.run(rec.adopt_running_engines(app)) == []
    with caplog.at_level("INFO"):
        outcome = asyncio.run(rec.replay_distributed_if_needed(app))
    assert outcome == {"action": "none", "reason": "node is a member"}
    assert asked == []
    assert "this node is a member now" in caplog.text


def test_a_replay_holds_the_launch_slot_until_the_engine_binds(monkeypatch):
    """Same contract as every other launch path: the slot is released on BIND (#96)."""
    config = _config()
    _record(config)
    monkeypatch.setattr(rec, "probe_peer", lambda ip, user, timeout=20.0: (True, "ok"))
    held: list = []

    async def _hold(app, port, backend, label):
        held.append((port, label, backend))
        mr.release_launch_slot()
        return True

    monkeypatch.setattr(mr, "hold_launch_slot_until_bound", _hold)

    async def _run():
        app = _app(config)
        outcome = await rec.replay_distributed_if_needed(app)
        # The slot is handed to the bind watch, which is a task: let it run.
        await asyncio.sleep(0)
        return outcome, app

    outcome, app = asyncio.run(_run())
    assert outcome["action"] == "relaunched"
    assert held and held[0][0] == 8000
    assert held[0][1].startswith("distributed replay ")
    assert held[0][2] is app["instances"].by_port(8000).backend
    assert mr.launch_owner() is None


def test_no_record_means_no_replay_and_no_degraded_report():
    app = _app()
    assert asyncio.run(rec.replay_distributed_if_needed(app))["action"] == "none"
    assert rec.degraded_instances() == []


def test_the_peer_probe_is_the_ssh_the_launch_depends_on(monkeypatch):
    """Same ssh options the head uses to place a peer container, asked as a question."""
    seen: dict = {}

    class _Done:
        returncode = 0
        stdout = "28.0.1\n"
        stderr = ""

    def _run(argv, **kwargs):
        seen["argv"] = argv
        return _Done()

    monkeypatch.setattr("subprocess.run", _run)
    ok, answer = rec.probe_peer(PEER, "sem")
    assert ok is True
    assert answer == "28.0.1"
    assert seen["argv"][0] == "ssh"
    assert "BatchMode=yes" in seen["argv"]
    assert "ConnectTimeout=10" in seen["argv"]
    assert f"sem@{PEER}" in seen["argv"]
    assert "docker version" in seen["argv"][-1]


def test_a_peer_that_answers_without_docker_is_not_reachable(monkeypatch):
    class _Failed:
        returncode = 127
        stdout = ""
        stderr = "bash: docker: command not found"

    monkeypatch.setattr("subprocess.run", lambda argv, **kw: _Failed())
    ok, answer = rec.probe_peer(PEER, "sem")
    assert ok is False
    assert "command not found" in answer


def test_the_probe_reports_a_timeout_instead_of_raising(monkeypatch):
    def _boom(argv, **kwargs):
        raise TimeoutError("timed out after 20s")

    monkeypatch.setattr("subprocess.run", _boom)
    ok, answer = rec.probe_peer(PEER, "sem")
    assert ok is False
    assert "timed out" in answer


# ------------------------------------------------------------- the surfaces --

def test_status_reports_a_degraded_instance(monkeypatch, tmp_path):
    """/api/status carries degraded_instances, empty on a healthy node."""
    import pytest_asyncio  # noqa: F401  (the aiohttp client needs the plugin)
    from aiohttp.test_utils import TestClient, TestServer

    from ainode.api.server import create_app

    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    config = _config()
    config._skip_replay = True  # no background replay in a test
    app = create_app(config=config, engine=None)

    async def _get():
        async with TestClient(TestServer(app)) as client:
            healthy = await (await client.get("/api/status")).json()
            record = _record(config)
            rec.mark_record_degraded(
                record, f"{MODEL} TP=2 is not running here and 1 of 1 peer(s) did "
                        f"not answer: {PEER} answered 'No route to host'",
                [{"peer_ip": PEER, "answer": "No route to host"}])
            degraded = await (await client.get("/api/status")).json()
            return healthy, degraded

    healthy, degraded = asyncio.run(_get())
    assert healthy["degraded_instances"] == []
    assert len(degraded["degraded_instances"]) == 1
    entry = degraded["degraded_instances"][0]
    assert entry["model"] == MODEL
    assert entry["peers_unreachable"] == [{"peer_ip": PEER, "answer": "No route to host"}]
    assert "No route to host" in entry["reason"]


def test_the_dashboard_draws_the_degraded_banner():
    """One hunk in app.js next to the update badge, fed by /api/status."""
    from ainode.web.serve import STATIC_DIR
    js = (STATIC_DIR / "js" / "app.js").read_text()
    assert "renderDegradedBanner() {" in js
    body = js.split("renderDegradedBanner() {", 1)[1].split("\n  },", 1)[0]
    assert "degraded_instances" in body
    assert "peers_unreachable" in body
    assert "tensor_parallel_size" in body
    # Rebuilt from the poll, so it clears itself when the shape comes back.
    assert "this.renderDegradedBanner();" in js
    css = (STATIC_DIR / "css" / "style.css").read_text()
    assert ".degraded-banner {" in css
