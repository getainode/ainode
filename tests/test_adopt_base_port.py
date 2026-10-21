"""Boot adoption and the destructive sweep, with synthetic Docker observations."""

import asyncio
import copy
import io
import json
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from ainode.core.config import NodeConfig
from ainode.engine import reconcile as rec
from ainode.engine.backends.nvidia import NVIDIA_VLLM_IMAGE
from ainode.engine.instance_manager import InstanceManager
from ainode.models import api_routes as mr


@pytest.fixture
def observed_stack(monkeypatch, tmp_path):
    config = NodeConfig(model=None, api_port=8000, node_id="node-a",
                        engine_backend="nvidia", distributed_mode="solo")
    manifest = [
        {"model": "example/Jeb-9B", "gpu_memory_utilization": 0.3,
         "max_model_len": 32768, "engine_image": ""},
        {"model": "example/Chat", "gpu_memory_utilization": 0.3},
        {"model": "example/Embedding", "gpu_memory_utilization": 0.1},
    ]
    containers = {}
    for index, entry in enumerate(manifest):
        port = 8000 + index
        name = rec.PRIMARY_ENGINE_NAME if index == 0 else rec.stacked_container_name(port)
        containers[name] = {
            "Id": f"{index + 1:x}" * 64, "Name": "/" + name,
            "Config": {"Image": NVIDIA_VLLM_IMAGE, "Entrypoint": ["vllm"],
                       "Cmd": ["serve", entry["model"], "--port", str(port),
                               "--gpu-memory-utilization", str(entry["gpu_memory_utilization"]),
                               "--max-model-len", "32768"]},
            "State": {"Running": True, "Status": "running",
                      "StartedAt": "2026-09-01T12:00:00Z"},
        }
    removed = []
    calls = []
    model_observations = {8000 + i: [e["model"]] for i, e in enumerate(manifest)}

    def selected(argv):
        prefixes = [arg.removeprefix("name=") for arg in argv if arg.startswith("name=")]
        return {n: c for n, c in containers.items() if any(p in n for p in prefixes)}

    def docker_read(argv, **kwargs):
        calls.append(list(argv))
        assert argv[0] == "docker", argv
        if argv[1] == "inspect":
            return json.dumps([containers[argv[2]]]) if argv[2] in containers else "[]"
        assert argv[1] == "ps", argv
        picked = selected(argv)
        if "{{.Names}}" in argv:
            return "\n".join(picked)
        return "\n".join(c["Id"][:12] for c in picked.values())

    def docker_run(argv, **kwargs):
        if argv[:3] == ["docker", "rm", "-f"]:
            calls.append(list(argv))
            removed.extend(argv[3:])
            for name, info in list(containers.items()):
                if any(info["Id"].startswith(cid) for cid in argv[3:]):
                    del containers[name]
            return SimpleNamespace(stdout="", returncode=0)
        return SimpleNamespace(stdout=docker_read(argv), returncode=0)

    class Backend:
        def __init__(self, config, instance_id=""):
            self.config = config
            self.instance_id = instance_id

        def start(self):
            pytest.fail("adoption relaunched a healthy engine")

        def stop(self):
            pytest.fail("adoption stopped a healthy engine")

    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(rec, "load_distributed_record", lambda: None)
    monkeypatch.setattr(mr, "load_instance_manifest", lambda: copy.deepcopy(manifest))
    monkeypatch.setattr(rec.subprocess, "check_output", docker_read)
    monkeypatch.setattr(rec.subprocess, "run", docker_run)
    # Override the suite's no-Docker seams with the actual read-only transports.
    monkeypatch.setattr(rec, "inspect_container", _REAL_INSPECT)
    monkeypatch.setattr(rec, "list_engine_containers", _REAL_LIST)
    monkeypatch.setattr(rec, "port_health", lambda port: True)
    monkeypatch.setattr(rec, "served_models", lambda port: model_observations.get(port, []))
    monkeypatch.setattr(rec, "port_serving", lambda port: True)
    monkeypatch.setattr("ainode.engine.backends.get_backend", Backend)
    monkeypatch.setattr(mr, "_BOOT_SWEEP_DONE", False)
    monkeypatch.setattr(mr.time, "sleep", lambda seconds: pytest.fail("sweep waited"))
    app = {"config": config, "engine": None, "instances": InstanceManager(8000)}
    return SimpleNamespace(config=config, manifest=manifest, containers=containers,
                           removed=removed, calls=calls, app=app)


# Capture before conftest's per-test safety patch, then replace only with fakes.
_REAL_INSPECT = rec.inspect_container
_REAL_LIST = rec.list_engine_containers
_REAL_HEALTH = rec.port_health
_REAL_MODELS = rec.served_models
_REAL_SERVING = rec.port_serving


def test_boot_sweep_and_replay_keep_manifest_stack_on_base_port(observed_stack, monkeypatch):
    observed = observed_stack
    original = copy.deepcopy(observed.containers)
    decision = rec.adopt_boot_engines(observed.config)
    swept = mr.sweep_engines_before_boot()
    assert swept == [], f"boot removed healthy manifest-backed engines: {observed.removed}"
    assert observed.containers == original
    assert decision["primary"] is None
    assert {e["api_port"] for e in decision["stacked"]} == {8000, 8001, 8002}
    adopted = asyncio.run(rec.adopt_running_engines(observed.app))
    assert [e["kind"] for e in adopted] == ["stacked"] * 3
    inst = observed.app["instances"].by_port(8000)
    assert inst.record.adopted and inst.record.status == "serving"
    assert inst.record.model == "example/Jeb-9B"
    assert inst.backend.instance_id == ""
    assert inst.backend.config.gpu_memory_utilization == 0.3
    assert inst.backend.config.max_model_len == 32768
    assert inst.backend.config.engine_image == ""
    assert observed.config.model is None
    assert observed.app["engine"] is None
    warmed = []
    monkeypatch.setattr(mr, "_schedule_decision_warmup",
                        lambda app, port, backend: warmed.append(port))
    mr._warm_adopted_stacked(observed.app, observed.config,
                             observed.app["instances"].records())
    assert warmed == [8000, 8001, 8002]

    def no_launch(*args, **kwargs):
        pytest.fail("replay relaunched a healthy adopted instance")

    monkeypatch.setattr(mr, "append_solo_instance", no_launch)
    async def adopted_primary_ready(app, config, loading=1):
        # Every synthetic stack was positively adopted above. The real port
        # waiter would otherwise probe a host engine on :8000 for 300 seconds.
        assert app is observed.app and config is observed.config
        assert all(instance.record.adopted for instance in app["instances"].instances())
        return True

    monkeypatch.setattr(mr, "_await_primary_bind", adopted_primary_ready)
    asyncio.run(mr._replay_serialized(observed.app, observed.config, observed.manifest))
    assert mr.sweep_engines_before_boot() == []
    assert observed.removed == []


@pytest.mark.parametrize("malformed", [
    None, {"api_port": 8000}, {"api_port": 8000, "serving": False},
    {"api_port": 8000, "serving": "false"},
    {"api_port": 8000, "serving": "true"},
    {"api_port": 8000, "serving": 1},
    {"api_port": "not-a-port", "serving": True},
    {"serving": True}, {"api_port": [], "serving": True},
    {"api_port": float("inf"), "serving": True},
])
def test_startup_replay_schedules_warmup_for_preserved_stacks(
        observed_stack, monkeypatch, malformed):
    from ainode.api import decide

    observed = observed_stack
    original = copy.deepcopy(observed.containers)
    config_before = copy.deepcopy(vars(observed.config))
    warmed, launched, requests = [], [], []

    def http_read(url, **kwargs):
        parsed = urlsplit(url)
        assert parsed.hostname == "localhost"
        assert parsed.port in (8000, 8001, 8002)
        assert parsed.path in ("/health", "/v1/models")
        requests.append((parsed.port, parsed.path))
        model = observed.manifest[parsed.port - 8000]["model"]
        payload = {} if parsed.path == "/health" else {"data": [{"id": model}]}
        response = io.BytesIO(json.dumps(payload).encode())
        response.status = 200
        response.getcode = lambda: 200
        return response

    def no_launch(*args, **kwargs):
        launched.append((args, kwargs))
        pytest.fail("startup replay relaunched a preserved stack")

    async def no_distributed(app):
        return {"action": "none"}

    # Run the real producer and caller. Only Docker/HTTP observations, launch
    # effects and the final scheduler are faked, so the handoff retains its shape.
    monkeypatch.setattr("urllib.request.urlopen", http_read)
    monkeypatch.setattr(rec, "port_health", _REAL_HEALTH)
    monkeypatch.setattr(rec, "served_models", _REAL_MODELS)
    monkeypatch.setattr(rec, "port_serving", _REAL_SERVING)
    async def adopt_with_malformed_summary(app):
        rows = await rec.adopt_running_engines(app)
        # Keep the real producer/caller chain, injecting a corrupt summary at
        # its boundary. It must neither warm nor block the genuine rows.
        if malformed is not None:
            rows.insert(0, malformed)
        return rows

    monkeypatch.setattr(mr, "adopt_running_engines", adopt_with_malformed_summary)
    monkeypatch.setattr(mr, "replay_distributed_if_needed", no_distributed)
    monkeypatch.setattr(mr, "_REPLAY_SETTLE_SECONDS", 0)
    monkeypatch.setattr(mr, "append_solo_instance", no_launch)
    monkeypatch.setattr(decide, "schedule_decision_warmup",
                        lambda app, port, backend: warmed.append((app, port, backend)))

    decision = rec.adopt_boot_engines(observed.config)
    assert decision["primary"] is None
    assert {row["api_port"] for row in decision["stacked"]} == {8000, 8001, 8002}
    assert mr.sweep_engines_before_boot() == []
    asyncio.run(mr.replay_instances_on_startup(observed.app))

    manager = observed.app["instances"]
    assert len(manager.records()) == 3
    assert all(record.adopted and record.status == "serving" for record in manager.records())
    assert manager.by_port(8000).backend.instance_id == ""
    assert manager.by_port(8001).backend.instance_id == "8001"
    assert manager.by_port(8002).backend.instance_id == "8002"
    assert warmed == [(observed.app, port, manager.by_port(port).backend)
                      for port in (8000, 8001, 8002)]
    assert vars(observed.config) == config_before
    assert observed.config.model is None and observed.app["engine"] is None
    assert observed.containers == original
    assert observed.removed == [] and launched == []
    assert {(port, path) for port in (8000, 8001, 8002)
            for path in ("/health", "/v1/models")} <= set(requests)


def test_adopted_summary_coerces_port_once(observed_stack, monkeypatch):
    observed = observed_stack
    asyncio.run(rec.adopt_running_engines(observed.app))
    warmed = []

    class Port:
        calls = 0

        def __int__(self):
            self.calls += 1
            assert self.calls == 1, "each row must normalize its port only once"
            return 8001

    port = Port()
    monkeypatch.setattr(mr, "_schedule_decision_warmup",
                        lambda app, port, backend: warmed.append(port))
    mr._warm_adopted_stacked(observed.app, observed.config,
                             [{"api_port": port, "serving": True},
                              {"api_port": "8002", "serving": True}])
    assert port.calls == 1
    assert warmed == [8001, 8002]


@pytest.mark.parametrize("rejection", [
    "missing_manifest", "wrong_image", "wrong_model", "wrong_width", "wrong_port",
    "exited", "unhealthy", "unbound", "wrong_served_model", "start_clean",
    "changed_primary", "head", "member", "foreign_name",
])
def test_base_port_requires_saved_recipe_identity_and_health(
        observed_stack, monkeypatch, rejection):
    observed = observed_stack
    base = observed.containers[rec.PRIMARY_ENGINE_NAME]
    if rejection == "missing_manifest":
        observed.manifest.pop(0)
    elif rejection == "wrong_image":
        base["Config"]["Image"] = "example/other-engine:v1"
    elif rejection == "wrong_model":
        base["Config"]["Cmd"][1] = "example/Unrecorded"
    elif rejection == "wrong_width":
        base["Config"]["Cmd"] += ["--tensor-parallel-size", "2"]
    elif rejection == "wrong_port":
        base["Config"]["Cmd"][3] = "9000"
    elif rejection == "exited":
        base["State"].update(Running=False, Status="exited")
    elif rejection == "unhealthy":
        monkeypatch.setattr(rec, "port_health", lambda port: port != 8000)
    elif rejection == "unbound":
        monkeypatch.setattr(rec, "served_models", lambda port: [])
    elif rejection == "wrong_served_model":
        monkeypatch.setattr(rec, "served_models", lambda port: ["example/Other"])
    elif rejection == "start_clean":
        observed.config._skip_replay = True
    elif rejection == "changed_primary":
        observed.config.model = "example/NewPrimary"
    elif rejection in ("head", "member"):
        observed.config.distributed_mode = rejection
    elif rejection == "foreign_name":
        observed.containers["foreign-engine"] = observed.containers.pop(rec.PRIMARY_ENGINE_NAME)

    decision = rec.adopt_boot_engines(observed.config)
    assert decision["primary"] is None
    assert base["Id"] not in rec.adopted_container_ids()
    # A failed sweep must not let asynchronous adoption bypass the same checks.
    asyncio.run(rec.adopt_running_engines(observed.app))
    assert observed.app["instances"].by_port(8000) is None
    assert base["Id"] not in rec.adopted_container_ids()
    assert observed.app["engine"] is None
    mr.sweep_engines_before_boot()
    if rejection == "foreign_name":
        assert base in observed.containers.values()
    else:
        assert base["Id"][:12] in observed.removed


def test_configured_primary_keeps_its_role_with_manifest_entry(observed_stack):
    observed = observed_stack
    observed.config.model = observed.manifest[0]["model"]
    decision = rec.adopt_boot_engines(observed.config)
    assert decision["primary"]["api_port"] == 8000
    assert {e["api_port"] for e in decision["stacked"]} == {8001, 8002}
    assert mr.sweep_engines_before_boot() == []
    adopted = asyncio.run(rec.adopt_running_engines(observed.app))
    assert [e["kind"] for e in adopted] == ["primary", "stacked", "stacked"]
    assert observed.app["engine"] is observed.app["instances"].by_port(8000).backend
    assert observed.config.model == observed.manifest[0]["model"]


def test_suffixed_name_must_match_the_engine_port(observed_stack):
    observed = observed_stack
    # Exercise orphan recovery, where no saved recipe can reject the mismatch.
    observed.manifest[:] = [e for e in observed.manifest if e["model"] != "example/Chat"]
    name = rec.stacked_container_name(8001)
    info = observed.containers[name]
    info["Config"]["Cmd"][3] = "9001"
    rec.adopt_boot_engines(observed.config)
    asyncio.run(rec.adopt_running_engines(observed.app))
    assert info["Id"] not in rec.adopted_container_ids()
    assert observed.app["instances"].by_port(8001) is None
    assert observed.app["instances"].by_port(9001) is None
    mr.sweep_engines_before_boot()
    assert observed.removed == [info["Id"][:12]]


def checker_payload(observed):
    return {
        "config": {"model": observed.config.model, "api_port": 8000,
                   "engine_backend": "nvidia", "distributed_mode": "solo"},
        "manifest": copy.deepcopy(observed.manifest),
        "containers": [
            {"name": name, "id": info["Id"], "image": info["Config"]["Image"],
             "argv": rec.container_argv(info), "running": info["State"]["Running"],
             "health": True, "models": [rec.container_shape(info)["model"]]}
            for name, info in observed.containers.items()
        ],
    }


def test_checker_uses_production_gates_without_io(observed_stack, monkeypatch):
    from ainode.engine.adoption_check import check_observations

    payload = checker_payload(observed_stack)

    def forbidden(*args, **kwargs):
        pytest.fail("checker attempted external I/O")

    monkeypatch.setattr(rec.subprocess, "run", forbidden)
    monkeypatch.setattr(rec.subprocess, "check_output", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("pathlib.Path.open", forbidden)
    assert check_observations(payload) == {
        "protected_count": 3, "all_running_owned_survive": True}
    payload["containers"][0]["health"] = False
    assert check_observations(payload) == {
        "protected_count": 2, "all_running_owned_survive": False}
    assert rec.adopted_container_ids() == set()
    assert rec.boot_decision() == {}


@pytest.mark.parametrize("change", ["image", "width", "models", "manifest", "primary",
                                    "start_clean", "member", "missing_health"])
def test_checker_reports_unprotected_engines(observed_stack, change):
    from ainode.engine.adoption_check import check_observations

    payload = checker_payload(observed_stack)
    if change == "image":
        payload["containers"][0]["image"] = "example/new:v2"
    elif change == "width":
        payload["containers"][0]["argv"] += ["--tensor-parallel-size", "2"]
    elif change == "models":
        payload["containers"][0]["models"] = []
    elif change == "manifest":
        payload["manifest"] = payload["manifest"][1:]
    elif change == "primary":
        payload["config"]["model"] = "example/NewPrimary"
    elif change == "start_clean":
        payload["start_clean"] = True
    elif change == "member":
        payload["config"]["distributed_mode"] = "member"
    elif change == "missing_health":
        del payload["containers"][0]["health"]
        with pytest.raises(ValueError):
            check_observations(payload)
        return
    assert check_observations(payload)["all_running_owned_survive"] is False


def test_checker_cli_prints_only_counts_and_verdict(observed_stack, monkeypatch, capsys):
    import io

    from ainode.engine import adoption_check

    payload = checker_payload(observed_stack)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert adoption_check.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "protected_count": 3, "all_running_owned_survive": True}
    payload["containers"][0]["health"] = False
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert adoption_check.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "protected_count": 2, "all_running_owned_survive": False}
    payload["config"]["cluster_secret"] = "synthetic-do-not-echo"
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert adoption_check.main() == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "protected_count": 0, "all_running_owned_survive": False}
    assert captured.err == ""
