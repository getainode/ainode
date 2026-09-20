"""Every number the dashboard shows is a measurement or says unknown.

The fleet audit of 2026-09-19 found seven numbers that were neither. They were
all confidently wrong, which is worse than blank: a user sizes a model against
them.

* A four-V100 host announced ONE 32 GB GPU, because the collector read device 0
  and the cluster counted one GPU per node (#163). The fleet's nine GPUs read as
  six and its total VRAM was short by 96 GB.
* ``available_vram_gb`` was a copy of ``total_vram_gb``, so the cluster was
  always empty however many models were loaded (#174).
* On a GB10 the "GPU memory used" figure was host RAM in use, page cache
  included, so every Spark sat at 84 to 100 percent forever (#175).
* GPU utilisation was 0 on every node, always, drawn as a real percentage: no
  way to tell an idle node from one whose driver does not expose the counter
  (#176).
* The Server view reported ``size_bytes: 0`` and ``quantization: null`` for
  every loaded model (#180).
* The topology crowned whichever node you opened the dashboard on, ignoring the
  election the server had already made (#203).
* ``/api/nodes`` said ``host: "localhost"`` for every node, so any link built
  from it pointed at the viewer's own machine (#178).

Fakes only: a fake NVML, a fake cluster, a snapshot directory in tmp_path.
"""

from __future__ import annotations

import json
import socket
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api import server
from ainode.api.server import engine_reserved_fraction
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import (
    MAX_ANNOUNCEMENT_BYTES,
    NodeAnnouncement,
    NodeStatus,
)
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.metrics.collector import MetricsCollector
from ainode.models import registry


# --------------------------------------------------------------------- fakes --

def _nvml(devices, driver="580.95.05"):
    """A fake NVML exposing *devices*, each ``(name, total_bytes, used_bytes,
    util_or_None, temp)``. A device with total 0 is a unified-memory part: NVML
    answers with zeros rather than raising, which is why a zero total has to be
    read as "the driver will not say" and not as a 0 GB card.
    """
    mod = MagicMock()
    handles = [object() for _ in devices]
    by_handle = {id(h): spec for h, spec in zip(handles, devices)}

    mod.nvmlDeviceGetCount.return_value = len(devices)

    def _handle(index):
        return handles[index]

    def _name(handle):
        return by_handle[id(handle)][0]

    def _memory(handle):
        _, total, used, _, _ = by_handle[id(handle)]
        return MagicMock(total=total, used=used, free=max(0, total - used))

    def _util(handle):
        value = by_handle[id(handle)][3]
        if value is None:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return MagicMock(gpu=value)

    def _temp(handle, _sensor):
        return by_handle[id(handle)][4]

    mod.nvmlDeviceGetHandleByIndex.side_effect = _handle
    mod.nvmlDeviceGetName.side_effect = _name
    mod.nvmlDeviceGetMemoryInfo.side_effect = _memory
    mod.nvmlDeviceGetUtilizationRates.side_effect = _util
    mod.nvmlDeviceGetTemperature.side_effect = _temp
    mod.nvmlDeviceGetCudaComputeCapability.return_value = (7, 0)
    mod.nvmlSystemGetDriverVersion.return_value = driver
    mod.nvmlSystemGetCudaDriverVersion_v2.return_value = 13000
    mod.NVML_TEMPERATURE_GPU = 0
    return mod


GB = 1024 ** 3

#: castor: four Tesla V100 32 GB, one of them holding a model.
CASTOR = [("Tesla V100-SXM2-32GB", 32 * GB, 20 * GB, 37, 29),
          ("Tesla V100-SXM2-32GB", 32 * GB, 0, 0, 28),
          ("Tesla V100-SXM2-32GB", 32 * GB, 0, 0, 30),
          ("Tesla V100-SXM2-32GB", 32 * GB, 0, 0, 29)]

#: A Spark: one GB10, unified memory, NVML reporting nothing but temperature.
SPARK = [("NVIDIA GB10", 0, 0, 0, 38)]


def _collector(devices, reserved=None, host_ram=(122 * GB, 120 * GB)):
    collector = MetricsCollector()
    if reserved is not None:
        collector.set_reservation_provider(lambda: reserved)
    total, used = host_ram
    fake_vm = MagicMock(total=total, used=used)
    with patch.dict("sys.modules", {"pynvml": _nvml(devices)}), \
         patch("psutil.virtual_memory", return_value=fake_vm):
        return collector.get_gpu_metrics()


def _node(**kw):
    defaults = dict(
        node_id="n", node_name="n", gpu_name="x", gpu_memory_gb=128.0,
        unified_memory=False, model="", status=NodeStatus.ONLINE,
        api_port=8000, web_port=3000, last_seen=time.time(),
        cluster_id="default", role="auto",
    )
    defaults.update(kw)
    return ClusterNode(**defaults)


@pytest.fixture
def config():
    # A port nothing listens on: the handlers probe localhost:<api_port>.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    return NodeConfig(node_id="spark3", node_name="Spark-3-DGX",
                      model="", api_port=free_port)


@pytest.fixture
def app(config):
    return server.create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


def _seed_cluster(app, nodes, local_id="spark3"):
    """Replace the app's cluster with a known set, local node first."""
    announcement = NodeAnnouncement(
        node_id=local_id, node_name="Spark-3-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=122.0, unified_memory=True, model="", status="serving",
        api_port=8000, web_port=3000, role="auto",
    )
    cluster = ClusterState(local_announcement=announcement)
    for node in nodes:
        cluster.add_node(node)
    app["cluster_state"] = cluster
    return cluster


# ----------------------------------------------------- #163 every device counts --

class TestMultiGpuNode:
    def test_four_devices_are_four_gpus_and_128_gb(self):
        """castro's four V100s summed, not device 0 read four times over."""
        metrics = _collector(CASTOR)

        assert metrics["gpu_count"] == 4
        assert metrics["memory_total_mb"] == 4 * 32 * 1024
        assert len(metrics["devices"]) == 4
        assert metrics["memory_kind"] == "dedicated"

    def test_memory_in_use_is_summed_across_devices(self):
        metrics = _collector(CASTOR)

        assert metrics["memory_used_mb"] == 20 * 1024
        assert metrics["memory_used_source"] == "nvml"

    def test_utilisation_is_the_mean_of_the_devices_that_answered(self):
        metrics = _collector(CASTOR)

        # 37 + 0 + 0 + 0 over four devices.
        assert metrics["utilization_percent"] == round(37 / 4)
        # The hottest device is the node's thermal story.
        assert metrics["temperature_c"] == 30

    def test_a_single_gpu_node_is_unchanged(self):
        metrics = _collector([("NVIDIA RTX 4090", 24 * GB, 8 * GB, 55, 61)])

        assert metrics["gpu_count"] == 1
        assert metrics["memory_total_mb"] == 24 * 1024
        assert metrics["memory_used_mb"] == 8 * 1024
        assert metrics["utilization_percent"] == 55

    def test_the_device_list_names_every_card(self):
        names = [d["name"] for d in _collector(CASTOR)["devices"]]

        assert names == ["Tesla V100-SXM2-32GB"] * 4

    def test_the_cluster_counts_gpus_not_nodes(self):
        cluster = ClusterState()
        cluster.add_node(_node(node_id="castor", gpu_count=4, gpu_memory_gb=128.0))
        cluster.add_node(_node(node_id="pollux", gpu_count=1, gpu_memory_gb=32.0))

        summary = cluster.cluster_summary()

        assert summary["total_nodes"] == 2
        assert summary["total_gpus"] == 5


# ----------------------------------------- #175 host RAM is not GPU memory --

class TestUnifiedMemory:
    def test_memory_is_reported_as_unified(self):
        metrics = _collector(SPARK)

        assert metrics["memory_kind"] == "unified"
        assert metrics["gpu_count"] == 1
        assert metrics["memory_total_mb"] == 122 * 1024

    def test_host_ram_in_use_is_not_published_as_vram(self):
        """The host has 120 of 122 GB in use, nearly all of it page cache. That
        is the number that had every Spark pinned at 98 percent (#175)."""
        metrics = _collector(SPARK, reserved=None)

        assert metrics["memory_used_mb"] is None
        assert metrics["system_memory_used_mb"] == 120 * 1024

    def test_usage_is_the_engines_reservation(self):
        metrics = _collector(SPARK, reserved=0.85)

        assert metrics["memory_used_mb"] == round(122 * 1024 * 0.85)
        assert metrics["memory_used_source"] == "engine_reservations"

    def test_an_idle_node_reserves_nothing_and_says_so(self):
        metrics = _collector(SPARK, reserved=0.0)

        assert metrics["memory_used_mb"] == 0
        assert metrics["memory_used_source"] == "engine_reservations"

    def test_utilisation_is_null_not_zero(self):
        """NVML answers 0 for the counter it does not populate. A 0 here reads as
        an idle node, on a node that is serving requests (#176)."""
        metrics = _collector(SPARK)

        assert metrics["utilization_percent"] is None
        assert metrics["devices"][0]["utilization_percent"] is None

    def test_temperature_still_comes_through(self):
        """Same NVML handle, a counter the driver DOES populate: it must not be
        dropped along with the ones it does not."""
        assert _collector(SPARK)["temperature_c"] == 38

    def test_an_unsupported_utilisation_call_is_null_on_a_discrete_gpu_too(self):
        metrics = _collector([("Tesla V100-SXM2-32GB", 32 * GB, 4 * GB, None, 44)])

        assert metrics["utilization_percent"] is None
        assert metrics["memory_used_mb"] == 4 * 1024  # memory still measured


class TestEngineReservations:
    def test_no_engine_reserves_nothing(self):
        app = {"config": SimpleNamespace(gpu_memory_utilization=0.5)}

        assert engine_reserved_fraction(app) == 0.0

    def test_instances_are_summed(self):
        def _inst(gmu):
            return SimpleNamespace(
                backend=SimpleNamespace(config=SimpleNamespace(gpu_memory_utilization=gmu)))

        app = {
            "config": SimpleNamespace(gpu_memory_utilization=0.5),
            "instances": SimpleNamespace(instances=lambda: [_inst(0.5), _inst(0.3)]),
        }

        assert engine_reserved_fraction(app) == pytest.approx(0.8)

    def test_a_replayed_head_not_in_the_manager_still_counts(self):
        app = {
            "config": SimpleNamespace(gpu_memory_utilization=0.85),
            "engine": SimpleNamespace(),
        }

        assert engine_reserved_fraction(app) == pytest.approx(0.85)

    def test_a_provider_that_raises_leaves_usage_unknown(self):
        collector = MetricsCollector()
        collector.set_reservation_provider(lambda: 1 / 0)
        fake_vm = MagicMock(total=122 * GB, used=120 * GB)
        with patch.dict("sys.modules", {"pynvml": _nvml(SPARK)}), \
             patch("psutil.virtual_memory", return_value=fake_vm):
            metrics = collector.get_gpu_metrics()

        assert metrics["memory_used_mb"] is None


# ------------------------------------------------- #203 / #178 the node rows --

@pytest.mark.asyncio
async def test_node_rows_carry_the_elected_role(client, app):
    """The election is the server's, and the rows now state it. The topology used
    to crown row 0, which is always the local node (#203)."""
    _seed_cluster(app, [_node(node_id="aaa-master", node_name="Spark-1-DGX")])

    rows = (await (await client.get("/api/nodes")).json())["nodes"]
    by_id = {r["node_id"]: r for r in rows}

    # The lowest node_id wins the auto election, and it is NOT the local node.
    assert by_id["aaa-master"]["effective_role"] == "master"
    assert by_id["aaa-master"]["is_leader"] is True
    assert by_id["spark3"]["effective_role"] == "worker"
    assert by_id["spark3"]["is_leader"] is False


@pytest.mark.asyncio
async def test_a_worker_role_node_is_never_crowned(client, app):
    _seed_cluster(app, [_node(node_id="aaa-worker", role="worker")])

    rows = (await (await client.get("/api/nodes")).json())["nodes"]
    crowned = [r["node_id"] for r in rows if r["effective_role"] == "master"]

    assert crowned == ["spark3"]


@pytest.mark.asyncio
async def test_node_rows_carry_a_reachable_address(client, app):
    """localhost is right for exactly one row. For a peer it is the address its
    announcement arrived from, then the fabric IP (#178)."""
    _seed_cluster(app, [
        _node(node_id="peer-udp", peer_ip="192.168.0.118", fabric_ip="10.100.0.11"),
        _node(node_id="peer-fabric", peer_ip=None, fabric_ip="10.100.0.15"),
    ])

    rows = (await (await client.get("/api/nodes")).json())["nodes"]
    hosts = {r["node_id"]: r["host"] for r in rows}

    assert hosts["spark3"] == "localhost"
    assert hosts["peer-udp"] == "192.168.0.118"
    assert hosts["peer-fabric"] == "10.100.0.15"


@pytest.mark.asyncio
async def test_a_node_that_cannot_measure_reports_null_not_zero(client, app):
    """An announcement with no telemetry produces nulls on the row, so the
    interface draws n/a. A 0 would read as an idle, empty node (#176)."""
    _seed_cluster(app, [_node(node_id="peer-quiet")])

    rows = (await (await client.get("/api/nodes")).json())["nodes"]
    peer = next(r for r in rows if r["node_id"] == "peer-quiet")

    assert peer["gpu_utilization"] is None
    assert peer["gpu_memory_used_pct"] is None
    assert peer["gpu_temp"] is None


@pytest.mark.asyncio
async def test_an_announced_measurement_survives_to_the_row(client, app):
    _seed_cluster(app, [_node(
        node_id="peer-busy", gpu_memory_gb=32.0, gpu_count=4,
        gpu_memory_used_mb=16384.0, gpu_memory_total_mb=32768.0,
        gpu_utilization=44.0, gpu_temp=29.0)])

    rows = (await (await client.get("/api/nodes")).json())["nodes"]
    peer = next(r for r in rows if r["node_id"] == "peer-busy")

    assert peer["gpu_memory_used_pct"] == 50
    assert peer["gpu_utilization"] == 44
    assert peer["gpu_count"] == 4


# ------------------------------------------------- #174 what is actually free --

@pytest.mark.asyncio
async def test_available_vram_is_not_a_copy_of_the_total(client, app):
    """Two 128 GB nodes with 100 GB in use between them have 156 free, not
    256 (#174)."""
    _seed_cluster(app, [
        _node(node_id="peer-a", gpu_memory_gb=128.0, gpu_memory_used_mb=60.0 * 1024),
        _node(node_id="peer-b", gpu_memory_gb=128.0, gpu_memory_used_mb=40.0 * 1024),
    ])
    app["metrics_collector"] = SimpleNamespace(
        get_gpu_metrics=lambda: {"error": "no GPU on this test runner"})

    data = await (await client.get("/api/cluster/resources")).json()

    assert data["available_vram_gb"] == 156.0
    assert data["total_vram_gb"] > data["available_vram_gb"]
    # The local node has no figure, so it is named rather than counted as free.
    assert data["vram_unknown_nodes"] == ["Spark-3-DGX"]
    assert data["available_vram_is_floor"] is True


@pytest.mark.asyncio
async def test_available_vram_is_null_when_no_node_can_say(client, app):
    """Showing nothing beats showing the total twice."""
    _seed_cluster(app, [_node(node_id="peer-quiet", gpu_memory_gb=128.0)])
    app["metrics_collector"] = SimpleNamespace(
        get_gpu_metrics=lambda: {"error": "no GPU on this test runner"})

    data = await (await client.get("/api/cluster/resources")).json()

    assert data["available_vram_gb"] is None


@pytest.mark.asyncio
async def test_cluster_resources_counts_every_gpu(client, app):
    _seed_cluster(app, [
        _node(node_id="castor", gpu_count=4, gpu_memory_gb=128.0),
        _node(node_id="pollux", gpu_count=1, gpu_memory_gb=32.0),
    ])

    data = await (await client.get("/api/cluster/resources")).json()
    per_node = {n["node_id"]: n for n in data["nodes"]}

    assert per_node["castor"]["gpus"] == 4
    assert per_node["pollux"]["gpus"] == 1
    assert data["total_gpus"] == 6  # four plus one plus the local node


# ------------------------------------------------ a bench record's telemetry --
#
# The same coercion lived in the bench's own telemetry reader, where it is worse:
# `bench/SCHEMA.md` says a missing measurement is never filled with an estimate,
# and a 0 written there is a published record claiming an idle GPU during a run.

class TestBenchTelemetryReader:
    def _read(self, node, collector=None):
        from ainode.bench.fleet import cluster_nodes_reader

        app = {
            "config": SimpleNamespace(node_id="local"),
            "cluster_state": SimpleNamespace(members=lambda: [node]),
            "metrics_collector": collector,
        }
        return cluster_nodes_reader(app, node.node_id)()

    def test_a_node_that_cannot_measure_writes_no_number(self):
        sample = self._read(_node(node_id="peer-quiet", gpu_memory_gb=122.0))

        assert sample["gpu_util_pct"] is None
        assert sample["gpu_mem_used_gb"] is None
        assert sample["gpu_mem_total_gb"] == 122.0

    def test_a_measurement_is_still_recorded(self):
        sample = self._read(_node(
            node_id="peer-busy", gpu_memory_gb=128.0,
            gpu_memory_used_mb=64.0 * 1024, gpu_memory_total_mb=128.0 * 1024,
            gpu_utilization=88.0, gpu_temp=41.0))

        assert sample["gpu_util_pct"] == 88
        assert sample["gpu_mem_used_gb"] == 64.0
        assert sample["temp_c"] == 41

    def test_the_peak_block_omits_what_was_never_measured(self):
        """`result()` drops None keys, so a run on a node with no utilisation
        counter reports no utilisation rather than a peak of 0."""
        from ainode.bench.measure import Telemetry

        telemetry = Telemetry(read=lambda: {
            "gpu_util_pct": None, "temp_c": 39,
            "gpu_mem_used_gb": None, "gpu_mem_total_gb": 121.7})
        telemetry.samples.append(telemetry._read())

        block = telemetry.result()

        assert "gpu_util_pct" not in block
        assert "gpu_mem_used_gb" not in block
        assert block["temp_c"] == 39


# ------------------------------------------- the announcement's byte ceiling --

def test_the_announcement_with_the_new_fields_still_fits():
    """The listener reads ONE datagram of MAX_ANNOUNCEMENT_BYTES. A payload past
    it is truncated on arrival, fails to parse, and the node vanishes from every
    peer's cluster view with nothing logged."""
    ann = NodeAnnouncement(
        node_id="spark1-head", node_name="Spark-1-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, gpu_count=4,
        model="unsloth/Qwen3.8-27B-NVFP4", status="starting",
        api_port=8000, web_port=3000, fabric_ip="10.100.0.1",
        distributed_mode="head", distributed_instance_id="spark1-head:big-moe",
        distributed_peers=["10.100.0.2", "10.100.0.3", "10.100.0.4"],
        gpu_memory_used_mb=104857.0, gpu_memory_total_mb=124611.0,
        gpu_utilization=97.0, gpu_temp=41.0,
        instances=[{"instance_id": f"spark1-head:model-{i}",
                    "model": "ornith-ai/Ornith-1.5-35B-A3B-NVFP4",
                    "head_node_id": "spark1-head", "peer_ips": ["10.100.0.2"],
                    "api_port": 8000 + i, "tensor_parallel_size": 2,
                    "status": "serving", "distributed_executor": "mp"}
                   for i in range(4)],
        load_phase="loading_weights", load_started_at=time.time(),
        load_elapsed_seconds=195.3, expected_ready_minutes=12.0,
    )

    assert len(ann.to_json().encode()) < MAX_ANNOUNCEMENT_BYTES


def test_a_null_telemetry_figure_survives_the_wire():
    """None must not become 0.0 on the way through JSON: a peer has to be able
    to tell "cannot measure" from "idle"."""
    ann = NodeAnnouncement(
        node_id="spark1", node_name="Spark-1-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=122.0, unified_memory=True, model="", status="serving",
        api_port=8000, web_port=3000, gpu_utilization=None,
        gpu_memory_used_mb=None,
    )

    restored = NodeAnnouncement.from_json(ann.to_json())

    assert restored.gpu_utilization is None
    assert restored.gpu_memory_used_mb is None
    assert ClusterNode.from_announcement(
        restored, NodeStatus.ONLINE).gpu_utilization is None


# ---------------------------------------------- #180 size and quantization --

def _snapshot(tmp_path, repo="unsloth/Qwen3.8-27B-NVFP4", *, weights=3, quant=None):
    """An HF cache entry: blobs plus a snapshot of symlinks into them."""
    slug = "models--" + repo.replace("/", "--")
    root = tmp_path / slug
    blobs = root / "blobs"
    snapshot = root / "snapshots" / "a1b2c3"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for i in range(weights):
        blob = blobs / f"blob{i}"
        blob.write_bytes(b"w" * (1024 * 1024))
        (snapshot / f"model-{i}.safetensors").symlink_to(blob)
    config = {"architectures": ["Qwen3MoeForCausalLM"]}
    if quant is not None:
        config["quantization_config"] = quant
    config_blob = blobs / "config"
    config_blob.write_text(json.dumps(config))
    (snapshot / "config.json").symlink_to(config_blob)
    return root, snapshot


@pytest.fixture(autouse=True)
def clear_disk_caches():
    registry._disk_size_cache.clear()
    registry._quantization_cache.clear()
    yield
    registry._disk_size_cache.clear()
    registry._quantization_cache.clear()


class TestModelOnDisk:
    def test_size_comes_from_the_snapshot_directory(self, tmp_path):
        _snapshot(tmp_path, weights=3)

        size = registry.model_disk_size_bytes(
            "unsloth/Qwen3.8-27B-NVFP4", models_dir=tmp_path)

        # Three 1 MiB weight files plus the small config.
        assert size is not None
        assert 3 * 1024 * 1024 <= size < 3 * 1024 * 1024 + 4096

    def test_a_blob_linked_twice_is_counted_once(self, tmp_path):
        _, snapshot = _snapshot(tmp_path, weights=1)
        (snapshot / "model-copy.safetensors").symlink_to(
            (tmp_path / "models--unsloth--Qwen3.8-27B-NVFP4" / "blobs" / "blob0"))

        size = registry.model_disk_size_bytes(
            "unsloth/Qwen3.8-27B-NVFP4", models_dir=tmp_path)

        assert size < 2 * 1024 * 1024

    def test_a_model_not_on_this_disk_is_none_not_zero(self, tmp_path):
        assert registry.model_disk_size_bytes("who/knows", models_dir=tmp_path) is None

    def test_the_newest_revision_is_the_one_measured(self, tmp_path):
        root, _ = _snapshot(tmp_path, weights=1)
        older = root / "snapshots" / "old"
        older.mkdir()
        (older / "model.safetensors").write_bytes(b"x" * 8)
        import os
        os.utime(older, (1, 1))

        size = registry.model_disk_size_bytes(
            "unsloth/Qwen3.8-27B-NVFP4", models_dir=tmp_path)

        assert size >= 1024 * 1024

    def test_quantization_comes_from_the_catalog_recipe(self, tmp_path):
        """A curated entry states the format AINode launches the model with."""
        quant, source = registry.model_quantization(
            "ornith-ai/Ornith-1.5-35B-A3B-NVFP4", models_dir=tmp_path)

        assert quant == "NVFP4"
        assert source == "catalog"

    def test_quantization_falls_back_to_the_models_own_config(self, tmp_path):
        _snapshot(tmp_path, repo="someone/Custom-Model",
                  quant={"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8"})

        quant, source = registry.model_quantization(
            "someone/Custom-Model", models_dir=tmp_path)

        assert quant == "NVFP4"
        assert source == "config.json"

    def test_a_transformers_quantizer_block_is_read_too(self, tmp_path):
        _snapshot(tmp_path, repo="someone/AWQ-Model",
                  quant={"quant_method": "awq", "bits": 4})

        quant, source = registry.model_quantization(
            "someone/AWQ-Model", models_dir=tmp_path)

        assert quant == "awq"
        assert source == "config.json"

    def test_an_unquantized_model_says_nothing(self, tmp_path):
        _snapshot(tmp_path, repo="someone/Plain-Model")

        assert registry.model_quantization(
            "someone/Plain-Model", models_dir=tmp_path) == (None, None)

    def test_the_direct_download_layout_is_found(self, tmp_path):
        """Our own downloader writes org--name with real files, no snapshots."""
        direct = tmp_path / "someone--Direct-Model"
        direct.mkdir()
        (direct / "model.safetensors").write_bytes(b"d" * 2048)

        assert registry.model_disk_size_bytes(
            "someone/Direct-Model", models_dir=tmp_path) == 2048


@pytest.mark.asyncio
async def test_the_server_view_fills_size_and_quantization(client, app, tmp_path,
                                                           monkeypatch):
    """The Server view's rows carry the measured size and a quantization with the
    source it came from, instead of 0 and null for every model (#180)."""
    model = "unsloth/Qwen3.8-27B-NVFP4"
    _snapshot(tmp_path, repo=model, weights=2)
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)

    from ainode.api import server_routes

    async def _probe(_session, _port):
        return [model]

    monkeypatch.setattr(server_routes, "_probe_loaded_models", _probe)

    data = await (await client.get("/api/server/status")).json()
    row = next(r for r in data["loaded_models"] if r["id"] == model)

    assert row["size_bytes"] >= 2 * 1024 * 1024
    assert row["quantization"] == "NVFP4"
    assert row["quantization_source"] == "catalog"


@pytest.mark.asyncio
async def test_the_server_view_says_unknown_for_weights_it_cannot_see(
        client, app, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)

    from ainode.api import server_routes

    async def _probe(_session, _port):
        return ["nowhere/Not-On-This-Disk"]

    monkeypatch.setattr(server_routes, "_probe_loaded_models", _probe)

    data = await (await client.get("/api/server/status")).json()
    row = data["loaded_models"][0]

    assert row["size_bytes"] is None
    assert row["quantization"] is None


# ------------------------------------------------------------- the browser --
#
# There is no JS test harness in this repo, so these assert on the text of the
# scripts the way tests/test_load_progress.py and tests/test_bench.py do.

def _js(name):
    from pathlib import Path
    return (Path(server.__file__).resolve().parent.parent
            / "web" / "static" / "js" / name).read_text()


def test_the_topology_has_no_row_order_crown():
    """The fallback that crowned incoming[0] is gone: with no role information,
    no node wears a crown (#203)."""
    source = _js("topology.js")

    assert "incoming[0].role = 'master'" not in source
    assert "n.data.effective_role === 'master' || n.data.is_leader" in source


def test_the_topology_draws_n_a_for_an_unmeasurable_counter():
    source = _js("topology.js")

    assert "'n/a (not exposed here)'" in source
    assert "usage n/a" in source


def test_the_browser_keeps_a_null_reading_null():
    """app.js must not coerce a null metric to 0 when it merges the local node's
    live figures into the topology data."""
    source = _js("app.js")

    assert "gm.memory_used_mb === null || gm.memory_used_mb === undefined" in source
    assert "n.gpu_memory_used_pct = usedPct;" in source
