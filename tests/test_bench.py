"""Tests for ainode.bench - the in-product benchmark, its routes and its jobs.

Everything here runs against a FAKE node: one aiohttp app that answers both the
engine's OpenAI surface (``/v1/models``, streaming ``/v1/chat/completions`` with
a usage block) and an AINode node's control surface (``/api/status``,
``/api/config``, ``/api/server/status``). That is enough to exercise the whole
path the real fleet takes - routing-truth resolution, the readiness probe,
placement from a live node config, telemetry off cluster state, the job
lifecycle, and every route - without a GPU.

The fake deliberately reports a completion_tokens count that does NOT match the
number of SSE chunks it sends, because that is the bug the measurement code is
built to avoid: under speculative decoding one chunk carries several accepted
tokens, and counting chunks halves the rate.
"""

import asyncio
import json
import socket

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.bench.api_routes import _options_from_body
from ainode.bench.cli import record_path
from ainode.bench.fleet import BenchTarget, resolve_target
from ainode.bench.measure import BenchOptions, Reporter, measure, slug
from ainode.bench.runner import BenchBusy, BenchManager, summarize
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode

MODEL = "fakeorg/Fake-Bench-30B-A3B-NVFP4"
OTHER_MODEL = "fakeorg/Other-7B-NVFP4"
FLAGS = ["--kv-cache-dtype", "fp8", "--enforce-eager"]


# ---------------------------------------------------------------- fake node

class FakeNode:
    """An OpenAI-compatible engine plus an AINode control surface, on one port."""

    def __init__(self):
        self.prompts = []
        self.requests = []
        self.chunk_delay = 0.002
        self.chunks = 5
        self.completion_tokens = 50     # NOT self.chunks: see module docstring
        self.served = [MODEL]
        self.requests_last_minute = 0

    def app(self):
        app = web.Application()
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/api/status", self.status)
        app.router.add_get("/api/config", self.config)
        app.router.add_get("/api/server/status", self.server_status)
        return app

    async def models(self, _request):
        return web.json_response({"object": "list", "data": [
            {"id": m, "object": "model", "max_model_len": 262144} for m in self.served]})

    async def chat(self, request):
        body = await request.json()
        prompt = body["messages"][0]["content"]
        self.prompts.append(prompt)
        self.requests.append(body)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        want = min(self.chunks, max(1, int(body.get("max_tokens") or 1)))
        for _ in range(want):
            await asyncio.sleep(self.chunk_delay)
            await resp.write(
                b'data: {"choices":[{"delta":{"content":"tok "}}]}\n\n')
        usage = {"prompt_tokens": max(1, len(prompt) // 4),
                 "completion_tokens": self.completion_tokens,
                 "total_tokens": self.completion_tokens}
        await resp.write(
            json.dumps({"choices": [], "usage": usage}).encode().join([b"data: ", b"\n\n"]))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def status(self, _request):
        return web.json_response({
            "node_id": "fake-node", "node_name": "Fake-Spark", "version": "9.9.9",
            "gpu": {"name": "NVIDIA GB10"}, "model": MODEL, "models_loaded": self.served})

    async def config(self, _request):
        return web.json_response({
            "model": MODEL, "engine_image": "vllm/vllm-openai:v0.27.1",
            "extra_vllm_args": list(FLAGS), "gpu_memory_utilization": 0.85,
            "kv_cache_dtype": "fp8", "max_model_len": 262144})

    async def server_status(self, _request):
        return web.json_response({
            "status": "running",
            "request_count_last_minute": self.requests_last_minute,
            "loaded_models": [{"id": m, "node_id": "fake-node", "port": 8000,
                               "ready": True, "parallel": 1} for m in self.served]})


@pytest.fixture
def fake():
    return FakeNode()


@pytest_asyncio.fixture
async def engine(fake):
    """The fake node, listening on a real port (the bench uses blocking urllib
    from a worker thread, so it needs a socket, not a mocked transport)."""
    server = TestServer(fake.app())
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def results(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest_asyncio.fixture
async def client(engine, results):
    """A real AINode app whose cluster state contains the fake node.

    Discovery is off so the seeded node is not swept away by a sync, and the
    bench manager writes into tmp_path instead of ~/.ainode.
    """
    config = NodeConfig(node_id="local-node", node_name="LocalNode", model=None,
                        api_port=_free_port(), web_port=_free_port(),
                        cluster_enabled=False)
    app = create_app(config=config, engine=None)
    app["bench_manager"] = BenchManager(results)
    app["cluster_state"].add_node(ClusterNode(
        node_id="fake-node", node_name="Fake-Spark", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model=MODEL,
        status=NodeStatus.ONLINE, api_port=engine.port, web_port=engine.port,
        last_seen=0.0, fabric_ip="127.0.0.1",
        gpu_memory_used_mb=60000.0, gpu_memory_total_mb=131072.0,
        gpu_utilization=0.0, gpu_temp=41.0,
    ))
    async with TestClient(TestServer(app)) as c:
        yield c


SMALL = {"model": MODEL, "sections": ["single"], "depths": [400], "streams": [1],
         "no_think": True, "max_tokens": 16, "label": "unit"}


async def _wait_done(client, run_id, timeout=20.0):
    """Poll the run until it leaves running, the way the view does."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/api/bench/runs/{run_id}")
        data = await resp.json()
        if data["status"] not in ("pending", "running"):
            return data
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} never finished: {data}")


# ---------------------------------------------------------------- measurement

@pytest.mark.asyncio
async def test_token_counts_come_from_the_server_not_the_chunks(engine, fake):
    """The whole reason the code reads usage: chunks != tokens under spec decode."""
    opts = BenchOptions(url=f"http://127.0.0.1:{engine.port}", model=MODEL,
                        label="t", sections=["single"], max_tokens=16, no_think=True)
    results, _seconds, _cpt = await asyncio.to_thread(measure, opts, Reporter())
    assert fake.chunks != fake.completion_tokens
    assert results["single_stream"]["gen_tokens"] == fake.completion_tokens


@pytest.mark.asyncio
async def test_every_prompt_leads_with_a_unique_nonce(engine, fake):
    """A trailing nonce would leave the prefix cacheable and time a cache hit."""
    opts = BenchOptions(url=f"http://127.0.0.1:{engine.port}", model=MODEL,
                        label="t", sections=["single", "prefill", "concurrency"],
                        depths=[400, 800], streams=[1, 2], max_tokens=8, no_think=True)
    await asyncio.to_thread(measure, opts, Reporter())
    assert len(fake.prompts) > 4
    heads = [p.split("]")[0] + "]" for p in fake.prompts]
    assert all(h.startswith("[run ") for h in heads)
    assert len(set(heads)) == len(heads), "a nonce repeated; prefix caching could hit"


@pytest.mark.asyncio
async def test_no_think_sends_enable_thinking_false_except_for_reasoning(engine, fake):
    opts = BenchOptions(url=f"http://127.0.0.1:{engine.port}", model=MODEL,
                        label="t", sections=["single", "reasoning"],
                        max_tokens=8, reasoning_tokens=8, no_think=True)
    await asyncio.to_thread(measure, opts, Reporter())
    flags = [(b.get("chat_template_kwargs") or {}).get("enable_thinking")
             for b in fake.requests if b.get("max_tokens") != 1]
    # single -> False; reasoning measures both states regardless of no_think
    assert flags.count(False) >= 2 and True in flags


def test_slug_matches_the_schema_filename_convention():
    assert slug("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4") == \
        "nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4"


def test_the_speed_record_filename_slugs_the_label(tmp_path):
    """A label with spaces and a comma names a file a shell can quote (#158).

    The speed section interpolated ``--label`` verbatim while the harness, agentic,
    decide and embed sections all ran theirs through ``slug()``, so the one section
    that predates them wrote the operator's prose into the filename.
    """
    path = record_path(tmp_path, "20260919-220011",
                       "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                       "Spark-4 solo, 32k ctx")
    assert path.name == ("20260919-220011-nvidia-nemotron-3_5-lightning-30b-a3b-nvfp4-"
                         "spark-4-solo-32k-ctx.json")
    assert not set(path.name) & set(' ,/\\"\'')


def test_the_speed_record_filename_survives_a_slash_in_the_label(tmp_path):
    """A slash was the one that did more than look bad: it made the write land in a
    directory that is not there, so the run measured and then lost the record.

    ``slug()`` keeps only the part after the last slash, because its first job is
    stripping the org off a model id. On a label that costs the text in front of
    the slash, which is lossy but is what the other four sections do with theirs,
    and the record still lands in ``out_dir`` under a name a shell can pass around.
    """
    path = record_path(tmp_path, "20260919-220011", "org/M", "tp2/roce")
    assert path.parent == tmp_path
    assert path.name == "20260919-220011-m-roce.json"


# ------------------------------------------------------- the record's model block

def _catalog_model_block(model_id):
    """The ``model`` block a record gets for a catalog model, built the way both
    describe paths build it: catalog metadata first, then what the id states."""
    from ainode.bench.fleet import _apply_catalog, _catalog_entry, derive_arch

    mb, pl = {"id": model_id}, {}
    info = _catalog_entry(model_id)
    assert info, f"{model_id} is not in the curated catalog any more"
    _apply_catalog(mb, pl, info)
    derive_arch(mb, model_id)
    return mb


def test_a_moe_entry_carries_its_active_params_even_with_no_marker_in_the_id():
    """The catalog states the shape, so an id like DeepSeek-V4-Flash-DSpark or
    Qwen3.8-Flash-Next - neither of which carries an A<n>B marker - is no longer
    recorded as a dense model that reads all of its weights per token."""
    deepseek = _catalog_model_block("fraserprice/DeepSeek-V4-Flash-DSpark")
    assert (deepseek["params_b"], deepseek["active_b"]) == (284.0, 13.0)
    assert deepseek["arch"] == "moe"

    flash_next = _catalog_model_block("nvidia/Qwen3.8-Flash-Next-NVFP4")
    assert (flash_next["params_b"], flash_next["active_b"]) == (125.0, 6.0)
    assert flash_next["arch"] == "moe"


def test_a_dense_entry_is_recorded_dense_and_reads_every_parameter():
    dense = _catalog_model_block("unsloth/Qwen3.8-27B-NVFP4")
    assert dense["arch"] == "dense"
    assert dense["active_b"] == dense["params_b"] == 27.0


def test_what_the_catalog_does_not_state_still_comes_off_the_model_id():
    """The fallback is unchanged for a model nothing curated describes."""
    from ainode.bench.fleet import derive_arch

    mb = {"params_b": 30.0}
    derive_arch(mb, "fakeorg/Fake-30B-A3B-NVFP4")
    assert (mb["arch"], mb["active_b"], mb["quant"]) == ("moe", 3, "NVFP4")

    mb = {"params_b": 8.0}
    derive_arch(mb, "fakeorg/Fake-8B")
    assert (mb["arch"], mb["active_b"]) == ("dense", 8.0)

    # An entry that states MoE without an active count keeps the field out rather
    # than claiming the model reads all of its parameters per token.
    mb = {"params_b": 504.0, "arch": "moe"}
    derive_arch(mb, "madeby561/GLM-5.2-NVFP4-REAP-504B")
    assert mb["arch"] == "moe" and "active_b" not in mb


def test_the_catalog_cache_round_trips_the_shape_fields():
    """The cache is asdict/ModelInfo(**d), and a file written before these fields
    existed still has to load."""
    from ainode.models.registry import CURATED_CLUSTER_MODELS, ModelInfo

    info = CURATED_CLUSTER_MODELS["deepseek-v4-flash-dspark"]
    cached = json.loads(json.dumps(info.to_dict()))
    again = ModelInfo(**cached)
    assert again == info
    assert (again.active_params_b, again.arch) == (13.0, "moe")

    stale = {k: v for k, v in cached.items() if k not in ("active_params_b", "arch")}
    assert ModelInfo(**stale).active_params_b is None
    assert ModelInfo(**stale).arch == ""


# ---------------------------------------------------------------- job lifecycle

async def _describe():
    return {"id": MODEL}, {"engine": "vllm", "node": "Fake-Spark"}, []


@pytest.mark.asyncio
async def test_job_runs_to_completion_and_writes_schema_1(engine, results):
    manager = BenchManager(results)
    opts = BenchOptions(url=f"http://127.0.0.1:{engine.port}", model=MODEL,
                        label="lifecycle", sections=["single"], max_tokens=8,
                        no_think=True)
    run = await manager.submit(opts, BenchTarget(MODEL, "127.0.0.1", engine.port,
                                                 node_id="fake-node",
                                                 node_name="Fake-Spark"), _describe)
    for _ in range(400):
        if run.status not in ("pending", "running"):
            break
        await asyncio.sleep(0.05)
    assert run.status == "completed", run.logs
    assert run.progress()["percent"] == 100.0
    assert run.logs and any("SINGLE STREAM" in line for line in run.logs)
    path = results / f"{run.run_id}.json"
    assert path.is_file()
    record = json.loads(path.read_text())
    assert record["schema"] == 1
    assert record["label"] == "lifecycle"
    assert record["model"]["id"] == MODEL
    assert record["results"]["single_stream"]["decode_tok_s"] > 0
    assert record["notes"] and "nothing loaded" in record["notes"][0]
    # The run id IS the file stem, so a result outlives the process that made it.
    assert manager.result_path(run.run_id) == path


@pytest.mark.asyncio
async def test_one_run_at_a_time_per_node(engine, results, fake):
    fake.chunk_delay = 0.05
    manager = BenchManager(results)
    target = BenchTarget(MODEL, "127.0.0.1", engine.port)
    opts = BenchOptions(url=target.url, model=MODEL, label="first",
                        sections=["sustained"], sustained_tokens=8, no_think=True)
    run = await manager.submit(opts, target, _describe)
    with pytest.raises(BenchBusy) as exc:
        await manager.submit(opts, target, _describe)
    assert exc.value.running_id == run.run_id
    await manager.cancel(run.run_id)


@pytest.mark.asyncio
async def test_cancel_stops_the_run_and_writes_nothing(engine, results, fake):
    fake.chunk_delay = 0.2
    fake.chunks = 40
    manager = BenchManager(results)
    target = BenchTarget(MODEL, "127.0.0.1", engine.port)
    opts = BenchOptions(url=target.url, model=MODEL, label="cancelme",
                        sections=["sustained"], sustained_tokens=40, no_think=True)
    run = await manager.submit(opts, target, _describe)
    for _ in range(100):
        if run.status == "running" and run.logs:
            break
        await asyncio.sleep(0.05)
    assert await manager.cancel(run.run_id) is True
    for _ in range(200):
        if run.status not in ("pending", "running"):
            break
        await asyncio.sleep(0.05)
    assert run.status == "cancelled"
    assert not list(results.glob("*.json")), "a cancelled run must not write a result"
    assert manager.active_id is None


@pytest.mark.asyncio
async def test_delete_refuses_a_live_run_then_removes_the_file(engine, results, fake):
    fake.chunk_delay = 0.2
    fake.chunks = 40
    manager = BenchManager(results)
    target = BenchTarget(MODEL, "127.0.0.1", engine.port)
    opts = BenchOptions(url=target.url, model=MODEL, label="del",
                        sections=["sustained"], sustained_tokens=40, no_think=True)
    run = await manager.submit(opts, target, _describe)
    with pytest.raises(BenchBusy):
        manager.delete(run.run_id)
    await manager.cancel(run.run_id)
    for _ in range(200):
        if run.status not in ("pending", "running"):
            break
        await asyncio.sleep(0.05)
    assert manager.delete(run.run_id) is True
    assert manager.get(run.run_id) is None
    assert manager.delete("nope") is False


def test_list_runs_restores_completed_results_from_disk(results):
    (results / "20260913-010203-fake-bench-30b-a3b-nvfp4-restored.json").write_text(
        json.dumps({"schema": 1, "stamp": "20260913-010203", "label": "restored",
                    "model": {"id": MODEL, "name": "Fake"},
                    "placement": {"node": "Fake-Spark"},
                    "results": {"single_stream": {"decode_tok_s": 41.5},
                                "concurrency": [{"streams": 16,
                                                 "aggregate_tok_s": 260.0}]}}))
    rows = BenchManager(results).list_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed" and row["restored"] is True
    assert row["summary"]["single_tok_s"] == 41.5
    assert row["summary"]["conc_streams"] == 16
    assert row["summary"]["conc_aggregate_tok_s"] == 260.0


def test_result_path_refuses_to_escape_the_results_directory(results):
    manager = BenchManager(results)
    for bad in ("../../etc/passwd", "..", ".hidden", "a/b", "a\\b"):
        assert manager.result_path(bad) is None


def test_summarize_leaves_an_unmeasured_section_as_none():
    out = summarize({"schema": 1, "model": {"id": MODEL}, "placement": {},
                     "results": {}})
    assert out["single_tok_s"] is None
    assert out["conc_aggregate_tok_s"] is None


# ---------------------------------------------------------------- body validation

def test_body_validation_rejects_what_the_form_should_not_send():
    with pytest.raises(ValueError, match="model is required"):
        _options_from_body({})
    with pytest.raises(ValueError, match="unknown section"):
        _options_from_body({"model": MODEL, "sections": ["singel"]})
    with pytest.raises(ValueError, match="depths"):
        _options_from_body({"model": MODEL, "depths": [0]})
    with pytest.raises(ValueError, match="streams"):
        _options_from_body({"model": MODEL, "streams": [999]})
    with pytest.raises(ValueError, match="max_tokens"):
        _options_from_body({"model": MODEL, "max_tokens": 99999})
    with pytest.raises(ValueError, match="label is too long"):
        _options_from_body({"model": MODEL, "label": "x" * 61})


def test_body_validation_accepts_comma_strings_and_canonicalises_order():
    opts = _options_from_body({"model": MODEL, "depths": "4000, 16000",
                               "streams": "1,4", "sections": ["concurrency", "single"]})
    assert opts.depths == [4000, 16000] and opts.streams == [1, 4]
    assert opts.sections == ["single", "concurrency"], "canonical section order"
    assert opts.label == "web"


# ---------------------------------------------------------------- routes

@pytest.mark.asyncio
async def test_sections_route_names_the_form_checkboxes(client):
    resp = await client.get("/api/bench/sections")
    assert resp.status == 200
    keys = [s["key"] for s in (await resp.json())["sections"]]
    assert keys == ["single", "prefill", "sustained", "concurrency", "reasoning"]


@pytest.mark.asyncio
async def test_post_run_resolves_the_fleet_and_records_live_placement(client, results):
    resp = await client.post("/api/bench/runs", json=SMALL)
    assert resp.status == 202, await resp.text()
    started = await resp.json()
    assert started["target"]["node"] == "Fake-Spark"
    assert started["target"]["port"] == client.app["cluster_state"] \
        .get_node("fake-node").api_port

    done = await _wait_done(client, started["run_id"])
    assert done["status"] == "completed", done["log"]
    pl = done["result"]["placement"]
    # Placement is read, never guessed: node + GPU off the instance snapshot,
    # flags + image + gmu off the owning node's live config.
    assert pl["node"] == "Fake-Spark" and pl["gpu"] == "NVIDIA GB10"
    assert pl["engine_image"] == "vllm/vllm-openai:v0.27.1"
    assert pl["flags"] == FLAGS
    assert pl["gpu_memory_utilization"] == 0.85
    assert pl["kv_cache_dtype"] == "fp8"
    assert pl["flags_source"].startswith("live node config")
    assert pl["tp"] == 1 and pl["stacked_with"] == []
    # Model metadata that the id itself states, not a model card.
    assert done["result"]["model"]["arch"] == "moe"
    assert done["result"]["model"]["active_b"] == 3
    assert done["result"]["model"]["quant"] == "NVFP4"
    # Telemetry sampled from the same fields /api/nodes renders.
    tel = done["result"]["results"]["telemetry"]
    assert tel["gpu_mem_total_gb"] == 128.0
    assert tel["temp_c"] == 41
    assert any("unified-memory GPU" in n for n in done["result"]["notes"])
    assert done["summary"]["single_tok_s"] > 0


@pytest.mark.asyncio
async def test_get_runs_lists_then_download_and_report_serve_the_file(client, results):
    resp = await client.post("/api/bench/runs", json=SMALL)
    run_id = (await resp.json())["run_id"]
    await _wait_done(client, run_id)

    listed = await (await client.get("/api/bench/runs")).json()
    assert [r["run_id"] for r in listed["runs"]] == [run_id]
    assert listed["running"] is None

    dl = await client.get(f"/api/bench/results/{run_id}.json")
    assert dl.status == 200
    assert dl.headers["Content-Type"].startswith("application/json")
    assert f'filename="{run_id}.json"' in dl.headers["Content-Disposition"]
    # Byte-identical to the file on disk: this is what a leaderboard consumes.
    assert await dl.read() == (results / f"{run_id}.json").read_bytes()

    report = await client.get("/api/bench/report")
    assert report.status == 200
    assert report.headers["Content-Type"].startswith("text/html")
    page = await report.text()
    assert "AINode Bench" in page and MODEL in page
    assert "<script" not in page.lower(), "the report is embedded in an iframe; no JS"


@pytest.mark.asyncio
async def test_report_renders_an_empty_results_directory(client):
    resp = await client.get("/api/bench/report")
    assert resp.status == 200
    assert "No results yet" in await resp.text()


@pytest.mark.asyncio
async def test_post_run_404s_a_model_nothing_is_serving(client):
    resp = await client.post("/api/bench/runs", json={**SMALL, "model": "nope/nope"})
    assert resp.status == 404
    assert "never loads a model" in (await resp.json())["error"]


@pytest.mark.asyncio
async def test_post_run_409s_when_the_instance_is_not_serving_that_model(client, fake):
    fake.served = [OTHER_MODEL]          # engine is up, but not with our model
    resp = await client.post("/api/bench/runs", json=SMALL)
    assert resp.status == 409
    assert "not ready" in (await resp.json())["error"]


@pytest.mark.asyncio
async def test_post_run_409s_while_another_run_is_going(client, fake):
    fake.chunk_delay = 0.2
    fake.chunks = 40
    first = await client.post("/api/bench/runs", json={
        **SMALL, "sections": ["sustained"], "sustained_tokens": 40})
    assert first.status == 202
    run_id = (await first.json())["run_id"]
    second = await client.post("/api/bench/runs", json=SMALL)
    assert second.status == 409
    body = await second.json()
    assert body["running"] == run_id
    await client.post(f"/api/bench/runs/{run_id}/cancel")


@pytest.mark.asyncio
async def test_post_run_400s_a_bad_body(client):
    resp = await client.post("/api/bench/runs", data=b"not json")
    assert resp.status == 400
    resp = await client.post("/api/bench/runs", json={"model": MODEL, "streams": [0]})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_cancel_route_stops_the_run_then_409s(client, fake):
    fake.chunk_delay = 0.2
    fake.chunks = 40
    started = await (await client.post("/api/bench/runs", json={
        **SMALL, "sections": ["sustained"], "sustained_tokens": 40})).json()
    run_id = started["run_id"]
    resp = await client.post(f"/api/bench/runs/{run_id}/cancel")
    assert resp.status == 200
    done = await _wait_done(client, run_id)
    assert done["status"] == "cancelled"
    again = await client.post(f"/api/bench/runs/{run_id}/cancel")
    assert again.status == 409
    assert (await client.post("/api/bench/runs/ghost/cancel")).status == 404


@pytest.mark.asyncio
async def test_delete_route_removes_the_result_file(client, results):
    run_id = (await (await client.post("/api/bench/runs", json=SMALL)).json())["run_id"]
    await _wait_done(client, run_id)
    assert (results / f"{run_id}.json").is_file()
    resp = await client.delete(f"/api/bench/runs/{run_id}")
    assert resp.status == 200 and (await resp.json())["deleted"] is True
    assert not (results / f"{run_id}.json").is_file()
    assert (await client.get(f"/api/bench/results/{run_id}.json")).status == 404
    assert (await client.delete(f"/api/bench/runs/{run_id}")).status == 404


@pytest.mark.asyncio
async def test_get_run_404s_an_unknown_id(client):
    assert (await client.get("/api/bench/runs/ghost")).status == 404


@pytest.mark.asyncio
async def test_a_busy_node_warns_but_still_runs(client, fake):
    """A busy node measures the queue. Say so; do not refuse the measurement."""
    fake.requests_last_minute = 12
    node = client.app["cluster_state"].get_node("fake-node")
    node.instances = [{"model": MODEL, "api_port": node.api_port, "status": "serving"},
                      {"model": OTHER_MODEL, "api_port": node.api_port + 1,
                       "status": "serving"}]
    resp = await client.post("/api/bench/runs", json=SMALL)
    assert resp.status == 202
    body = await resp.json()
    assert any("served 12 request" in w for w in body["warnings"])
    assert any(OTHER_MODEL in w for w in body["warnings"])
    done = await _wait_done(client, body["run_id"])
    assert done["status"] == "completed"
    # The warnings survive into the record, where they belong: a result that does
    # not name the co-resident model is not reproducible.
    assert any(OTHER_MODEL in n for n in done["result"]["notes"])
    assert done["result"]["placement"]["stacked_with"] == [OTHER_MODEL]


@pytest.mark.asyncio
async def test_resolve_target_uses_the_proxys_own_routing(client):
    """A bench must hit the instance a chat request would have hit."""
    from ainode.api.server import _routing_candidates

    app = client.app
    target = resolve_target(app, MODEL)
    cands = _routing_candidates(app["cluster_state"], MODEL,
                                app["config"].node_id, app["config"].api_port)
    assert (target.host, target.port) == cands[0]
    assert resolve_target(app, "nope/nope") is None


# ---------------------------------------------------------------- the view

def test_dashboard_wires_the_bench_view():
    """The four shared lines the view needs, and nothing more."""
    from ainode.web.serve import STATIC_DIR, get_index_html

    html = get_index_html()
    assert 'data-view="bench"' in html
    assert 'id="view-bench"' in html and 'id="bench-content"' in html
    assert "/static/js/bench.js" in html
    assert "/static/css/bench.css" in html
    # bench.js must load before app.js: app.js's view switch calls into it.
    assert html.index("/static/js/bench.js") < html.index("/static/js/app.js")
    assert (STATIC_DIR / "js" / "bench.js").is_file()
    assert (STATIC_DIR / "css" / "bench.css").is_file()


def test_app_js_hands_the_poll_tick_to_the_bench_view():
    from ainode.web.serve import STATIC_DIR

    app_js = (STATIC_DIR / "js" / "app.js").read_text()
    assert "case 'bench':" in app_js
    assert "window.AINodeBench.render(this)" in app_js


def test_bench_js_only_reads_the_fleet_and_never_loads_a_model():
    """A bench is inference only; the view must not be able to start a load."""
    from ainode.web.serve import STATIC_DIR

    bench_js = (STATIC_DIR / "js" / "bench.js").read_text()
    assert "/api/server/status" in bench_js, "the picker is built from fleet truth"
    assert "/api/bench/report" in bench_js
    for forbidden in ("/api/models/load", "/api/models/unload", "/api/cluster/load",
                      "/api/cluster/unload", "/eject"):
        assert forbidden not in bench_js


@pytest.mark.asyncio
async def test_progress_reports_section_and_step(client, fake):
    fake.chunk_delay = 0.1
    fake.chunks = 10
    started = await (await client.post("/api/bench/runs", json={
        **SMALL, "sections": ["prefill"], "depths": [400, 800, 1200],
        "max_tokens": 10})).json()
    run_id = started["run_id"]
    seen = []
    for _ in range(200):
        data = await (await client.get(f"/api/bench/runs/{run_id}")).json()
        seen.append(data["progress"])
        if data["status"] != "running":
            break
        await asyncio.sleep(0.05)
    assert any(p["section"] == "prefill" for p in seen)
    assert any(p["section_label"] == "Prefill scaling" for p in seen)
    assert any(p["step_total"] == 3 for p in seen)
    assert max(p["percent"] for p in seen) == 100.0


# ------------------------------------- the width of a single-node TP launch

class TestFlagWidth:
    """``tp`` / ``gpus`` for a solo launch across several cards in one box.

    The instance record counts NODES, so it says 1 for every solo launch, and
    castor's first record therefore read ``tp: 1, gpus: 1`` beside its own
    ``flags: [--tensor-parallel-size, 4]``. A record that contradicts itself is
    the one thing placement may not do (``bench/SCHEMA.md``), and this block is
    what a reader weighs before trusting a recipe on their own hardware.
    """

    def test_a_stated_width_corrects_a_node_count_of_one(self):
        from ainode.bench.fleet import apply_flag_width
        pl = {"tp": 1, "gpus": 1,
              "flags": ["--tensor-parallel-size", "4", "--max-num-seqs", "4"]}
        apply_flag_width(pl)
        assert pl["tp"] == 4 and pl["gpus"] == 4

    def test_the_equals_form_is_read_too(self):
        from ainode.bench.fleet import apply_flag_width
        pl = {"tp": 1, "gpus": 1, "flags": ["--tensor-parallel-size=8"]}
        apply_flag_width(pl)
        assert pl["tp"] == 8 and pl["gpus"] == 8

    def test_no_stated_width_stays_one_rather_than_being_guessed(self):
        from ainode.bench.fleet import apply_flag_width
        pl = {"tp": 1, "gpus": 1, "flags": ["--max-num-seqs", "4"]}
        apply_flag_width(pl)
        assert pl["tp"] == 1 and pl["gpus"] == 1

    def test_a_flagless_placement_is_left_alone(self):
        from ainode.bench.fleet import apply_flag_width
        pl = {"tp": 2, "gpus": 2}
        apply_flag_width(pl)
        assert pl == {"tp": 2, "gpus": 2}

    def test_a_multi_node_width_is_never_lowered(self):
        """A head's ``tp`` is its node count, read from the announcement. The
        flags carry the same number, and a smaller one there (a recipe pinning a
        per-node width) must not shrink it."""
        from ainode.bench.fleet import apply_flag_width
        pl = {"tp": 4, "gpus": 4, "flags": ["--tensor-parallel-size", "2"]}
        apply_flag_width(pl)
        assert pl["tp"] == 4 and pl["gpus"] == 4

    def test_an_unparseable_width_is_ignored(self):
        from ainode.bench.fleet import apply_flag_width
        pl = {"tp": 1, "gpus": 1, "flags": ["--tensor-parallel-size", "auto"]}
        apply_flag_width(pl)
        assert pl["tp"] == 1
