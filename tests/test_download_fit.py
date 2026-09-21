"""A download that cannot fit is refused before it starts (#184, point 4).

A pull with nowhere to go does not report a disk error. It stops part way, leaves
a partial blob cache behind, and the engine that was waiting for those weights
looks like an engine that crashed. The node knew the checkpoint's size and its own
free space before it began, so it refuses and says all three numbers: needed,
free, and which path was measured.

Three properties this file exists to hold:

* **Unknown is unknown, never zero.** A repo the Hub will not describe (no
  network, a gated repo, an answer with no file sizes) proceeds exactly as it did
  before the check existed. Reporting 0 would wave every download through as
  comfortably fitting a full disk.
* **The refusal is escapable.** ``force`` downloads it anyway, for the operator
  resuming a mostly complete pull or pointing at a filesystem this process cannot
  stat properly.
* **No HTTP round trip on the launch path.** A launch decides from what the node
  already knows (a remembered size, the catalog entry), because it must not wait
  on huggingface.co before it starts an engine.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

from ainode.models import api_routes as mr
from ainode.models import fit
from ainode.models.registry import CatalogAggregator, ModelManager

GB = fit.GB


# ---------------------------------------------------------------------------
# Learning the size
# ---------------------------------------------------------------------------

def _info(files=None, safetensors=None):
    """A stand-in for huggingface_hub's ModelInfo, with only what we read."""
    siblings = [SimpleNamespace(rfilename=name, size=size, lfs=None)
                for name, size in (files or [])]
    return SimpleNamespace(siblings=siblings, safetensors=safetensors)


def test_the_size_is_the_sum_of_every_file_the_hub_describes():
    info = _info([("model-00001.safetensors", 200 * GB),
                  ("model-00002.safetensors", 200 * GB),
                  ("config.json", 1024)])
    assert fit._files_total_bytes(info) == 400 * GB + 1024


def test_an_lfs_pointer_carries_the_size_when_the_file_entry_does_not():
    info = SimpleNamespace(siblings=[
        SimpleNamespace(rfilename="model.safetensors", size=None,
                        lfs={"size": 35 * GB}),
    ], safetensors=None)
    assert fit._files_total_bytes(info) == 35 * GB


def test_a_repo_with_no_file_sizes_is_unknown_and_not_zero():
    """The rule this whole module turns on: 0 would read as "fits in nothing"."""
    info = _info([("model.safetensors", None), ("config.json", 0)])
    assert fit._files_total_bytes(info) is None


def test_the_safetensors_breakdown_is_the_fallback():
    info = _info([], safetensors=SimpleNamespace(parameters={"BF16": 35_000_000_000}))
    # 35e9 params at 2 bytes each, from registry's one home for the dtype table.
    assert fit._safetensors_total_bytes(info) == 70 * GB


def test_a_hub_that_will_not_answer_reports_unknown(monkeypatch):
    """The failure path: the API is down, the token is wrong, DNS is gone."""
    class _Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo, **kw):
            raise OSError("Name or service not known")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub",
                        SimpleNamespace(HfApi=_Api))
    size = fit.fetch_repo_size("big/repo")
    assert size.total_bytes is None
    assert size.known is False


def test_no_huggingface_hub_installed_reports_unknown(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)
    assert fit.fetch_repo_size("big/repo").known is False


# ---------------------------------------------------------------------------
# Remembering it
# ---------------------------------------------------------------------------

def test_a_learned_size_is_remembered_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "size_cache_path", lambda: tmp_path / "repo-sizes.json")
    fit.remember_repo_size("a/b", fit.RepoSize(total_bytes=42 * GB, source="hub_files"))
    again = fit.cached_repo_size("a/b")
    assert again.total_bytes == 42 * GB
    assert again.source == "cache"


def test_an_unknown_size_is_never_remembered(tmp_path, monkeypatch):
    """A Hub outage must not be frozen in as a fact for a month."""
    monkeypatch.setattr(fit, "size_cache_path", lambda: tmp_path / "repo-sizes.json")
    fit.remember_repo_size("a/b", fit.RepoSize())
    assert not (tmp_path / "repo-sizes.json").exists()
    assert fit.cached_repo_size("a/b").known is False


def test_a_stale_entry_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "repo-sizes.json"
    monkeypatch.setattr(fit, "size_cache_path", lambda: path)
    path.write_text(json.dumps({"a/b": {
        "total_bytes": 10 * GB,
        "at": time.time() - fit.SIZE_CACHE_TTL_SECONDS - 60}}))
    assert fit.cached_repo_size("a/b").known is False


def test_an_unreadable_cache_is_not_a_crash(tmp_path, monkeypatch):
    path = tmp_path / "repo-sizes.json"
    monkeypatch.setattr(fit, "size_cache_path", lambda: path)
    path.write_text("{not json")
    assert fit.cached_repo_size("a/b").known is False


def test_the_cache_beats_the_hub_and_the_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "size_cache_path", lambda: tmp_path / "repo-sizes.json")
    fit.remember_repo_size("a/b", fit.RepoSize(total_bytes=7 * GB, source="hub_files"))
    monkeypatch.setattr(fit, "fetch_repo_size",
                        lambda *a, **k: pytest.fail("the Hub must not be asked"))
    assert fit.repo_size("a/b", catalog_size_gb=99).total_bytes == 7 * GB


def test_learn_false_never_asks_the_hub(tmp_path, monkeypatch):
    """The launch path: the catalog answers, and nothing waits on the network."""
    monkeypatch.setattr(fit, "size_cache_path", lambda: tmp_path / "repo-sizes.json")
    monkeypatch.setattr(fit, "fetch_repo_size",
                        lambda *a, **k: pytest.fail("the Hub must not be asked"))
    size = fit.repo_size("a/b", catalog_size_gb=35.0, learn=False)
    assert size.total_bytes == 35 * GB
    assert size.source == "catalog"


def test_a_catalog_size_of_zero_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "size_cache_path", lambda: tmp_path / "repo-sizes.json")
    monkeypatch.setattr(fit, "fetch_repo_size", lambda *a, **k: fit.RepoSize())
    assert fit.repo_size("a/b", catalog_size_gb=0).known is False


# ---------------------------------------------------------------------------
# Free space and the verdict
# ---------------------------------------------------------------------------

def test_free_space_is_measured_on_the_nearest_parent_that_exists(tmp_path):
    """A models dir that does not exist yet is created on its parent's
    filesystem, so that is the filesystem to measure."""
    assert fit.free_bytes(tmp_path / "models" / "not" / "there") == \
        fit.free_bytes(tmp_path)


def test_a_path_nothing_can_stat_is_unknown(monkeypatch):
    import shutil

    def boom(path):
        raise OSError("no filesystem here")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    assert fit.free_bytes("/nowhere") is None


def test_a_checkpoint_bigger_than_the_free_space_does_not_fit():
    v = fit.FitVerdict(hf_repo="big/moe", path="/models",
                       needed_bytes=400 * GB, free_bytes=70 * GB)
    assert v.fits is False
    assert v.known is True
    assert "400.0 GB" in v.message()
    assert "70.0 GB" in v.message()
    assert "/models" in v.message()


def test_one_that_fits_says_so():
    v = fit.FitVerdict(hf_repo="small/model", path="/models",
                       needed_bytes=7 * GB, free_bytes=70 * GB)
    assert v.fits is True


def test_an_unknown_size_is_not_a_refusal():
    v = fit.FitVerdict(hf_repo="mystery/repo", path="/models",
                       needed_bytes=None, free_bytes=70 * GB)
    assert v.fits is None
    assert "size unknown" in v.message()


def test_unreadable_free_space_is_not_a_refusal_either():
    v = fit.FitVerdict(hf_repo="a/b", path="/models",
                       needed_bytes=400 * GB, free_bytes=None)
    assert v.fits is None
    assert "could not be read" in v.message()


def test_the_verdict_dict_carries_every_number_and_the_path():
    v = fit.FitVerdict(hf_repo="big/moe", path="/models", needed_bytes=400 * GB,
                       free_bytes=70 * GB, size_source="hub_files")
    payload = v.to_dict()
    assert payload["needed_gb"] == 400.0
    assert payload["free_gb"] == 70.0
    assert payload["fits"] is False
    assert payload["path"] == "/models"
    assert payload["size_source"] == "hub_files"


def test_the_refusal_payload_names_the_way_past():
    v = fit.FitVerdict(hf_repo="big/moe", path="/models",
                       needed_bytes=400 * GB, free_bytes=70 * GB)
    payload = fit.refusal_payload(v)
    assert "not enough free space" in payload["error"]
    assert payload["fit"]["fits"] is False
    assert "force" in payload["force"]


def test_check_fit_never_raises_on_a_hopeless_path(monkeypatch):
    monkeypatch.setattr(fit, "fetch_repo_size", lambda *a, **k: fit.RepoSize())
    verdict = fit.check_fit("a/b", "/definitely/not/here")
    assert verdict.fits is None


# ---------------------------------------------------------------------------
# The HTTP shape
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _offline_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(CatalogAggregator, "CACHE_FILE", tmp_path / "catalog-cache.json")
    monkeypatch.setattr(CatalogAggregator, "fetch", lambda self, force_refresh=False: [])


@pytest.fixture
def client_factory(tmp_path, aiohttp_client):
    """An app on a models dir whose free space and repo sizes the test decides."""
    async def _build(*, free_bytes, sizes):
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)
        app = web.Application()
        register = mr.register_model_routes
        register(app, manager=ModelManager(models_dir=models))
        return await aiohttp_client(app), models

    return _build


def _wire(monkeypatch, *, free, sizes):
    """Point the fit check at fixed free space and fixed repo sizes."""
    monkeypatch.setattr(fit, "free_bytes", lambda path: free)
    monkeypatch.setattr(
        fit, "fetch_repo_size",
        lambda repo, token=None: (fit.RepoSize(total_bytes=sizes[repo], source="hub_files")
                                  if repo in sizes else fit.RepoSize()))


@pytest.mark.asyncio
async def test_a_400gb_repo_is_refused_against_70gb_free(client_factory, monkeypatch):
    client, models = await client_factory(free_bytes=None, sizes=None)
    _wire(monkeypatch, free=70 * GB, sizes={"big/moe-400b": 400 * GB})

    resp = await client.post("/api/models/download-repo", json={"hf_repo": "big/moe-400b"})
    assert resp.status == mr.FIT_REFUSED_STATUS == 507
    body = await resp.json()
    assert body["fit"]["needed_gb"] == 400.0
    assert body["fit"]["free_gb"] == 70.0
    assert body["fit"]["path"] == str(models)
    assert body["fit"]["fits"] is False
    assert "force" in body


@pytest.mark.asyncio
async def test_a_small_repo_is_accepted(client_factory, monkeypatch):
    client, _ = await client_factory(free_bytes=None, sizes=None)
    _wire(monkeypatch, free=70 * GB, sizes={"small/embed": 1 * GB})
    started = []
    monkeypatch.setattr(mr, "_run_download_repo",
                        _record(started))

    resp = await client.post("/api/models/download-repo", json={"hf_repo": "small/embed"})
    assert resp.status == 202
    assert (await resp.json())["status"] == "downloading"


@pytest.mark.asyncio
async def test_force_downloads_it_anyway(client_factory, monkeypatch):
    client, _ = await client_factory(free_bytes=None, sizes=None)
    _wire(monkeypatch, free=70 * GB, sizes={"big/moe-400b": 400 * GB})
    started = []
    monkeypatch.setattr(mr, "_run_download_repo", _record(started))

    resp = await client.post("/api/models/download-repo",
                             json={"hf_repo": "big/moe-400b", "force": True})
    assert resp.status == 202
    assert started, "the download has to actually start"


@pytest.mark.asyncio
async def test_a_repo_whose_size_is_unknown_still_downloads(client_factory, monkeypatch):
    """The Hub is down: the check has no evidence, so it does not refuse."""
    client, _ = await client_factory(free_bytes=None, sizes=None)
    _wire(monkeypatch, free=1 * GB, sizes={})
    started = []
    monkeypatch.setattr(mr, "_run_download_repo", _record(started))

    resp = await client.post("/api/models/download-repo", json={"hf_repo": "mystery/repo"})
    assert resp.status == 202
    assert started


@pytest.mark.asyncio
async def test_a_catalog_download_is_refused_on_the_entrys_own_size(client_factory,
                                                                   monkeypatch):
    """No Hub answer at all: the catalog's size_gb is the evidence."""
    client, _ = await client_factory(free_bytes=None, sizes=None)
    _wire(monkeypatch, free=1 * GB, sizes={})

    resp = await client.post("/api/models/llama-3.2-3b/download")
    assert resp.status == 507
    body = await resp.json()
    assert body["fit"]["size_source"] == "catalog"
    assert body["fit"]["fits"] is False


@pytest.mark.asyncio
async def test_a_catalog_download_that_fits_is_still_accepted(client_factory,
                                                             monkeypatch):
    client, _ = await client_factory(free_bytes=None, sizes=None)
    _wire(monkeypatch, free=500 * GB, sizes={})
    monkeypatch.setattr(mr, "_run_download", _record([]))

    resp = await client.post("/api/models/llama-3.2-3b/download")
    assert resp.status == 202


def _record(seen):
    """A stand-in download task that records the call and returns at once."""
    async def _task(*args, **kwargs):
        seen.append(args)
        return None
    return _task


# ------------------------------------------------------------ the launch path

class _Cfg:
    api_port = 8000
    web_port = 3000
    model = ""
    models_dir = ""
    gpu_memory_utilization = 0.6
    distributed_mode = "solo"
    engine_backend = "nvidia"

    def save(self):
        return None


@pytest.fixture
def load_client(tmp_path, aiohttp_client):
    async def _build():
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)
        app = web.Application()
        cfg = _Cfg()
        cfg.models_dir = str(models)
        app["config"] = cfg
        app["engine"] = None
        app["cluster_state"] = None
        mr.register_model_routes(app, manager=ModelManager(models_dir=models))
        return await aiohttp_client(app), models

    return _build


@pytest.mark.asyncio
async def test_a_launch_that_must_download_is_refused_too(load_client, monkeypatch):
    """The engine downloads the weights itself when they are not on disk, so the
    launch pays the same disk question and fails the same way without it."""
    client, _ = await load_client()
    monkeypatch.setattr(fit, "free_bytes", lambda path: 1 * GB)
    monkeypatch.setattr(mr, "catalog_size_gb", lambda model: 400.0)
    monkeypatch.setattr(fit, "fetch_repo_size",
                        lambda *a, **k: pytest.fail("a launch must not ask the Hub"))
    monkeypatch.setattr(mr, "append_solo_instance",
                        lambda *a, **k: pytest.fail("nothing may launch"))

    resp = await client.post("/api/models/load", json={"model": "big/moe-400b"})
    assert resp.status == 507
    assert (await resp.json())["fit"]["needed_gb"] == 400.0


@pytest.mark.asyncio
async def test_a_model_already_on_disk_is_never_size_checked(load_client, monkeypatch):
    """Nothing is downloaded, so there is nothing to refuse."""
    client, models = await load_client()
    weights = models / "big--moe-400b"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text("{}")
    monkeypatch.setattr(fit, "free_bytes", lambda path: 1 * GB)
    monkeypatch.setattr(mr, "catalog_size_gb", lambda model: 400.0)
    monkeypatch.setattr(mr, "launch_solo_serialized", _ok_launch)

    resp = await client.post("/api/models/load", json={"model": "big/moe-400b"})
    assert resp.status != 507


async def _ok_launch(app, model, gmu=None, **kw):
    return {"ok": True, "model": model, "api_port": 8000}


# ------------------------------------------------------------- catalog lookup

def test_the_catalog_size_lookup_is_offline_and_answers_by_id_or_repo():
    from ainode.models.registry import CURATED_CLUSTER_MODELS

    entry = next(iter(CURATED_CLUSTER_MODELS.values()))
    assert mr.catalog_size_gb(entry.id) == pytest.approx(entry.size_gb)
    assert mr.catalog_size_gb(entry.hf_repo) == pytest.approx(entry.size_gb)
    assert mr.catalog_size_gb("nobody/knows") is None
    assert mr.catalog_size_gb("") is None


def test_force_reads_both_spellings_and_only_a_real_yes():
    assert mr._wants_force({"force": True}) is True
    assert mr._wants_force({"force_download": "yes"}) is True
    assert mr._wants_force({"force": "true"}) is True
    assert mr._wants_force({"force": False}) is False
    assert mr._wants_force({"force": "maybe"}) is False
    assert mr._wants_force({}) is False


def test_the_size_cache_lives_under_the_ainode_home(tmp_path, monkeypatch):
    """Resolved per call, so a node whose home moved does not keep writing the
    old one. ``size_cache_path`` itself is redirected for the whole suite by
    conftest's ``no_hub_size_lookup``, so the resolution is what is asserted."""
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    assert fit._home() == Path(tmp_path)
    assert fit.SIZE_CACHE_NAME == "repo-sizes.json"


# ------------------------------------------------------- the card, under node
#
# app.js is plain script with one global, so node can run it against a stub DOM:
# no bundler, no jsdom. Skipped where node is absent rather than asserted on the
# file's text, because what matters is what the function ANSWERS for each of the
# four cases (no size, no free figure, fits, does not fit).

_CARD_CASE = r"""
const fs = require('fs');
const vm = require('vm');

function el() {
  return { style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {},
           contains() { return false; } },
           children: [], value: '', textContent: '', innerHTML: '',
           addEventListener() {}, appendChild() {}, remove() {},
           querySelector() { return el(); }, querySelectorAll() { return []; },
           getAttribute() { return null; }, setAttribute() {} };
}
global.document = { body: el(), createElement: el, getElementById() { return el(); },
                    querySelector() { return el(); }, querySelectorAll() { return []; },
                    addEventListener() {} };
global.window = { location: { href: 'http://localhost:3000/' }, addEventListener() {} };
global.requestAnimationFrame = (fn) => fn();
global.navigator = {};
global.performance = { now: () => Date.now() };
global.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
global.fetch = async () => ({ ok: true, status: 200, json: async () => ({}) });

const src = fs.readFileSync(process.argv[2], 'utf8') + '\nglobalThis.AINode = AINode;\n';
vm.runInThisContext(src, { filename: 'app.js' });

function note(model, disk) {
  AINode.state.status = disk === null ? null : { disk: disk };
  return AINode.modelFitNote(model);
}

const plenty = { models: { path: '/models', free_gb: 670.0, total_gb: 3700.0,
                           free_fraction: 0.18, warn: false } };
const tight = { models: { path: '/models', free_gb: 12.0, total_gb: 3700.0,
                          free_fraction: 0.003, warn: true } };
const unknown = { models: { path: '/models', free_gb: null, total_gb: null,
                            free_fraction: null, warn: false } };

console.log(JSON.stringify({
  fits: note({ size_gb: 35 }, plenty),
  does_not_fit: note({ size_gb: 400 }, tight),
  size_unknown: note({ size_gb: 0 }, plenty),
  free_unknown: note({ size_gb: 35 }, unknown),
  on_disk: note({ size_gb: 35, downloaded: true }, plenty),
  no_status_at_all: note({ size_gb: 35 }, null),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not on PATH; the app.js behaviour test needs it")
def test_the_card_says_whether_it_fits_on_this_node(tmp_path):
    from ainode.web.serve import get_static_path

    script = tmp_path / "case.js"
    script.write_text(textwrap.dedent(_CARD_CASE))
    proc = subprocess.run(
        ["node", str(script), str(get_static_path() / "js" / "app.js")],
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["fits"] == {"text": "~35 GB to fetch, 670 GB free", "cls": ""}
    assert out["does_not_fit"]["cls"] == "md-fit-no"
    assert "only 12 GB free" in out["does_not_fit"]["text"]
    # Unknown in either direction is said, never drawn as a fit or a refusal.
    assert out["size_unknown"] == {"text": "size unknown", "cls": ""}
    assert out["free_unknown"]["text"] == "~35 GB, free space unknown"
    assert out["free_unknown"]["cls"] == ""
    assert out["on_disk"]["text"] == "~35 GB on disk, 670 GB free"
    assert out["no_status_at_all"]["cls"] == ""
