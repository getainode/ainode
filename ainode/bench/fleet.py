"""Where a bench run gets its truth about the fleet.

Two callers, one set of rules:

  * ``describe_via_http`` is what ``scripts/ainode-bench.py`` uses. It reads a
    node's control endpoints over HTTP because it runs outside the process.
  * ``resolve_target`` / ``describe_from_app`` are what the ``/api/bench`` routes
    use. They resolve the engine host:port through the SAME function the proxy
    routes ``/v1/*`` with (``_routing_candidates``), so a bench can never measure
    a different instance than the one chat would have hit, and they read
    placement off the instance snapshot in cluster state plus a control-plane
    read of the node that owns the instance.

Every placement field is a read, never a guess. A field the API does not expose
is omitted, so ``bench/SCHEMA.md``'s "never fill a missing measurement with an
estimate" holds for placement as well as for the numbers.

Nothing here loads, unloads or restarts anything. The only writes this module
performs are to its own result files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ainode.bench.measure import CTL_TIMEOUT, get_json, node_sample
from ainode.metrics.collector import optional_float


# ---------------------------------------------------------------- shared

def strip_paren(name):
    return re.sub(r"\s*\([^)]*\)\s*$", "", name or "").strip()


def parse_quant(model_id):
    for q in ("NVFP4", "FP8", "MXFP4", "AWQ", "GPTQ", "INT4", "INT8", "BF16"):
        if q.lower() in (model_id or "").lower():
            return q
    return None


def derive_arch(mb: dict, model: str) -> None:
    """Fill arch/active_b/quant, in place, for whatever the catalog did not state.

    The catalog comes first: a curated entry carries ``arch`` and
    ``active_params_b`` (``ainode/models/registry.py``), which ``_apply_catalog``
    has already copied in, and those are the vendor's numbers rather than a guess.

    Only what is still missing is read off the model id, which usually states it:
    the "A3B" in 30B-A3B is the vendor's own active-parameter count, and no
    ``A<n>B`` marker means the row is recorded as dense. A model neither the
    catalog nor its id describes keeps the field absent - a MoE whose active count
    nobody stated is not silently reported as reading all of its weights.
    """
    mm = re.search(r"(?:^|[-_])A(\d+(?:\.\d+)?)B(?:[-_]|$)", model or "", re.I)
    if not mb.get("arch"):
        if mm:
            mb["arch"] = "moe"
        elif mb.get("params_b"):
            mb["arch"] = "dense"
    if mb.get("active_b") is None:
        if mb.get("arch") == "moe" and mm:
            mb["active_b"] = float(mm.group(1)) if "." in mm.group(1) else int(mm.group(1))
        elif mb.get("arch") == "dense" and mb.get("params_b"):
            mb["active_b"] = mb["params_b"]
    mb.setdefault("quant", parse_quant(model))
    if not mb.get("quant"):
        mb.pop("quant", None)


def _apply_catalog(mb: dict, pl: dict, info: dict) -> None:
    """Copy catalog metadata into the model block, and the launch recipe into
    placement only if a live read did not already supply the flags."""
    if info.get("name"):
        mb["name"] = strip_paren(info["name"])
    if info.get("params_b"):
        mb["params_b"] = info["params_b"]
    # Active params and shape, when the entry states them. These beat the id: an
    # entry like MiniMax-M2.7 is a MoE whose id carries no A<n>B marker at all, so
    # reading the id alone recorded it as dense.
    if info.get("active_params_b"):
        mb["active_b"] = info["active_params_b"]
    if info.get("arch"):
        mb["arch"] = info["arch"]
    if info.get("quantization"):
        mb["quant"] = info["quantization"]
    if info.get("context_length"):
        mb["context"] = info["context_length"]
    if info.get("license"):
        mb["license"] = info["license"]
    mb["vision"] = "vision" in (info.get("capabilities") or [])
    if "flags" not in pl:
        if info.get("engine_image"):
            pl["engine_image"] = info["engine_image"]
        if info.get("extra_vllm_args"):
            pl["flags"] = list(info["extra_vllm_args"])
            pl["flags_source"] = ("curated catalog recipe (/api/models); the live "
                                  "container command line was not readable")


def _apply_live_config(pl: dict, cfg: dict) -> None:
    """Placement fields from a node's live ``/api/config``.

    Only ever called when ``cfg["model"]`` IS the model being benched:
    ``/api/config`` is a shared mutable object that keeps the last load's
    overrides, so reading it for a different model would record the wrong flags.
    """
    if cfg.get("engine_image"):
        pl["engine_image"] = cfg["engine_image"]
    if cfg.get("extra_vllm_args"):
        pl["flags"] = list(cfg["extra_vllm_args"])
    if cfg.get("gpu_memory_utilization") is not None:
        pl["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    if cfg.get("kv_cache_dtype"):
        pl["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    if cfg.get("max_model_len"):
        pl["max_model_len"] = cfg["max_model_len"]
    if cfg.get("distributed_mode"):
        pl["distributed_mode"] = cfg["distributed_mode"]
    pl["flags_source"] = "live node config (/api/config)"
    if cfg.get("extra_env"):
        pl["extra_env"] = dict(cfg["extra_env"])


# ---------------------------------------------------------------- serving-node resolution (CLI)

def resolve_serving_node(base_url: str, model: str):
    """Which node serves ``model``, from the master's fleet view.

    Reads ``/api/server/status`` on the master to find the node that hosts the
    loaded model. Returns ``(node_name, engine_port, gpu_name, warn)``.
    ``node_name`` is "" when the model is not loaded anywhere; ``warn``
    explains an unreadable master. The caller keeps describing the fleet
    through the master: a peer's own web port is only reachable over the
    cluster fabric, which the bench CLI usually is not on, so the record's
    placement is corrected by name rather than by re-addressing the request.
    """
    base = base_url.rstrip("/")
    ss = get_json(f"{base}/api/server/status")
    if ss.get("_error"):
        return "", None, "", f"master /api/server/status unreadable: {ss['_error']}"
    node_name = ""
    engine_port = None
    for m in (ss.get("loaded_models") or []):
        if m.get("id") == model:
            node_name = m.get("node_hostname") or ""
            engine_port = m.get("port") or 8000
            break
    if not node_name:
        return "", None, "", ""
    gpu_name = ""
    nodes = get_json(f"{base}/api/nodes")
    if not nodes.get("_error"):
        for n in (nodes.get("nodes") if isinstance(nodes, dict) else nodes) or []:
            if n.get("node_name") == node_name:
                gpu_name = n.get("gpu_name") or ""
                break
    return node_name, engine_port, gpu_name, ""


# ---------------------------------------------------------------- HTTP describe (CLI)

def describe_via_http(ainode, engine_url, model):
    """Build the model + placement blocks from what the fleet actually reports.

    The out-of-process path, used by ``scripts/ainode-bench.py``. Returns
    (model_block, placement, node_id, warnings).
    """
    warn = []
    mb = {"id": model}
    pl = {"engine": "vllm"}
    node_id = None

    # --- engine itself: served context window (authoritative, it is serving)
    ml = get_json(engine_url.rstrip("/") + "/v1/models", timeout=CTL_TIMEOUT)
    for entry in (ml.get("data") or []):
        if entry.get("id") == model and entry.get("max_model_len"):
            pl["max_model_len"] = entry["max_model_len"]
    if ml.get("_error"):
        warn.append(f"engine /v1/models unreadable: {ml['_error']}")

    if not ainode:
        return mb, pl, node_id, warn
    base = ainode.rstrip("/")

    # --- the node we are hitting
    st = get_json(f"{base}/api/status")
    if st.get("_error"):
        warn.append(f"/api/status unreadable: {st['_error']}")
    else:
        node_id = st.get("node_id")
        if st.get("node_name"):
            pl["node"] = st["node_name"]
        if (st.get("gpu") or {}).get("name"):
            pl["gpu"] = st["gpu"]["name"]
        if st.get("version"):
            pl["ainode"] = st["version"]
        serves = [st.get("model")] + list(st.get("models_loaded") or [])
        if model not in serves:
            warn.append(f"{pl.get('node', 'this AINode')} does not report serving {model}; "
                        "placement and telemetry may describe the wrong node")

    # --- tensor parallelism and what else shares the node
    ss = get_json(f"{base}/api/server/status")
    stacked = []
    for m in (ss.get("loaded_models") or []):
        if m.get("id") == model:
            if m.get("parallel"):
                pl["tp"] = m["parallel"]
                pl["gpus"] = m["parallel"]
        elif node_id and m.get("node_id") == node_id:
            stacked.append(m["id"])
    pl["stacked_with"] = stacked
    pl.setdefault("tp", 1)
    pl.setdefault("gpus", 1)

    # --- launch flags. /api/config is the LIVE config of this node's primary
    # instance, but it is a shared mutable object that keeps the last load's
    # overrides, so it is only trustworthy when its own `model` is the model we
    # are benching. Otherwise fall back to the curated catalog recipe and say so.
    cfg = get_json(f"{base}/api/config")
    if not cfg.get("_error") and cfg.get("model") == model:
        _apply_live_config(pl, cfg)

    # --- model metadata from the catalog, matched on hf_repo or catalog id
    cat = get_json(f"{base}/api/models", timeout=30)
    info = None
    for m in (cat.get("models") or []):
        if model in (m.get("hf_repo"), m.get("id")):
            info = m
            break
    if info:
        _apply_catalog(mb, pl, info)
    else:
        warn.append(f"{model} is not in the AINode catalog; fill model metadata "
                    "(params_b, license, context) by hand")

    derive_arch(mb, model)
    return mb, pl, node_id, warn


# ---------------------------------------------------------------- in-process (routes)

@dataclass
class BenchTarget:
    """The one loaded instance a run points at."""

    model: str
    host: str
    port: int
    node_id: str = ""
    node_name: str = ""
    web_port: int = 3000
    is_local: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def node_url(self) -> str:
        return f"http://{self.host}:{self.web_port}"


def _owning_node(cluster, local_node_id, host, port):
    """The cluster member that serves (host, port), or None.

    ``_routing_candidates`` hands back addresses, not nodes, and placement needs
    the node. Matched the same way the candidate was built: the local node is
    addressed as "localhost", peers by fabric IP.
    """
    for n in (cluster.members() if cluster is not None else []):
        is_local = n.node_id == local_node_id
        nhost = "localhost" if is_local else (getattr(n, "fabric_ip", "") or "")
        if nhost != host:
            continue
        ports = {getattr(n, "api_port", 0)}
        for inst in (getattr(n, "instances", []) or []):
            if isinstance(inst, dict) and inst.get("api_port"):
                ports.add(inst["api_port"])
        # The local node's own engine answers on config.api_port, which is what
        # the candidate carries; accept any port this node advertises.
        if port in ports or is_local:
            return n
    return None


def resolve_target(app, model: str):
    """The (host, port) a bench of ``model`` must hit, or None if nothing serves it.

    Delegates to the proxy's own ``_routing_candidates`` so a bench measures the
    instance a chat request would have reached. Imported lazily: the route module
    is registered *by* ``ainode.api.server``, so a module-level import would be a
    cycle.
    """
    from ainode.api.server import _routing_candidates

    config = app["config"]
    cluster = app.get("cluster_state")
    cands = _routing_candidates(cluster, model, config.node_id, config.api_port)
    if not cands:
        return None
    host, port = cands[0]
    node = _owning_node(cluster, config.node_id, host, port)
    is_local = host == "localhost"
    return BenchTarget(
        model=model, host=host, port=port,
        node_id=(node.node_id if node else (config.node_id if is_local else "")),
        node_name=(node.node_name if node else (config.node_name if is_local else host)),
        web_port=(getattr(config, "web_port", 3000) if is_local
                  else (getattr(node, "web_port", 3000) if node else 3000)),
        is_local=is_local,
    )


async def _aget_json(session, url, timeout=5.0):
    """GET JSON in-process. Same contract as measure.get_json: never raises."""
    if session is None:
        return {"_error": "no client session"}
    try:
        import aiohttp
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                return {"_error": f"HTTP {r.status}"}
            return await r.json(content_type=None)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:140]}"}


async def probe_ready(app, target: BenchTarget) -> tuple[bool, str]:
    """Is the target instance actually serving the model right now?

    Probes the engine's own ``/v1/models``, which is the readiness signal the
    rest of this codebase treats as authoritative: an instance record's status is
    a latch stamped when the engine first answered, so a crashed engine keeps
    reading "serving" indefinitely. A bench that started against a dead engine
    would write a result file full of connection errors.
    """
    d = await _aget_json(app.get("client_session"), f"{target.url}/v1/models", timeout=5.0)
    if d.get("_error"):
        return False, (f"{target.node_name or target.host}:{target.port} did not answer "
                       f"/v1/models ({d['_error']})")
    ids = [m.get("id") for m in (d.get("data") or [])]
    if target.model not in ids:
        served = ", ".join(i for i in ids if i) or "nothing"
        return False, (f"{target.node_name or target.host}:{target.port} is serving "
                       f"{served}, not {target.model}")
    return True, ""


def _catalog_entry(model: str):
    """Curated catalog metadata for a model id, or None.

    Reads the in-process catalog dicts rather than ``/api/models``: that endpoint
    runs the whole aggregator (HF sweep, on-disk scan) in an executor, which is
    far too much work to do before a benchmark and can block on the network.
    The launch recipe this needs lives in the curated tables either way.
    """
    from ainode.models.registry import CURATED_CLUSTER_MODELS, MODEL_CATALOG

    for table in (CURATED_CLUSTER_MODELS, MODEL_CATALOG):
        for cid, info in (table or {}).items():
            if model in (getattr(info, "hf_repo", ""), cid):
                return info.to_dict()
    return None


def _target_tp(app, target: BenchTarget, node) -> int:
    """Tensor-parallel width of the target instance.

    Local: the InstanceManager record, which is the launch truth. Remote: the
    node's own distributed announcement (head + peers). Anything we cannot read
    is recorded as 1 rather than guessed upward.
    """
    if target.is_local:
        manager = app.get("instances")
        if manager is not None:
            try:
                for inst in manager.instances():
                    rec = inst.record
                    if rec.model == target.model and rec.api_port == target.port:
                        return max(1, int(rec.tensor_parallel_size or 1))
            except Exception:
                pass
    if node is not None and getattr(node, "model", "") == target.model:
        if (getattr(node, "distributed_mode", "solo") or "solo") == "head":
            return 1 + len(list(getattr(node, "distributed_peers", []) or []))
    return 1


def _stacked_with(cluster, target: BenchTarget) -> list:
    """Other models loaded on the same node. They share the node's memory
    bandwidth, which is the whole ceiling on this hardware, so a result that does
    not name them is not reproducible."""
    out = []
    for n in (cluster.members() if cluster is not None else []):
        if n.node_id != target.node_id:
            continue
        if getattr(n, "model", "") and n.model != target.model:
            out.append(n.model)
        for inst in (getattr(n, "instances", []) or []):
            if not isinstance(inst, dict):
                continue
            m = inst.get("model")
            if m and m != target.model and m not in out:
                out.append(m)
    return out


async def describe_from_app(app, target: BenchTarget):
    """Model + placement blocks for an in-process run.

    Returns (model_block, placement, warnings). Placement comes from the instance
    snapshot in cluster state (node, GPU, tp, what else is stacked) plus a
    control-plane read of the node that owns the instance (engine image, vLLM
    flags, gmu, KV dtype) over the same fabric address the master already uses to
    dispatch loads. Read-only throughout.
    """
    from ainode import __version__

    config = app["config"]
    cluster = app.get("cluster_state")
    session = app.get("client_session")
    node = _owning_node(cluster, config.node_id, target.host, target.port)

    warn = []
    mb = {"id": target.model}
    pl = {"engine": "vllm"}

    # --- the engine itself: served context window (it is serving, so it wins)
    ml = await _aget_json(session, f"{target.url}/v1/models", timeout=5.0)
    for entry in (ml.get("data") or []):
        if entry.get("id") == target.model and entry.get("max_model_len"):
            pl["max_model_len"] = entry["max_model_len"]
    if ml.get("_error"):
        warn.append(f"engine /v1/models unreadable: {ml['_error']}")

    # --- node identity and GPU, from the instance snapshot
    if node is not None:
        if getattr(node, "node_name", ""):
            pl["node"] = node.node_name
        if getattr(node, "gpu_name", ""):
            pl["gpu"] = node.gpu_name
        if (getattr(node, "distributed_mode", "solo") or "solo") != "solo":
            pl["distributed_mode"] = node.distributed_mode
    elif target.is_local:
        pl["node"] = config.node_name

    tp = _target_tp(app, target, node)
    pl["tp"] = tp
    pl["gpus"] = tp
    pl["stacked_with"] = _stacked_with(cluster, target)

    # --- launch flags, from the node that owns the instance. Only trusted when
    # that node's live config names the model we are benching (see
    # _apply_live_config); otherwise the catalog recipe fills in and says so.
    st = await _aget_json(session, f"{target.node_url}/api/status", timeout=5.0)
    if not st.get("_error") and st.get("version"):
        pl["ainode"] = st["version"]
    elif target.is_local:
        pl["ainode"] = __version__
    cfg = await _aget_json(session, f"{target.node_url}/api/config", timeout=5.0)
    if not cfg.get("_error") and cfg.get("model") == target.model:
        _apply_live_config(pl, cfg)

    info = _catalog_entry(target.model)
    if info:
        _apply_catalog(mb, pl, info)
    else:
        warn.append(f"{target.model} is not in the AINode catalog; fill model metadata "
                    "(params_b, license, context) by hand")
    derive_arch(mb, target.model)
    return mb, pl, warn


async def busy_warnings(app, target: BenchTarget) -> list:
    """Reasons this run will measure a queue as well as a model.

    Not a refusal. Benching a node that is serving live traffic is a legitimate
    thing to want to do; silently reporting the result as if the node had been
    idle is not. Both signals are reads of the target node.
    """
    out = []
    cluster = app.get("cluster_state")
    stacked = _stacked_with(cluster, target)
    if stacked:
        out.append(f"{target.node_name or target.host} is also serving "
                   f"{', '.join(stacked)}; co-resident models share the node's memory "
                   "bandwidth, which is the ceiling on this hardware")
    ss = await _aget_json(app.get("client_session"),
                          f"{target.node_url}/api/server/status", timeout=5.0)
    rpm = ss.get("request_count_last_minute")
    if isinstance(rpm, int) and rpm > 0:
        out.append(f"{target.node_name or target.host} served {rpm} request(s) in the last "
                   "minute; a busy node measures the queue as well as the engine")
    return out


# ---------------------------------------------------------------- telemetry readers

def cluster_nodes_reader(app, node_id: str):
    """Telemetry reader for an in-process run.

    Reads the same four fields ``/api/nodes`` renders, from the same place that
    handler reads them: the node's broadcast-carried GPU telemetry in cluster
    state, and the local metrics collector for this node (whose ClusterNode is
    built once at startup). Read in-process rather than over HTTP so telemetry
    does not depend on the API key configuration of the node benching itself.
    """
    if not node_id:
        return None
    config = app["config"]
    cluster = app.get("cluster_state")
    collector = app.get("metrics_collector")

    def read():
        node = None
        try:
            for n in (cluster.members() if cluster is not None else []):
                if n.node_id == node_id:
                    node = n
                    break
        except RuntimeError:
            # Cluster state mutated mid-iteration by the discovery sync. A lost
            # sample is a missing sample, never a failed run.
            return None
        if node is None:
            return None
        # A figure this node cannot measure stays None all the way into the
        # record. Coerced to 0 it becomes a measurement: a bench record claiming
        # an idle GPU and an empty node while the run was in flight, which is
        # exactly what a record must never contain.
        used_mb = optional_float(getattr(node, "gpu_memory_used_mb", None))
        total_mb = optional_float(getattr(node, "gpu_memory_total_mb", None))
        util = optional_float(getattr(node, "gpu_utilization", None))
        temp = optional_float(getattr(node, "gpu_temp", None))
        if node_id == config.node_id and collector is not None:
            try:
                m = collector.get_gpu_metrics() or {}
                if not m.get("error"):
                    used_mb = optional_float(m.get("memory_used_mb"))
                    total_mb = optional_float(m.get("memory_total_mb")) or total_mb
                    util = optional_float(m.get("utilization_percent"))
                    temp = optional_float(m.get("temperature_c"))
            except Exception:
                pass
        if not total_mb and getattr(node, "gpu_memory_gb", 0):
            total_mb = float(node.gpu_memory_gb) * 1024
        if not total_mb:
            return None
        return node_sample({
            "gpu_memory_gb": total_mb / 1024,
            "gpu_memory_used_pct": (round(used_mb / total_mb * 100)
                                    if used_mb is not None else None),
            "gpu_utilization": (round(util) if util is not None else None),
            "gpu_temp": (round(temp) if temp is not None else None),
        })

    return read
