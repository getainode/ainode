"""Chat-view routes: the per-instance model card and the capability probe.

The chat view needs two answers the rest of the API does not already give:

1. What is actually behind the model the user picked: which node, which GPU,
   how much VRAM, TP, quantization, engine image, speculative decoding, and the
   context length the engine itself reports. ``GET /api/models/card`` assembles
   that per instance so the browser makes ONE call instead of stitching
   /api/nodes + /api/server/status + /api/config + the engine's /v1/models
   together in JavaScript.

2. Whether that engine will accept an image or a tools array:
   ``GET /api/models/caps``. Probed with a real (one-token) request, never
   inferred from the model name: we serve models whose weights are multimodal
   with vision off, and no name tells you that. Reasoning is deliberately NOT
   probed: a model that chooses not to think on one prompt is not a model that
   cannot think, so the chat view marks reasoning from turns it actually saw.

Both routes are read-only inference. Neither loads, unloads nor restarts
anything. Any field we cannot see is ``null``, never guessed.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

# Node states that can serve traffic: the same set _routing_candidates accepts,
# so the card describes the instance the proxy would actually route to.
_SERVING_STATES = ("online", "serving", "member-ready")

# A 16x16 mid-gray PNG. NOT 1x1: several engines reject an image smaller than
# ~10px during preprocessing, which reads as "vision refused" when the real
# answer is "that image was too small to grade". 16x16 costs a handful of
# prompt tokens and gets a truthful accept/refuse.
_PROBE_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFElEQVR42mNoIBEwjGoY"
    "1TB8NQAAJYSAELxv8c8AAAAASUVORK5CYII="
)

# Flags that mean "this instance runs speculative decoding". vLLM has spelled it
# several ways across the versions we ship (0.17 --speculative-config,
# 0.27 --speculative_config.model), so match on the prefix rather than one name.
_SPECULATIVE_PREFIXES = (
    "--speculative-config",
    "--speculative_config",
    "--speculative-model",
    "--speculative_model",
    "--num-speculative-tokens",
    "--num_speculative_tokens",
    "--speculative-draft",
)

# Quantization names that appear verbatim in a model id. Read off the id only as
# a FALLBACK when the catalog has no entry, and reported with its source so the
# UI never presents a parsed id as catalog truth.
_QUANT_TOKENS = (
    "NVFP4", "MXFP4", "FP8", "FP4", "AWQ", "GPTQ", "GGUF",
    "W4A16", "W8A8", "INT4", "INT8", "BNB",
)

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _status_of(node) -> str:
    status = getattr(node, "status", "")
    return status.value if hasattr(status, "value") else str(status)


def fleet_instances(app) -> list[dict]:
    """Every (model, node, port) the fleet is serving right now.

    Built from cluster state, the same source ``_routing_candidates`` routes on
    and ``/api/server/status`` lists, so the card cannot describe an instance
    the proxy would not reach. The local node's own InstanceManager is merged in
    as well: a stacked model that has just launched is in the manager before the
    next broadcast tick puts it in our announcement.
    """
    config = app.get("config")
    cluster = app.get("cluster_state")
    local_id = getattr(config, "node_id", None)
    local_port = getattr(config, "api_port", 8000)
    out: list[dict] = []
    seen: set = set()

    def add(entry: dict) -> None:
        if not entry["model"]:
            return
        key = (entry["model"], entry["node_id"], entry["port"])
        if key not in seen:
            seen.add(key)
            out.append(entry)
            return
        # Same instance seen twice: a node advertises its primary model both as
        # `model` and as an entry in `instances`. Keep the one carrying the
        # instance record, which is where TP and the member node set live.
        if entry.get("record"):
            for i, existing in enumerate(out):
                if (existing["model"], existing["node_id"], existing["port"]) == key \
                        and not existing.get("record"):
                    out[i] = entry
                    return

    for node in (cluster.members() if cluster is not None else []):
        if _status_of(node) not in _SERVING_STATES:
            continue
        is_local = node.node_id == local_id
        host = "localhost" if is_local else (getattr(node, "fabric_ip", "") or "")
        if not host:
            continue  # remote node with no fabric IP is unroutable, so unreportable
        node_port = local_port if is_local else node.api_port
        base = {
            "node_id": node.node_id,
            "node_name": node.node_name,
            "host": host,
            "web_port": getattr(node, "web_port", 3000),
            "local": is_local,
            "gpu_name": getattr(node, "gpu_name", "") or None,
            "gpu_memory_gb": getattr(node, "gpu_memory_gb", None) or None,
            "unified_memory": getattr(node, "unified_memory", None),
        }
        if getattr(node, "model", ""):
            add({**base, "model": node.model, "port": node_port, "record": None})
        for inst in (getattr(node, "instances", []) or []):
            if not isinstance(inst, dict) or not inst.get("model"):
                continue
            add({**base, "model": inst["model"],
                 "port": inst.get("api_port") or node_port, "record": inst})

    manager = app.get("instances")
    if manager is not None and config is not None:
        try:
            for inst in manager.instances():
                rec = inst.record
                if not rec.model:
                    continue
                add({
                    "model": rec.model,
                    "node_id": getattr(config, "node_id", None),
                    "node_name": getattr(config, "node_name", None),
                    "host": "localhost",
                    "web_port": getattr(config, "web_port", 3000),
                    "local": True,
                    "gpu_name": None,
                    "gpu_memory_gb": None,
                    "unified_memory": None,
                    "port": rec.api_port,
                    "record": rec.to_dict(),
                })
        except Exception:  # pragma: no cover - defensive
            logger.exception("chat card: failed to read the local instance manager")
    return out


def _pick_instance(entries: list[dict], model: str,
                   node_id: str = "", port: str = "") -> Optional[dict]:
    """Pick the instance the caller asked about.

    The picker sends node_id+port because one model id can be served on several
    nodes at once. With neither, fall back to the first entry serving the model
    (local first, same order the proxy tries).
    """
    matches = [e for e in entries if e["model"] == model]
    if node_id:
        matches = [e for e in matches if e["node_id"] == node_id] or matches
    if port:
        try:
            want = int(port)
            matches = [e for e in matches if e["port"] == want] or matches
        except (TypeError, ValueError):
            pass
    matches.sort(key=lambda e: not e.get("local"))
    return matches[0] if matches else None


def _local_instance_config(app, entry: dict):
    """The per-instance NodeConfig snapshot behind a LOCAL instance.

    Each stacked model launches with its own config snapshot (its own engine
    image, extra flags, KV dtype), so the shared app config is only right for
    the primary. Matched on port, which is what makes the instance unique.
    """
    manager = app.get("instances")
    if manager is not None:
        try:
            for inst in manager.instances():
                if inst.record.api_port == entry["port"] and inst.record.model == entry["model"]:
                    cfg = getattr(inst.backend, "config", None)
                    if cfg is not None:
                        return cfg
        except Exception:  # pragma: no cover - defensive
            logger.exception("chat card: failed to read a local instance config")
    config = app.get("config")
    if config is not None and getattr(config, "model", None) == entry["model"] \
            and getattr(config, "api_port", None) == entry["port"]:
        return config
    return None


async def _remote_node_config(session, entry: dict) -> Optional[dict]:
    """A remote node's own live config, used only when it describes THIS model.

    A peer publishes its config over /api/config. That config describes the
    node's primary model, so it is only usable here when config.model matches
    the instance we are describing. Otherwise we would be showing model A's
    engine image on model B's card.
    """
    if session is None:
        return None
    url = f"http://{entry['host']}:{entry['web_port']}/api/config"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=4)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("model") != entry["model"]:
        return None
    return data


async def engine_model_meta(session, host: str, port: int, model: str) -> Optional[dict]:
    """What the engine says about the model it is serving.

    The live context window is the engine's ``max_model_len`` and nothing else:
    config.max_model_len is frequently unset (the engine derives it from the
    checkpoint), and a catalog number is a guess about a different build. Fetched
    server-side so the browser still makes a single call for the card.
    """
    if session is None:
        return None
    url = f"http://{host}:{port}/v1/models"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=4)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None
    entries = (data or {}).get("data") or []
    for m in entries:
        if m.get("id") == model:
            return m
    # A served_model_name alias means the engine answers under a different id
    # than the one we route on; with exactly one model served it is still that
    # model's metadata.
    return entries[0] if len(entries) == 1 else None


def catalog_entry(model: str):
    """The catalog's own entry for a model id or HF repo, or None.

    Only a real catalog hit gives us a description, a repo link and a
    quantization we did not have to parse out of a name.
    """
    try:
        from ainode.models.registry import CURATED_CLUSTER_MODELS, FALLBACK_CATALOG
    except Exception:  # pragma: no cover - registry import guarded for tests
        return None
    m = (model or "").strip()
    if not m:
        return None
    for source in (CURATED_CLUSTER_MODELS, FALLBACK_CATALOG):
        for info in source.values():
            if m in (getattr(info, "id", None), getattr(info, "hf_repo", None)):
                return info
    return None


def _quant_from_model_id(model: str) -> Optional[str]:
    """Quantization named in the model id itself (…-NVFP4, …-AWQ).

    Not an inference about the weights: a published id that says NVFP4 is the
    publisher stating the format. Reported with source="model_id" so the UI can
    say where it came from.
    """
    upper = (model or "").upper()
    for token in _QUANT_TOKENS:
        if re.search(rf"(?:^|[^A-Z0-9]){token}(?:$|[^A-Z0-9])", upper):
            return token
    return None


def _speculative(extra_args) -> Optional[str]:
    """The speculative-decoding flags this instance was launched with, or None."""
    args = [str(a) for a in (extra_args or [])]
    hits = [a for a in args if any(a.startswith(p) for p in _SPECULATIVE_PREFIXES)]
    if not hits:
        return None
    # Include the value that follows a bare flag so the card can show the draft
    # model / token count rather than just "on".
    out: list[str] = []
    for i, arg in enumerate(args):
        if any(arg.startswith(p) for p in _SPECULATIVE_PREFIXES):
            out.append(arg)
            if "=" not in arg and i + 1 < len(args) and not args[i + 1].startswith("--"):
                out.append(args[i + 1])
    return " ".join(out)


def _instance_nodes(record: Optional[dict], entry: dict) -> Optional[list]:
    """Node names an instance spans, when the instance record tells us."""
    if not record:
        return None
    members = [n for n in (record.get("member_node_ids") or []) if n]
    if members:
        return members
    peers = [p for p in (record.get("peer_ips") or []) if p]
    if peers:
        return [entry["node_id"]] + peers
    return None


async def handle_model_card(request: web.Request) -> web.Response:
    """GET /api/models/card?model=<id>[&node_id=<id>][&port=<n>]

    One JSON blob describing the instance behind a picked model. Unknown fields
    are null, and every derived field carries the source it came from.
    """
    model = (request.query.get("model") or "").strip()
    if not model:
        return web.json_response(
            {"error": {"message": "model query parameter required",
                       "type": "invalid_request"}}, status=400)

    app = request.app
    entries = fleet_instances(app)
    entry = _pick_instance(entries, model, request.query.get("node_id", ""),
                           request.query.get("port", ""))
    if entry is None:
        return web.json_response(
            {"error": {"message": f"'{model}' is not being served by any node",
                       "type": "model_not_found"}}, status=404)

    session = app.get("client_session")
    record = entry.get("record")

    # --- how it is served ------------------------------------------------
    cfg = _local_instance_config(app, entry) if entry.get("local") else None
    config_source = None
    engine_image = kv_cache_dtype = engine_backend = None
    gmu = max_model_len_cfg = None
    extra_args: list = []
    quant_cfg = None
    if cfg is not None:
        config_source = "instance"
        engine_image = getattr(cfg, "engine_image", "") or None
        kv_cache_dtype = getattr(cfg, "kv_cache_dtype", "") or None
        engine_backend = getattr(cfg, "engine_backend", "") or None
        gmu = getattr(cfg, "gpu_memory_utilization", None)
        max_model_len_cfg = getattr(cfg, "max_model_len", None)
        extra_args = [str(a) for a in (getattr(cfg, "extra_vllm_args", None) or [])]
        quant_cfg = getattr(cfg, "quantization", None) or None
    elif not entry.get("local"):
        remote = await _remote_node_config(session, entry)
        if remote:
            config_source = "node_config"
            engine_image = remote.get("engine_image") or None
            kv_cache_dtype = remote.get("kv_cache_dtype") or None
            engine_backend = remote.get("engine_backend") or None
            gmu = remote.get("gpu_memory_utilization")
            max_model_len_cfg = remote.get("max_model_len")
            extra_args = [str(a) for a in (remote.get("extra_vllm_args") or [])]
            quant_cfg = remote.get("quantization") or None

    # --- what the engine itself reports ----------------------------------
    live = await engine_model_meta(session, entry["host"], entry["port"], model)
    max_model_len = (live or {}).get("max_model_len")

    # --- catalog ----------------------------------------------------------
    info = catalog_entry(model)
    quant, quant_source = None, None
    if quant_cfg:
        quant, quant_source = quant_cfg, config_source
    elif info is not None and getattr(info, "quantization", None):
        quant, quant_source = info.quantization, "catalog"
    else:
        parsed = _quant_from_model_id(model)
        if parsed:
            quant, quant_source = parsed, "model_id"

    hf_repo = getattr(info, "hf_repo", None) if info is not None else None
    hf_source = "catalog" if hf_repo else None
    if not hf_repo and _REPO_RE.match(model):
        hf_repo, hf_source = model, "model_id"

    nodes = _instance_nodes(record, entry)
    tp = (record or {}).get("tensor_parallel_size")
    gpu_count = len(nodes) if nodes else (tp if tp and tp > 1 else None)
    if gpu_count is None and entry.get("gpu_name"):
        gpu_count = 1  # AINode models one GPU per node; the node reported one

    return web.json_response({
        "model": model,
        "instance": {
            "node_id": entry["node_id"],
            "node_name": entry["node_name"],
            "port": entry["port"],
            "local": bool(entry.get("local")),
            "nodes": nodes,
            "instance_id": (record or {}).get("instance_id"),
            "status": (record or {}).get("status"),
        },
        "hardware": {
            "gpu_name": entry.get("gpu_name"),
            "gpu_count": gpu_count,
            "gpu_memory_gb": entry.get("gpu_memory_gb"),
            "unified_memory": entry.get("unified_memory"),
        },
        "serving": {
            "tensor_parallel_size": tp,
            "quantization": quant,
            "quantization_source": quant_source,
            "engine_image": engine_image,
            "engine_backend": engine_backend,
            "kv_cache_dtype": kv_cache_dtype,
            "gpu_memory_utilization": gmu,
            "speculative": _speculative(extra_args),
            "extra_vllm_args": extra_args or None,
            "config_source": config_source,
            "max_model_len": max_model_len,
            "max_model_len_source": "engine" if max_model_len else None,
            "max_model_len_configured": max_model_len_cfg,
        },
        "catalog": None if info is None else {
            "id": getattr(info, "id", None),
            "name": getattr(info, "name", None),
            "description": getattr(info, "description", None) or None,
            "params_b": getattr(info, "params_b", None) or None,
            "family": getattr(info, "family", None) or None,
            "context_length": getattr(info, "context_length", None) or None,
            "capabilities": list(getattr(info, "capabilities", None) or []) or None,
            "verified": bool(getattr(info, "verified", False)),
        },
        "hf_repo": hf_repo,
        "hf_url": f"https://huggingface.co/{hf_repo}" if hf_repo else None,
        "hf_source": hf_source,
    })


async def _probe(session, host: str, port: int, body: dict, timeout: float):
    """One probe request. Returns (status, payload) or (None, {}) if unreachable."""
    url = f"http://{host}:{port}/v1/chat/completions"
    try:
        async with session.post(url, json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            try:
                data = await resp.json(content_type=None)
            except Exception:
                data = {}
            return resp.status, (data if isinstance(data, dict) else {})
    except Exception as exc:
        return None, {"_transport": str(exc)}


def _error_text(payload: dict) -> Optional[str]:
    """The engine's OWN words for a refusal, trimmed for a badge tooltip."""
    err = payload.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
    elif isinstance(err, str):
        msg = err
    else:
        msg = payload.get("message") or payload.get("_transport")
    if not msg:
        return None
    msg = " ".join(str(msg).split())
    return msg[:300]


async def probe_caps(session, host: str, port: int, model: str) -> dict:
    """Ask the engine what it will accept, rather than reading the name.

    Two real requests capped at one token each: an image part, and a tools
    array. Structural: the engine either takes them or rejects them, and its
    rejection text is the honest reason (``not a multimodal model``,
    ``no tool-call parser``). Vision is served off on models whose weights
    support it, so this is the only way to be right.
    """
    caps: dict = {"vision": None, "tools": None,
                  "vision_error": None, "tools_error": None}
    if session is None:
        return caps

    status, payload = await _probe(session, host, port, {
        "model": model, "max_tokens": 1,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": _PROBE_PNG}}]}],
    }, 60)
    if status is not None:
        caps["vision"] = status == 200
        if status != 200:
            caps["vision_error"] = _error_text(payload)
    else:
        caps["vision_error"] = _error_text(payload)

    status, payload = await _probe(session, host, port, {
        "model": model, "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "ping", "description": "probe",
            "parameters": {"type": "object", "properties": {}}}}],
    }, 45)
    if status is not None:
        caps["tools"] = status == 200
        if status != 200:
            caps["tools_error"] = _error_text(payload)
    else:
        caps["tools_error"] = _error_text(payload)
    return caps


async def handle_model_caps(request: web.Request) -> web.Response:
    """GET /api/models/caps?model=<id>[&node_id=][&port=][&fresh=1]

    Cached per (node, port, model) because each probe is a real engine request;
    ``fresh=1`` drops the cached answer first, which is what you want after an
    engine relaunch changed the flags.
    """
    model = (request.query.get("model") or "").strip()
    if not model:
        return web.json_response(
            {"error": {"message": "model query parameter required",
                       "type": "invalid_request"}}, status=400)

    app = request.app
    entry = _pick_instance(fleet_instances(app), model,
                           request.query.get("node_id", ""),
                           request.query.get("port", ""))
    if entry is None:
        return web.json_response(
            {"error": {"message": f"'{model}' is not being served by any node",
                       "type": "model_not_found"}}, status=404)

    cache = app.setdefault("chat_caps_cache", {})
    key = (entry["node_id"], entry["port"], model)
    if request.query.get("fresh"):
        cache.pop(key, None)
    if key not in cache:
        cache[key] = await probe_caps(app.get("client_session"),
                                      entry["host"], entry["port"], model)
        cache[key]["probed"] = True
    caps = dict(cache[key])
    caps.update({
        "model": model,
        "node_id": entry["node_id"],
        "node_name": entry["node_name"],
        "port": entry["port"],
        # Reasoning is never probed: one prompt cannot separate "will not think"
        # from "did not think this time". The client marks it from turns seen.
        "reasoning": None,
        "reasoning_note": "marked from turns observed, not probed",
    })
    return web.json_response(caps)


def register_chat_routes(app: web.Application) -> None:
    """Register the chat view's two routes.

    Must be registered BEFORE register_model_routes: aiohttp resolves in
    registration order and ``/api/models/{model_id}`` would otherwise swallow
    both of these paths.
    """
    app.router.add_get("/api/models/card", handle_model_card)
    app.router.add_get("/api/models/caps", handle_model_caps)
