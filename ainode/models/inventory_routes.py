"""Live local and fleet inventory for Catalog."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from ipaddress import ip_address

import aiohttp
from aiohttp import web

from ainode.api.engine_headers import engine_headers
from ainode.api.server_routes import _probe_loaded_models
from ainode.auth.fleet import fleet_headers
from ainode.auth.middleware import is_authenticated
from ainode.discovery.broadcast import NodeStatus
from ainode.models.inventory import ModelCopy, scan_model_copies

logger = logging.getLogger(__name__)

NODE_INVENTORY_VERSION = 1
SCAN_TIMEOUT = 3
PEER_TIMEOUT = 6
MODEL_STATUSES = {"available", "incomplete", "downloading", "missing", "failed"}
ROW_TEXT_FIELDS = ("id", "layout", "revision", "file")


def _matches(row: ModelCopy, model: str) -> bool:
    return row.id == model or (row.layout == "Model store"
                               and row.id == model.replace("--", "/", 1))


def _apply_download_jobs(copies: list[ModelCopy], jobs: dict, max_age: float) -> None:
    now = time.time()
    latest = {}
    for job in jobs.values():
        finished = job.get("finished_at")
        if finished is None or now - finished <= max_age:
            latest[job.get("hf_repo") or job["model_id"]] = job["status"]
    for model_id, status in latest.items():
        direct = [row for row in copies if row.layout == "Model store" and _matches(row, model_id)]
        if status == "downloading":
            if direct:
                for row in direct:
                    row.status = "downloading"
            else:
                copies.append(ModelCopy(model_id, "Model store", "downloading"))
        elif not direct and status in ("complete", "completed", "failed"):
            copies.append(ModelCopy(model_id, "Model store",
                                    "failed" if status == "failed" else "missing"))


async def _apply_loaded_state(request: web.Request, copies: list[ModelCopy]) -> None:
    app = request.app
    config = app["config"]
    primary = app.get("engine")
    primary_model = getattr(getattr(primary, "config", None), "model", None) or config.model
    records = {}
    if primary is not None and primary_model:
        records[config.api_port] = (primary_model, getattr(primary, "config", config))
    manager = app.get("instances")
    if manager is not None:
        for instance in manager.instances():
            record = instance.record
            if record.model:
                records[record.api_port] = (record.model, getattr(instance.backend, "config", None))
    targets = {model for model, _ in records.values()}
    if primary_model:
        targets.add(primary_model)

    available = [row for row in copies if row.status == "available"]
    # A distributed worker may hold a headless rank with no local serving port.
    # With multiple copies of one ID the engine's public ID cannot identify a file.
    if config.distributed_mode != "member":
        for row in available:
            if not any(_matches(row, target) for target in targets):
                row.loaded = False

    session = app.get("client_session")
    headers = engine_headers(request)

    async def mark_loaded(port, model, launch_config) -> None:
        matching = [row for row in available if _matches(row, model)]
        if len(matching) != 1:
            return
        names = getattr(launch_config, "served_model_name", None) or []
        expected = {model, matching[0].id, *names}
        served_ids = await _probe_loaded_models(session, port, headers=headers)
        if expected.intersection(served_ids):
            matching[0].loaded = True

    await asyncio.gather(*(mark_loaded(port, *record) for port, record in records.items()))


def _node_result(node, status: str, copies: list | None = None) -> dict:
    return {
        "node_id": node.node_id,
        "node_name": node.node_name or node.node_id,
        "status": status,
        "models": copies if copies is not None else [],
    }


async def _local_inventory(request: web.Request) -> dict:
    from ainode.models.api_routes import _DOWNLOAD_JOB_MAX_AGE

    app = request.app
    config = app["config"]
    models_dir = app["model_manager"].models_dir
    scan = app["model_inventory_scan"]
    task = scan.get("task")
    loop = asyncio.get_running_loop()
    now = loop.time()
    if task is None or task.done():
        task = loop.run_in_executor(None, scan_model_copies, models_dir, config.hf_cache_dir)
        scan.update(task=task, started=now)
        # Retrieve failures even if all waiting requests timed out.
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
    remaining = SCAN_TIMEOUT - (now - scan["started"])
    if remaining <= 0:
        return _node_result(config, "unavailable")
    try:
        rows = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
    except (OSError, RuntimeError, ValueError, asyncio.TimeoutError):
        logger.debug("Could not check local model inventory", exc_info=True)
        return _node_result(config, "unavailable")

    # A concurrent request may share the same disk scan. The jobs and runtime
    # decoration belong to this response only.
    copies = [replace(row) for row in rows]
    _apply_download_jobs(copies, app["download_jobs"], _DOWNLOAD_JOB_MAX_AGE)
    await _apply_loaded_state(request, copies)
    return _node_result(config, "ok", [row.to_dict() for row in copies])


async def handle_node_inventory(request: web.Request) -> web.Response:
    """Versioned response of GET /api/models/downloaded?inventory=1."""
    # An open entry node may ask for public inventory. Its fleet credential
    # alone must not expose a protected peer's disk inventory to that caller.
    if request.query.get("anonymous") == "1" and getattr(
        request.app.get("auth_config"), "enabled", False
    ):
        raise web.HTTPForbidden()
    node = await _local_inventory(request)
    return web.json_response({"inventory_version": NODE_INVENTORY_VERSION, **node},
                             status=200 if node["status"] == "ok" else 503,
                             headers={"Cache-Control": "no-store"})


def _valid_row(row) -> bool:
    return (isinstance(row, dict) and isinstance(row.get("status"), str)
            and row["status"] in MODEL_STATUSES and "loaded" in row
            and (row["loaded"] is None or isinstance(row["loaded"], bool))
            and all(isinstance(row.get(field), str) for field in ROW_TEXT_FIELDS))


async def _peer_inventory(request: web.Request, node, hosts: list[str]) -> dict:
    if node.status in (NodeStatus.STALE, NodeStatus.OFFLINE):
        return _node_result(node, "stale")
    session = request.app.get("client_session")
    if session is None or type(node.web_port) is not int or not 1 <= node.web_port <= 65535:
        return _node_result(node, "unavailable")
    params = {"inventory": str(NODE_INVENTORY_VERSION)}
    if not is_authenticated(request):
        params["anonymous"] = "1"
    headers = fleet_headers(request.app)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + PEER_TIMEOUT
    for host in hosts:
        try:
            # A legacy unsigned heartbeat only establishes its UDP source.
            if not headers and host != node.peer_ip:
                continue
            address = ip_address(host)
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            endpoint = f"[{address}]" if address.version == 6 else str(address)
            url = f"http://{endpoint}:{node.web_port}/api/models/downloaded"
            async with session.get(url, params=params, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=remaining, connect=1),
                                   allow_redirects=False) as response:
                if response.status != 200:
                    continue
                payload = await response.json()
                if (isinstance(payload, dict)
                        and payload.get("inventory_version") == NODE_INVENTORY_VERSION
                        and payload.get("node_id") == node.node_id
                        and payload.get("status") == "ok"
                        and isinstance(payload.get("models"), list)
                        and all(_valid_row(row) for row in payload["models"])):
                    fields = (*ROW_TEXT_FIELDS, "loaded", "status")
                    return _node_result(node, "ok", [
                        {field: row[field] for field in fields} for row in payload["models"]
                    ])
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            continue
    return _node_result(node, "unavailable")


async def handle_fleet_inventory(request: web.Request) -> web.Response:
    """Read the local store and the inventories of known cluster members."""
    from ainode.api.server import _local_fabric_ip, member_hosts

    app = request.app
    config = app["config"]
    cluster = app.get("cluster_state")
    local_fabric = _local_fabric_ip(cluster, config.node_id)
    peers = [node for node in (cluster.members(config.cluster_id) if cluster else [])
             if node.node_id != config.node_id]
    peers.sort(key=lambda node: ((node.node_name or node.node_id).casefold(), node.node_id))
    nodes = await asyncio.gather(
        _local_inventory(request),
        *(_peer_inventory(request, node, member_hosts(node, local_fabric)) for node in peers),
    )
    return web.json_response({"nodes": nodes}, headers={"Cache-Control": "no-store"})
