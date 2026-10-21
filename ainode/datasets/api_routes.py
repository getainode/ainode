"""HTTP routes for dataset management, mounted under /api/datasets/."""

from __future__ import annotations

from aiohttp import web

from ainode.auth.accounts import may_use_server_paths
from ainode.datasets.manager import DatasetManager, DatasetSource


def setup_dataset_routes(app: web.Application, manager: DatasetManager) -> None:
    """Register dataset API routes on the aiohttp app."""
    app["dataset_manager"] = manager
    app.router.add_get("/api/datasets", handle_list)
    app.router.add_post("/api/datasets", handle_create)
    app.router.add_post("/api/datasets/upload", handle_upload)
    app.router.add_get("/api/datasets/{dataset_id}", handle_get)
    app.router.add_delete("/api/datasets/{dataset_id}", handle_delete)
    app.router.add_get("/api/datasets/{dataset_id}/preview", handle_preview)


async def handle_list(request: web.Request) -> web.Response:
    mgr: DatasetManager = request.app["dataset_manager"]
    return web.json_response({"datasets": mgr.list()})


async def handle_get(request: web.Request) -> web.Response:
    mgr: DatasetManager = request.app["dataset_manager"]
    ds = mgr.get(request.match_info["dataset_id"])
    if ds is None:
        return web.json_response({"error": "Dataset not found"}, status=404)
    return web.json_response(ds.to_dict(public=True))


async def handle_delete(request: web.Request) -> web.Response:
    mgr: DatasetManager = request.app["dataset_manager"]
    ok = mgr.delete(request.match_info["dataset_id"])
    if not ok:
        return web.json_response({"error": "Dataset not found"}, status=404)
    return web.json_response({"status": "deleted"})


async def handle_preview(request: web.Request) -> web.Response:
    mgr: DatasetManager = request.app["dataset_manager"]
    try:
        limit = int(request.query.get("limit", "3"))
    except ValueError:
        limit = 3
    limit = max(1, min(limit, 25))
    try:
        return web.json_response(mgr.preview(request.match_info["dataset_id"], limit=limit))
    except KeyError:
        return web.json_response({"error": "Dataset not found"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def handle_create(request: web.Request) -> web.Response:
    """JSON-body create for HuggingFace / local / URL sources."""
    mgr: DatasetManager = request.app["dataset_manager"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    source = (body.get("source") or "").strip().lower()
    name = body.get("name")
    description = body.get("description", "") or ""

    try:
        if source == DatasetSource.HUGGINGFACE.value:
            ds = mgr.add_huggingface(
                repo_id=body.get("repo_id") or body.get("path") or "",
                name=name,
                config=body.get("config"),
                split=body.get("split"),
                description=description,
            )
        elif source == DatasetSource.LOCAL.value:
            if not may_use_server_paths(request):
                return web.json_response(
                    {"error": "Local datasets require an admin or operator"}, status=403)
            ds = mgr.add_local(
                path=body.get("path") or "", name=name, description=description,
            )
        elif source == DatasetSource.URL.value:
            ds = await mgr.add_url_async(
                url=body.get("url") or body.get("path") or "",
                name=name,
                description=description,
                allow_internal=may_use_server_paths(request),
            )
        else:
            return web.json_response(
                {"error": f"Unsupported source '{source}'. Use 'huggingface', 'local', or 'url'."},
                status=400,
            )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:  # Server filesystem details are not part of the API.
        return web.json_response({"error": "Failed to add dataset"}, status=500)

    return web.json_response(ds.to_dict(public=True), status=201)


async def handle_upload(request: web.Request) -> web.Response:
    """Multipart upload for JSON/JSONL/CSV/Parquet files."""
    mgr: DatasetManager = request.app["dataset_manager"]
    reader = await request.multipart()

    filename: str = ""
    content: bytes = b""
    name: str = ""
    description: str = ""

    async for part in reader:
        if part.name == "file":
            filename = part.filename or "upload.dat"
            content = await part.read(decode=False)
        elif part.name == "name":
            name = (await part.text()).strip()
        elif part.name == "description":
            description = (await part.text()).strip()

    if not filename or not content:
        return web.json_response({"error": "Missing 'file' field"}, status=400)

    try:
        ds = mgr.add_upload(
            filename=filename, content=content, name=name or None,
            description=description,
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    return web.json_response(ds.to_dict(public=True), status=201)
