"""Recover a previously serving engine once after positive evidence of death."""

import asyncio
import logging

from ainode.engine.maintenance import held

logger = logging.getLogger(__name__)


async def recover_engines(app, *, allow_recovery=True):
    from ainode.models.api_routes import (
        LaunchBusy, _GPU_RELEASE_SECONDS, _engine_exited, _log_failed_engine_output,
        _wait_for_bind, launch_slot,
    )
    from ainode.api.server import _engine_serving

    manager = app.get("instances")
    if manager is None:
        return
    loop = asyncio.get_running_loop()
    recovery = app["engine_recovery"]
    for inst in manager.instances():
        record, backend = inst.record, inst.backend
        key = record.instance_id
        if await _engine_serving(backend, loop):
            record.status = "serving"
            recovery.pop(key, None)
            continue
        # Never restart a loading engine because its API is temporarily quiet.
        if record.status not in ("serving", "failed"):
            continue
        if not await loop.run_in_executor(None, _engine_exited, backend):
            continue
        record.status = "failed"
        if not allow_recovery or held() or recovery.get(key) is backend:
            continue
        try:
            async with launch_slot(f"recover {record.model}", wait=0):
                # An unload or hold may have arrived while the probe awaited.
                if manager.get(key) is not inst or held():
                    continue
                recovery[key] = backend
                await _log_failed_engine_output(backend, record.model, "engine exited")
                await asyncio.sleep(_GPU_RELEASE_SECONDS)
                if manager.get(key) is not inst or held():
                    recovery.pop(key, None)
                    continue
                logger.warning("Engine %s exited; attempting one recovery", record.model)
                record.status = "starting"
                try:
                    started = await loop.run_in_executor(None, backend.start)
                    bound = False
                    if started:
                        bound, _, _ = await _wait_for_bind(
                            app, record.api_port, backend)
                    if manager.get(key) is inst:
                        record.status = "serving" if bound else "failed"
                except Exception:
                    record.status = "failed"
                    logger.exception("Engine recovery failed for %s", record.model)
        except LaunchBusy:
            continue


async def recovery_loop(app):
    while True:
        await asyncio.sleep(5)
        replay = app.get("_instance_replay_task")
        try:
            await recover_engines(app, allow_recovery=(replay is None or replay.done()))
        except Exception:
            logger.exception("Engine recovery probe failed")
