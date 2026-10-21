"""Read sanitized observations from stdin and evaluate the boot adoption gates.

Run the reviewed checker image with --network none --read-only, without any mounts,
devices or Docker socket, using: python -B -m ainode.engine.adoption_check.
This predicts protection by adoption; it never runs a sweep or changes policy.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

from ainode.core.config import NodeConfig
from ainode.engine import reconcile as rec


# Only launch shape is input. Credentials, paths and operator endpoints are not.
CONFIG_FIELDS = frozenset({
    "model", "api_port", "engine_backend", "engine_image", "distributed_mode",
    "distributed_executor", "peer_ips", "extra_vllm_args", "served_model_name",
})
CONTAINER_FIELDS = frozenset({"name", "id", "image", "argv", "running", "health", "models"})


def check_observations(data: dict) -> dict:
    """Use production adoption against supplied observations, with no I/O."""
    from ainode.models import api_routes as mr

    if not isinstance(data, dict) or set(data) - {
            "config", "manifest", "containers", "distributed_record", "start_clean"}:
        raise ValueError("invalid observations")
    config_data = data["config"]
    if not isinstance(config_data, dict) or set(config_data) - CONFIG_FIELDS:
        raise ValueError("invalid config shape")
    config = NodeConfig(**config_data)
    if type(data.get("start_clean", False)) is not bool:
        raise ValueError("invalid start-clean observation")
    config._skip_replay = data.get("start_clean", False)
    manifest = data["manifest"]
    if not isinstance(manifest, list) or not all(isinstance(e, dict) for e in manifest):
        raise ValueError("invalid manifest")
    if not isinstance(data["containers"], list):
        raise ValueError("invalid containers")
    containers = {}
    health = {}
    models = {}
    ids = set()
    for item in data["containers"]:
        if not isinstance(item, dict) or set(item) != CONTAINER_FIELDS:
            raise ValueError("incomplete container observation")
        name, cid = item["name"], item["id"]
        if (not isinstance(name, str) or not name or name in containers
                or not isinstance(cid, str) or len(cid) != 64
                or any(c not in "0123456789abcdef" for c in cid) or cid in ids
                or not isinstance(item["image"], str) or not item["image"]
                or type(item["running"]) is not bool or type(item["health"]) is not bool
                or not isinstance(item["argv"], list)
                or not all(isinstance(a, str) for a in item["argv"])
                or not isinstance(item["models"], list)
                or not all(isinstance(m, str) for m in item["models"])):
            raise ValueError("invalid container shape")
        ids.add(cid)
        info = {"Id": cid, "Name": "/" + name,
                "Config": {"Image": item["image"], "Cmd": item["argv"]},
                "State": {"Running": item["running"]}}
        containers[name] = info
        port = rec.container_shape(info)["api_port"] or config.api_port
        if item["running"]:
            if port in health:
                raise ValueError("ambiguous port observation")
            health[port], models[port] = item["health"], item["models"]

    rec.reset_state_for_tests()
    try:
        with (patch.object(rec, "inspect_container", containers.get),
              patch.object(rec, "list_engine_containers", lambda: list(containers)),
              patch.object(rec, "port_health", lambda port: health.get(port, False)),
              patch.object(rec, "served_models", lambda port: models.get(port, [])),
              patch.object(rec, "load_distributed_record",
                           lambda: data.get("distributed_record")),
              patch.object(mr, "load_instance_manifest", lambda: manifest)):
            rec.adopt_boot_engines(config)
            # Use the sweep's actual ID comparison, without invoking Docker.
            running_owned = [info["Id"] for name, info in containers.items()
                             if info["State"]["Running"] and (
                                 rec.PRIMARY_ENGINE_NAME in name
                                 or rec.HEAD_ENGINE_PREFIX in name)]
            unprotected = mr._without_adopted(running_owned)
            return {"protected_count": len(running_owned) - len(unprotected),
                    "all_running_owned_survive": not unprotected}
    finally:
        rec.reset_state_for_tests()


def main() -> int:
    try:
        result = check_observations(json.load(sys.stdin))
    except Exception:
        # Never echo supplied observations, paths or exception details.
        print(json.dumps({"protected_count": 0, "all_running_owned_survive": False}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_running_owned_survive"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
