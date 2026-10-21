#!/usr/bin/env python3
"""Run bounded guard mutations in a disposable checkout copy, without Docker."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_launch_guards_r2.py", "tests/test_launch_path_guards.py"]
MUTATIONS = [
    ("open remote gains host authority", "ainode/auth/accounts.py",
     "return ipaddress.ip_address(request.remote).is_loopback", "return True"),
    ("fleet transport gains launch authority", "ainode/auth/accounts.py",
     'return request.headers.get("X-AINode-Launch-Authorized") == "true"', "return True"),
    ("dispatch skips original caller", "ainode/api/server.py",
     "if not may_supply_launch_fields(request):", "if False:"),
    ("launch parser skips privilege", "ainode/models/api_routes.py",
     "if not privileged and any(body.get(k)", "if False and any(body.get(k)"),
    ("raw arguments bypass privilege", "ainode/models/api_routes.py",
     '"extra_vllm_args", "extra_env")):', '"extra_env")):'),
    ("config skips host authority", "ainode/api/server.py",
     "        if key not in OPEN_PATCH_CONFIG_FIELDS and not privileged:\n", "        if False:\n"),
    ("storage accepts system roots", "ainode/core/storage_policy.py",
     "if str(path) in _SYSTEM or any(", "if False and any("),
    ("startup skips persisted policy", "ainode/core/config.py",
     "            sanitize_storage(config)", "            pass"),
    ("mount accepts sockets", "ainode/core/storage_policy.py",
     "if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):", "if False:"),
    ("mount skips root containment", "ainode/core/storage_policy.py",
     "if not any(path.is_relative_to(root) for root in approved):", "if False:"),
    ("backend skips mount recheck", "ainode/engine/backends/nvidia.py",
     "spec = request_volume(raw, config=self.config)", "spec = raw"),
    ("peer launch omits validation", "ainode/engine/backends/nvidia.py",
     'guard = self._peer_mount_guard(peer_models_dir) + " && "', 'guard = ""'),
    ("peer skips actual filesystem validation", "ainode/engine/backends/nvidia.py",
     '"    validate_mount_source(source, roots, home)\\n"', '"    pass\\n"'),
    ("cluster update skips host authority", "ainode/api/server.py",
     'if not may_control_host(request):\n        return web.json_response({"error": "Updates require an admin or operator"}, status=403)',
     "if False:\n        pass"),
    ("node update skips host authority", "ainode/api/server.py",
     'if not may_control_host(request):\n        return web.json_response({"error": "Node updates require an admin or operator"}, status=403)',
     "if False:\n        pass"),
    ("mount mode defaults writable", "ainode/engine/launch_validation.py",
     "parts[2] if len(parts) == 3 else 'ro'", "parts[2] if len(parts) == 3 else 'rw'"),
    ("storage loses secret-home protection", "ainode/core/storage_policy.py",
     "home.resolve().is_relative_to(path) for home in (", "False for home in ("),

    ("container mount skips namespace check", "ainode/engine/launch_validation.py",
     'if os.environ.get("AINODE_HOST_HOME") and (', 'if False and ('),

]


def run(root):
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *TESTS], cwd=root,
                            env=env, capture_output=True, text=True, timeout=30)
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="ainode-guard-mutations-") as directory:
        copy = Path(directory) / "repo"
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"))
        baseline = run(copy)
        if baseline.returncode:
            print("BASELINE FAILED")
            print(baseline.stdout + baseline.stderr)
            return 1
        print("BASELINE PASS: " + baseline.stdout.strip().splitlines()[-1])
        killed = 0
        for name, file, before, after in MUTATIONS:
            target = copy / file
            original = target.read_text()
            if before not in original:
                print("INVALID MUTATION: " + name)
                return 1
            target.write_text(original.replace(before, after, 1))
            try:
                result = run(copy)
            finally:
                target.write_text(original)
            # Collection or syntax errors are not evidence of a killed guard.
            valid_failure = result.returncode == 1 and " failed" in result.stdout and "ERROR collecting" not in result.stdout
            print(("KILLED: " if valid_failure else "SURVIVED OR INVALID: ") + name)
            if valid_failure:
                killed += 1
        print(f"SUMMARY: {killed}/{len(MUTATIONS)} mutations killed by focused tests")
        print("LIMITS: bounded source mutations and mocked launches; no live Docker, GPU or SSH fleet")
        return 0 if killed == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
