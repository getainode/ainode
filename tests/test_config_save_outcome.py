"""The Settings toast tells the truth when PATCH /api/config rejected fields (PR 311 round 6)."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "ainode/web/static/js/app.js"
NODE = shutil.which("node")


def outcome(body, restart_hint=False):
    src = APP_JS.read_text()
    fn = src[src.index("// config-save-outcome:start"):src.index("// config-save-outcome:end")]
    script = fn + f"\nprocess.stdout.write(JSON.stringify(configSaveOutcome({json.dumps(body)}, {json.dumps(restart_hint)})));"
    return json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True).stdout)


pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")
REASON = "This setting changes the node's identity, network, cluster, storage or security"


def test_all_rejected_is_an_error_with_the_reason():
    got = outcome({"applied": {}, "rejected": ["node_name", "host"], "rejected_reasons": {"node_name": REASON}})
    assert got["type"] == "error" and "node_name, host" in got["message"] and REASON in got["message"]
    assert "Saved" not in got["message"]


def test_mixed_is_a_warning_naming_both_sides():
    got = outcome({"applied": {"max_model_len": 4096}, "rejected": ["master_address"],
                   "rejected_reasons": {"master_address": REASON}})
    assert got["type"] == "warning" and "max_model_len" in got["message"] and "master_address" in got["message"]


def test_clean_save_and_restart_hint():
    assert outcome({"applied": {"model": "x"}, "rejected": []}) == {"type": "success", "message": "Saved"}
    assert outcome({"applied": {"host": "0.0.0.0"}, "rejected": []}, True)["type"] == "info"


def test_patch_config_uses_the_outcome():
    src = APP_JS.read_text()
    body = src[src.index("  async _patchConfig(patch, opts) {"):]
    body = body[:body.index("\n  },")]
    assert "configSaveOutcome(body, opts.restartHint)" in body and "'Saved'" not in body
