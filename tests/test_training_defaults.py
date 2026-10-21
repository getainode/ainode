"""Training defaults stay typed on disk and stay data in dashboard inputs."""

import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
import pytest_asyncio

from ainode.api.server import handle_get_config, handle_patch_config
from ainode.core.config import NodeConfig


APP_JS = Path(__file__).resolve().parents[1] / "ainode/web/static/js/app.js"
NODE = shutil.which("node")
PAYLOAD = '1"><img src=x onerror="globalThis.injected=true">'
FIELDS = (
    "training_default_epochs", "training_default_batch_size",
    "training_default_learning_rate",
)
BAD_VALUES = [PAYLOAD, '1" autofocus onfocus="alert(1)', "3", True, False,
              None, [PAYLOAD], {"value": PAYLOAD}, float("nan"), float("inf"),
              float("-inf"), 0, -1, 10**400]


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    return tmp_path


@pytest_asyncio.fixture(params=["open", "member"])
async def client(request, config_home):
    @web.middleware
    async def caller(req, handler):
        req = req.clone(remote="198.51.100.3")
        if request.param == "member":
            req.update(authenticated=True, user="example", user_role="member")
        return await handler(req)

    app = web.Application(middlewares=[caller])
    app["config"] = NodeConfig()
    app["auth_config"] = SimpleNamespace(enabled=request.param == "member")
    app.router.add_patch("/api/config", handle_patch_config)
    app.router.add_get("/api/config", handle_get_config)
    async with TestClient(TestServer(app)) as test_client:
        yield test_client


@pytest.mark.parametrize("key", FIELDS)
@pytest.mark.parametrize("value", BAD_VALUES)
@pytest.mark.asyncio
async def test_patch_rejects_unsafe_training_defaults(client, config_home, key, value):
    response = await client.patch("/api/config", json={key: value})
    assert response.status == 200  # PATCH reports per-field rejection.
    result = await response.json()
    assert result["applied"] == {}
    assert result["rejected"] == [key]
    assert key in result["rejected_reasons"]
    expected = getattr(NodeConfig(), key)
    assert getattr(client.app["config"], key) == expected
    assert json.loads((config_home / "config.json").read_text())[key] == expected


@pytest.mark.parametrize("key,value", [
    ("training_default_epochs", 1.5), ("training_default_epochs", 101),
    ("training_default_batch_size", 1.5), ("training_default_batch_size", 129),
    ("training_default_learning_rate", 1.01),
    ("training_default_method", PAYLOAD), ("training_default_method", []),
    ("training_default_method", None), ("training_default_method", True),
])
@pytest.mark.asyncio
async def test_patch_rejects_invalid_ranges_and_methods(client, key, value):
    result = await (await client.patch("/api/config", json={key: value})).json()
    assert result["rejected"] == [key]
    assert key not in result["applied"]
    assert key in result["rejected_reasons"]


@pytest.mark.parametrize("method", ["lora", "qlora", "full"])
@pytest.mark.parametrize("epochs,batch,rate", [(1, 1, 1e-8), (100, 128, 1), (3, 4, 2e-4)])
@pytest.mark.asyncio
async def test_valid_training_defaults_round_trip(client, method, epochs, batch, rate):
    patch = dict(zip(FIELDS, (epochs, batch, rate)))
    patch["training_default_method"] = method
    result = await (await client.patch("/api/config", json=patch)).json()
    assert result["applied"] == patch
    assert result["rejected"] == []
    loaded = NodeConfig.load()
    fetched = await (await client.get("/api/config")).json()
    for key, value in patch.items():
        assert getattr(loaded, key) == fetched[key] == value


@pytest.mark.asyncio
async def test_mixed_patch_applies_only_valid_fields(client):
    result = await (await client.patch("/api/config", json={
        "training_default_epochs": PAYLOAD, "training_default_batch_size": 8,
    })).json()
    assert result["applied"] == {"training_default_batch_size": 8}
    assert result["rejected"] == ["training_default_epochs"]


@pytest.mark.parametrize("key", FIELDS + ("training_default_method",))
@pytest.mark.parametrize("value", BAD_VALUES + [1.5, 129])
def test_load_sanitizes_invalid_persisted_defaults(config_home, key, value):
    # These values are invalid for all four fields, including an excessive
    # learning rate. Each invalid field falls back without losing its neighbor.
    config_file = config_home / "config.json"
    config_file.write_text(json.dumps({key: value, "node_name": "preserved"}))
    loaded = NodeConfig.load()
    assert getattr(loaded, key) == getattr(NodeConfig(), key)
    assert loaded.node_name == "preserved"


RENDER_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const nodes = {};
function node() {
  return { style: {}, addEventListener() {}, remove() {},
    appendChild() {}, value: '',
    set textContent(v) { this._text = String(v); },
    get innerHTML() {
      return this._text === undefined ? (this._html || '') :
        this._text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    },
    set innerHTML(v) { this._html = v; },
  };
}
const document = {
  addEventListener() {}, querySelector() { return null; },
  querySelectorAll() { return []; }, createElement: node, body: node(),
  getElementById(id) { return nodes[id] || (nodes[id] = node()); },
};
const ctx = vm.createContext({document, console});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);
const app = vm.runInContext('AINode', ctx);
const config = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
app.fetchJSON = async function(url) { return url === '/api/config' ? config : {}; };
app._updateEstimate = function() {};
(async function() {
  await app.showNewRunWizard();
  app.state.wizardState.step = 3;
  app.renderWizardStep();
  const html = nodes['wizard-panel'].innerHTML;
  assert.ok(!html.includes('<img'), 'hostile default became an element');
  assert.ok(!html.includes('onerror='), 'hostile default reached HTML');
  [['wz-epochs', 'training_default_epochs'], ['wz-batch', 'training_default_batch_size'],
   ['wz-lr', 'training_default_learning_rate']].forEach(([id, key]) => {
    assert.strictEqual(nodes[id].value, config[key], 'input was not populated safely');
  });
  console.log(JSON.stringify({html, field: app._field('Epochs', 'training_default_epochs',
    config.training_default_epochs)}));
})().catch(e => { console.error(e); process.exit(1); });
"""


class Inputs(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.inputs = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        assert tag != "img"
        assert not any(key.startswith("on") for key, _ in attrs)
        if tag == "input":
            self.inputs.append(dict(attrs))


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_persisted_hostile_defaults_are_data_in_wizard_and_settings(tmp_path):
    # Feed old, unsanitized disk values straight into the UI as a separate
    # defense, independent of validation on PATCH and startup.
    config_file = tmp_path / "hostile-config.json"
    config_file.write_text(json.dumps(dict.fromkeys(FIELDS, PAYLOAD)))
    proc = subprocess.run([NODE, "-e", RENDER_HARNESS, str(APP_JS), str(config_file)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert len(Inputs(result["html"]).inputs) == 8
    field = Inputs(result["field"]).inputs
    assert len(field) == 1
    assert field[0]["value"] == PAYLOAD
