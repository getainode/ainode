"""A request with no ``model`` never defaults to a model this node does not serve.

``proxy_to_vllm`` used to route a JSON body with no ``model`` to ``config.model``.
On a routing-only master (Atlas, which serves nothing) that name is the dataclass
default or whatever config.json was copied from, so the caller got a 404 or a 502
about a model they never asked for. The fallback now needs a node that SERVES a
primary; without one the answer is a 400 naming the missing field, and nothing is
forwarded. A node that does serve one keeps today's behaviour exactly.
"""

from __future__ import annotations

import pytest

from ainode.core.config import NodeConfig
from tests.test_routing_caps import _Collector, _Session, _cluster, _node, _proxy

MODEL = "Qwen/Qwen3.8-27B"
CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def _master(config_model=None, engine=None, peers=None):
    """A node with the given config.model and primary engine, plus serving peers."""
    kwargs = {} if config_model is None else {"model": config_model}
    config = NodeConfig(node_id="atlas", node_name="Atlas", api_port=8000,
                        web_port=3000, **kwargs)
    nodes = [_node("atlas", model="")] + list(peers or [])
    session = _Session()
    return {
        "config": config,
        "engine": engine,
        "cluster_state": _cluster(nodes),
        "client_session": session,
        "metrics_collector": _Collector(),
    }, session


def _assert_missing_field(status, payload, session):
    assert status == 400
    err = payload["error"]
    assert err["code"] == "missing_model_field"
    assert err["param"] == "model"
    assert '"model"' in err["message"]
    assert session.tried == [], "a request with no model was forwarded somewhere"


@pytest.mark.parametrize("config_model", [
    None,                 # the NodeConfig default, a name nothing here serves
    "",                   # an explicitly cleared config
    MODEL,                # a config.json copied from a node that served it
])
def test_a_node_serving_nothing_answers_400_naming_the_field(config_model):
    peer = _node("spark1", model=MODEL, fabric="10.100.0.11")
    app, session = _master(config_model=config_model, peers=[peer])

    status, payload = _proxy(app, CHAT)

    _assert_missing_field(status, payload, session)
    assert app["metrics_collector"].calls[-1][1] is True, "not counted as an error"


@pytest.mark.parametrize("body", [
    dict(CHAT, model=""),
    dict(CHAT, model="   "),
    dict(CHAT, model=None),
])
def test_an_empty_model_is_as_missing_as_an_absent_one(body):
    app, session = _master()

    status, payload = _proxy(app, body)

    _assert_missing_field(status, payload, session)


def test_a_body_that_is_not_a_json_object_is_refused_the_same_way():
    app, session = _master()

    status, payload = _proxy(app, ["not", "an", "object"])

    _assert_missing_field(status, payload, session)


def test_a_named_model_still_routes_through_a_node_serving_nothing():
    peer = _node("spark1", model=MODEL, fabric="10.100.0.11")
    app, session = _master(peers=[peer])

    status, _ = _proxy(app, dict(CHAT, model=MODEL))

    assert status == 200
    assert session.tried == ["http://10.100.0.11:8000/v1/chat/completions"]


def test_a_node_that_serves_a_primary_keeps_its_default():
    """Today's behaviour, unchanged: no model means this node's own model."""
    local = _node("atlas", model=MODEL)
    app, session = _master(config_model=MODEL, engine=object())
    app["cluster_state"] = _cluster([local])

    status, _ = _proxy(app, CHAT)

    assert status == 200
    assert session.tried == ["http://localhost:8000/v1/chat/completions"]
    assert app["metrics_collector"].calls[-1] == (MODEL, False)


def test_an_empty_model_on_a_serving_node_also_takes_the_default():
    local = _node("atlas", model=MODEL)
    app, session = _master(config_model=MODEL, engine=object())
    app["cluster_state"] = _cluster([local])

    status, _ = _proxy(app, dict(CHAT, model=""))

    assert status == 200
    assert session.tried == ["http://localhost:8000/v1/chat/completions"]
