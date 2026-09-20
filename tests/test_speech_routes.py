"""Speech to text through the fleet endpoint: routing on a multipart form field.

``/v1/audio/transcriptions`` is the first forwarded path whose body is not JSON.
OpenAI's audio API is a file upload, so the model id that decides which node
serves the request arrives as a form field beside the audio, and the body can only
be parsed against the boundary in its own ``Content-Type`` header. That makes two
things load-bearing and both are tested here: the router reads the ``model`` field
out of the buffered body, and it forwards those bytes UNCHANGED under the caller's
own Content-Type, because a re-encoded body would carry a boundary the forwarded
header no longer describes.

The rest is the behaviour every forwarded path already had, exercised through the
same handler: the model picks the node (including a stacked instance on its own
port), a dead candidate fails over to a live one, and a model nobody serves is a
404 rather than a 502. A request that never names a model is the one new refusal:
a 400 naming the field, because there is no JSON body to fall back on and routing
someone's audio to whatever this node happens to serve would answer a chat
engine's refusal as if the fleet had nothing.

Run against a REAL AINode app with a fake engine in its cluster state.
"""

import socket

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.multipart import boundary_of, form_fields, is_multipart
from ainode.api.server import create_app
from ainode.api.server_routes import serving_kind
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode
from ainode.models.registry import CURATED_CLUSTER_MODELS

SPEECH_MODEL = "openai/whisper-large-v3-turbo"
CHAT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"

#: A multipart body as curl writes one: `-F file=@sample.wav -F model=... -F
#: language=en`. Built by hand rather than with aiohttp's FormData so the test
#: knows the exact bytes it expects the engine to receive.
BOUNDARY = "------------------------aN0d3Sp33ch"
WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00" \
            b"\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00\x00\x00\xff\x7f\x01\x80"


def multipart_body(fields: dict, *, boundary: str = BOUNDARY,
                   filename: str = "sample.wav") -> bytes:
    """A multipart/form-data body: the wav as a file part, the rest as text parts."""
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        if isinstance(value, bytes):
            out += (f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n').encode()
            out += b"Content-Type: audio/wav\r\n\r\n"
            out += value
        else:
            out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            out += str(value).encode()
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out)


def content_type(boundary: str = BOUNDARY) -> str:
    return f"multipart/form-data; boundary={boundary}"


class FakeEngine:
    """A vLLM serving a Whisper model, recording the exact bytes it was handed."""

    def __init__(self):
        self.seen: list = []
        self.transcript = "The quick brown fox jumps over the lazy dog."

    def app(self):
        app = web.Application()
        app.router.add_post("/v1/audio/transcriptions", self.transcriptions)
        app.router.add_post("/v1/audio/translations", self.translations)
        return app

    async def _record(self, request):
        body = await request.read()
        self.seen.append({"path": request.path, "query": dict(request.query),
                          "headers": dict(request.headers), "body": body})
        return body

    async def transcriptions(self, request):
        await self._record(request)
        return web.json_response({"text": self.transcript})

    async def translations(self, request):
        await self._record(request)
        return web.json_response({"text": "translated into English"})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def engine_fake():
    return FakeEngine()


@pytest_asyncio.fixture
async def engine(engine_fake):
    server = TestServer(engine_fake.app())
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


@pytest_asyncio.fixture
async def client(engine):
    """An AINode app whose fleet serves a chat model and a stacked Whisper instance.

    That is the shape a stacked speech model has on this hardware: the chat engine
    on the node's own port, Whisper beside it on a port of its own, which is the
    routing case the model form field has to get right.
    """
    config = NodeConfig(node_id="local-node", node_name="LocalNode", model=None,
                        api_port=_free_port(), web_port=_free_port(),
                        cluster_enabled=False)
    app = create_app(config=config, engine=None)
    app["cluster_state"].add_node(ClusterNode(
        node_id="spark-4", node_name="Spark-4-GX10", gpu_name="NVIDIA GB10",
        gpu_memory_gb=121.7, unified_memory=True, model=CHAT_MODEL,
        status=NodeStatus.ONLINE, api_port=engine.port, web_port=engine.port,
        last_seen=0.0, fabric_ip="127.0.0.1",
        instances=[{"model": SPEECH_MODEL, "api_port": engine.port,
                    "status": "serving"}]))
    async with TestClient(TestServer(app)) as c:
        yield c


# --------------------------------------------------- the multipart field reader

def test_a_multipart_content_type_is_recognized_and_its_boundary_read():
    assert is_multipart(content_type()) is True
    assert is_multipart("application/json") is False
    assert boundary_of(content_type()) == BOUNDARY
    assert boundary_of('multipart/form-data; boundary="quoted-one"') == "quoted-one"
    assert boundary_of("multipart/form-data") == ""


def test_the_text_fields_are_read_and_the_file_part_is_skipped():
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL,
                           "language": "en", "response_format": "json"})
    fields = form_fields(body, content_type())
    assert fields == {"model": SPEECH_MODEL, "language": "en",
                      "response_format": "json"}


def test_a_body_the_reader_cannot_parse_yields_no_fields_rather_than_raising():
    """A caller's malformed body is answered on the MISSING field, not on a
    traceback: the proxy has one thing to say about it and a parse error is not
    something the caller can act on."""
    assert form_fields(b"not multipart at all", content_type()) == {}
    assert form_fields(multipart_body({"model": SPEECH_MODEL}),
                       "multipart/form-data") == {}
    assert form_fields(b"", content_type()) == {}


def test_an_unquoted_field_name_is_accepted_and_a_repeat_keeps_the_first():
    raw = (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=model\r\n\r\n"
           f"{SPEECH_MODEL}\r\n"
           f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n"
           f"someone/Else\r\n"
           f"--{BOUNDARY}--\r\n").encode()
    assert form_fields(raw, content_type()) == {"model": SPEECH_MODEL}


def test_an_oversized_field_value_is_treated_as_absent():
    """The reader exists for a model id and a language code. A part with no
    filename but a megabyte of content is not decoded into memory to route on."""
    from ainode.api.multipart import MAX_FIELD_BYTES
    body = multipart_body({"model": "x" * (MAX_FIELD_BYTES + 1)})
    assert form_fields(body, content_type()) == {}


# -------------------------------------------------------------------- routing

@pytest.mark.asyncio
async def test_transcription_is_routed_by_the_model_form_field(client, engine_fake):
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL, "language": "en"})
    resp = await client.post("/v1/audio/transcriptions", data=body,
                             headers={"Content-Type": content_type()})
    assert resp.status == 200
    assert (await resp.json())["text"] == engine_fake.transcript
    assert [s["path"] for s in engine_fake.seen] == ["/v1/audio/transcriptions"]


@pytest.mark.asyncio
async def test_translation_is_forwarded_too(client, engine_fake):
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL})
    resp = await client.post("/v1/audio/translations", data=body,
                             headers={"Content-Type": content_type()})
    assert resp.status == 200
    assert (await resp.json())["text"] == "translated into English"
    assert engine_fake.seen[0]["path"] == "/v1/audio/translations"


@pytest.mark.asyncio
async def test_the_body_reaches_the_engine_byte_for_byte(client, engine_fake):
    """Identical bytes under the caller's own boundary. A multipart body is only
    parseable against the boundary in its Content-Type, so re-encoding the parts
    (or forwarding a different header) hands the engine a body it cannot read."""
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL,
                           "language": "en", "temperature": "0"})
    resp = await client.post("/v1/audio/transcriptions", data=body,
                             headers={"Content-Type": content_type()})
    assert resp.status == 200
    seen = engine_fake.seen[0]
    assert seen["body"] == body
    headers = {k.lower(): v for k, v in seen["headers"].items()}
    assert headers["content-type"] == content_type()
    assert headers["content-length"] == str(len(body))


@pytest.mark.asyncio
async def test_an_unusual_boundary_is_forwarded_as_the_caller_wrote_it(client,
                                                                      engine_fake):
    odd = "----=_Part.9;weird"
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL}, boundary=odd)
    resp = await client.post(
        "/v1/audio/transcriptions", data=body,
        headers={"Content-Type": f'multipart/form-data; boundary="{odd}"'})
    assert resp.status == 200
    seen = engine_fake.seen[0]
    assert seen["body"] == body
    assert f'boundary="{odd}"' in seen["headers"]["Content-Type"]


@pytest.mark.asyncio
async def test_a_pinned_target_still_forwards_the_body_unchanged(client, engine_fake,
                                                                engine):
    """Pinning (#197) and a multipart body have to compose.

    The pin travels in headers, so it works on an audio path with no body field to
    strip, and the strip-and-re-serialize step that a JSON pin can take must not
    touch these bytes: re-encoding the parts would break the boundary the
    forwarded Content-Type names.
    """
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL})
    resp = await client.post("/v1/audio/transcriptions", data=body,
                             headers={"Content-Type": content_type(),
                                      "X-AINode-Node": "spark-4",
                                      "X-AINode-Port": str(engine.port)})
    assert resp.status == 200
    assert engine_fake.seen[0]["body"] == body


@pytest.mark.asyncio
async def test_a_pin_at_a_node_the_cluster_does_not_know_is_a_404(client, engine_fake):
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL})
    resp = await client.post("/v1/audio/transcriptions", data=body,
                             headers={"Content-Type": content_type(),
                                      "X-AINode-Node": "no-such-node"})
    assert resp.status == 404
    assert (await resp.json())["error"]["code"] == "unknown_node"
    assert engine_fake.seen == []


@pytest.mark.asyncio
async def test_the_query_string_and_headers_survive(client, engine_fake):
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL})
    resp = await client.post("/v1/audio/transcriptions?stream=false", data=body,
                             headers={"Content-Type": content_type(),
                                      "authorization": "Bearer placeholder"})
    assert resp.status == 200
    assert engine_fake.seen[0]["query"] == {"stream": "false"}
    headers = {k.lower(): v for k, v in engine_fake.seen[0]["headers"].items()}
    assert headers["authorization"] == "Bearer placeholder"


# ------------------------------------------------------------- the refusals

@pytest.mark.asyncio
async def test_a_missing_model_field_is_a_400_naming_the_field(client, engine_fake):
    body = multipart_body({"file": WAV_BYTES, "language": "en"})
    for path in ("/v1/audio/transcriptions", "/v1/audio/translations"):
        resp = await client.post(path, data=body,
                                 headers={"Content-Type": content_type()})
        assert resp.status == 400, path
        err = (await resp.json())["error"]
        assert err["param"] == "model"
        assert err["code"] == "missing_model_field"
        assert "model" in err["message"]
    assert engine_fake.seen == []


@pytest.mark.asyncio
async def test_an_empty_model_field_reads_as_missing(client, engine_fake):
    resp = await client.post("/v1/audio/transcriptions",
                             data=multipart_body({"file": WAV_BYTES, "model": "  "}),
                             headers={"Content-Type": content_type()})
    assert resp.status == 400
    assert (await resp.json())["error"]["code"] == "missing_model_field"
    assert engine_fake.seen == []


@pytest.mark.asyncio
async def test_a_model_nobody_serves_is_a_404_not_a_bad_gateway(client, engine_fake):
    body = multipart_body({"file": WAV_BYTES, "model": "who/Knows-ASR"})
    resp = await client.post("/v1/audio/transcriptions", data=body,
                             headers={"Content-Type": content_type()})
    assert resp.status == 404
    assert (await resp.json())["error"]["code"] == "model_not_found"
    assert engine_fake.seen == []


@pytest.mark.asyncio
async def test_a_dead_candidate_fails_over_to_a_live_one(client, engine_fake, engine):
    """The same failover the JSON paths get: a ghost claim from a node that died
    must not 502 a request another instance can serve."""
    cluster = client.app["cluster_state"]
    cluster.add_node(ClusterNode(
        node_id="ghost", node_name="Ghost", gpu_name="NVIDIA GB10",
        gpu_memory_gb=121.7, unified_memory=True, model=SPEECH_MODEL,
        status=NodeStatus.ONLINE, api_port=_free_port(), web_port=_free_port(),
        last_seen=0.0, fabric_ip="127.0.0.1"))
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL})
    resp = await client.post("/v1/audio/transcriptions", data=body,
                             headers={"Content-Type": content_type()})
    assert resp.status == 200
    assert (await resp.json())["text"] == engine_fake.transcript
    assert engine_fake.seen[0]["body"] == body


@pytest.mark.asyncio
async def test_an_audio_path_under_v1_is_not_a_catch_all(client, engine_fake):
    """Still a route table: only the two paths vLLM actually serves are forwarded."""
    body = multipart_body({"file": WAV_BYTES, "model": SPEECH_MODEL})
    for path in ("/v1/audio/speech", "/v1/audio", "/v1/realtime"):
        resp = await client.post(path, data=body,
                                 headers={"Content-Type": content_type()})
        assert resp.status == 404, path
    assert engine_fake.seen == []


# ------------------------------------------------------ the model on the fleet

@pytest.mark.asyncio
async def test_v1_models_lists_the_speech_model_like_any_other(client):
    resp = await client.get("/v1/models")
    assert resp.status == 200
    ids = [row["id"] for row in (await resp.json())["data"]]
    assert SPEECH_MODEL in ids
    assert CHAT_MODEL in ids


@pytest.mark.asyncio
async def test_the_capability_probe_reports_speech_and_probes_nothing(client):
    """Both probes are chat completions and a Whisper engine serves no chat path,
    so the answer comes off the catalog: speech true, vision and tools null."""
    resp = await client.get(f"/api/models/caps?model={SPEECH_MODEL}")
    assert resp.status == 200
    caps = await resp.json()
    assert caps["speech"] is True
    assert caps["vision"] is None and caps["tools"] is None
    assert caps["probed"] is False
    # Nothing was cached either: there was no probe to remember the answer to.
    assert not client.app.get("chat_caps_cache")


@pytest.mark.asyncio
async def test_a_chat_model_is_not_reported_as_speech(client):
    resp = await client.get(f"/api/models/caps?model={CHAT_MODEL}")
    assert resp.status == 200
    assert (await resp.json())["speech"] is False


@pytest.mark.asyncio
async def test_the_model_card_carries_the_speech_capability(client):
    resp = await client.get(f"/api/models/card?model={SPEECH_MODEL}")
    assert resp.status == 200
    card = await resp.json()
    assert card["catalog"]["capabilities"] == ["speech"]
    assert card["catalog"]["id"] == "whisper-large-v3-turbo"


def test_a_speech_instance_is_its_own_kind_not_a_chat_engine():
    """`type` is what keeps it out of the chat and bench pickers, the same way it
    keeps an embedding model out."""
    kind, caps = serving_kind(SPEECH_MODEL)
    assert kind == "speech"
    assert caps == ["transcriptions", "translations"]
    assert serving_kind(CHAT_MODEL)[0] == "llm"


# ---------------------------------------------------------- the catalog entry

def test_the_catalog_entry_is_shaped_for_a_stacked_speech_model():
    info = CURATED_CLUSTER_MODELS["whisper-large-v3-turbo"]
    assert info.hf_repo == SPEECH_MODEL
    assert info.capabilities == ["speech"]
    assert info.curated is True and info.proven_tp == 1
    assert info.size_gb == 1.6 and info.params_b == 0.81
    assert info.license == "MIT"
    # Small enough to leave running beside a chat model, which is the point of
    # having it in the catalog at all.
    assert 0 < info.recommended_gmu <= 0.10
    # 448 is Whisper's decoder window, and the engine reports it as max_model_len.
    assert info.context_length == 448
    # The audio extras are not in any fleet image, so the recipe pins the build
    # that has them, and pinning a non-default image turns off the backend's
    # automatic GB10 workaround, so the recipe states that too.
    assert info.engine_image == "ghcr.io/getainode/ainode-whisper:0.17.0-t5"
    assert "--enforce-eager" in info.extra_vllm_args
    # Stated, not inherited: a node's fp8 KV default is not a combination anyone
    # has proven on an encoder-decoder model here.
    assert info.kv_cache_dtype == "auto"


def test_the_entry_pins_an_image_a_fresh_node_can_pull():
    """The point of publishing it: an entry pinning a hand-built tag is a LAUNCH
    button that fails on every node but the one that built it. The tag is the base
    engine image's tag, and the Dockerfile that builds it states the same base, so
    the two cannot drift apart silently."""
    from pathlib import Path

    from ainode.engine.backends.nvidia import ENGINE_IMAGE_DOCKERFILES, image_repo

    info = CURATED_CLUSTER_MODELS["whisper-large-v3-turbo"]
    assert info.engine_image.startswith("ghcr.io/getainode/")
    repo, _, tag = info.engine_image.rpartition(":")
    root = Path(__file__).resolve().parents[1]
    dockerfile = root / ENGINE_IMAGE_DOCKERFILES[image_repo(info.engine_image)]
    assert dockerfile.exists()
    base = [line for line in dockerfile.read_text().splitlines()
            if line.startswith("ARG BASE=")]
    assert base, "the Dockerfile has to state its base as ARG BASE= for CI to read"
    assert base[0].rpartition(":")[2] == tag, (
        f"{repo} is tagged {tag} but the Dockerfile builds on {base[0]}")
    workflow = root / ".github" / "workflows" / "publish-whisper-image.yml"
    assert workflow.exists()
    assert repo in workflow.read_text()


def test_the_speech_entry_is_the_only_one_in_the_catalog():
    """One ASR entry on purpose. Another needs its own proof on this hardware,
    and vLLM's transcription support differs per architecture."""
    speech = [cid for cid, info in CURATED_CLUSTER_MODELS.items()
              if "speech" in (info.capabilities or [])]
    assert speech == ["whisper-large-v3-turbo"]
