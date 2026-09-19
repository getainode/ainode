"""The embedding bench: request shapes, response shapes, metrics, record, table.

Every backend in `ainode/bench` is split `request()` / `parse()` so this file can pin
both halves from canned payloads with no server anywhere, and the metrics are plain
functions of plain values so the same is true of every number in a record. Nothing
here touches the network; the one client that would is replaced by a fake whose
answers the test writes.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ainode.bench.embed import (
    EmbedClient,
    EmbedError,
    build_embed_block,
    build_notes,
    build_record,
    cosine,
    latency_block,
    parse,
    percentile,
    quality_block,
    request_for,
    run_latency,
    run_quality,
    run_throughput,
    throughput_row,
)
from ainode.bench.embed.cli import build_parser, main, record_path, settings_for
from ainode.bench.embed.corpus import (
    BATCH_SIZES,
    LATENCY_TEXTS,
    PAIR_TEXTS,
    PAIRS,
    corpus_block,
    cycle_texts,
)

MODEL = "Qwen/Qwen3-Embedding-0.6B"
ENDPOINT = "http://100.72.9.84:8001/v1"


def _response(vectors, tokens=12, model=MODEL, shuffle=False):
    """A canned OpenAI embeddings response over the given vectors."""
    rows = [{"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)]
    if shuffle:
        rows = list(reversed(rows))
    return {"object": "list", "model": model, "data": rows,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


class _FakeClient:
    """An EmbedClient with the network removed: every answer is scripted.

    Vectors are derived from the text so the quality check has something real to
    order, and every call is recorded so the runner's batching can be asserted.
    """

    def __init__(self, dims=4, tokens_per_text=3, fail_on=(), wall_ms=10.0,
                 floor=7.5):
        self.endpoint = ENDPOINT
        self.model = MODEL
        self.reported_model = MODEL
        self.dims = dims
        self.tokens_per_text = tokens_per_text
        self.fail_on = set(fail_on)
        self.wall_ms = wall_ms
        self.floor = floor
        self.calls: list = []

    def _vector(self, text):
        # Deterministic and NOT unit length, so cosine() has to normalise.
        seed = sum(ord(c) for c in text)
        return [float((seed >> shift) % 7 + 1) * 2.0 for shift in range(self.dims)]

    def embed(self, texts):
        from ainode.bench.embed.client import Reply

        self.calls.append(list(texts))
        if len(self.calls) in self.fail_on:
            return Reply(wall_ms=self.wall_ms, error="HTTP 503: engine loading")
        return Reply(vectors=[self._vector(t) for t in texts], model=MODEL,
                     tokens=self.tokens_per_text * len(texts),
                     wall_ms=self.wall_ms)

    def ping(self):
        return self.floor

    def protocol(self):
        return {"path": "POST /v1/embeddings", "endpoint": self.endpoint}


# ------------------------------------------------------------ request shape --

def test_the_request_carries_the_model_and_the_texts_and_nothing_else():
    req = request_for(ENDPOINT, MODEL, ["a", "b"], api_key="sekret")
    assert req.url == "http://100.72.9.84:8001/v1/embeddings"
    assert req.payload == {"model": MODEL, "input": ["a", "b"]}
    assert req.headers == {"Authorization": "Bearer sekret"}


def test_a_shown_request_names_no_key_and_no_wall_of_text():
    req = request_for(ENDPOINT, MODEL, ["a"] * 64, api_key="sekret")
    shown = req.curl_safe()
    assert "sekret" not in shown
    assert "<64 text(s)>" in shown


def test_a_client_refuses_to_exist_without_an_endpoint_or_a_model():
    with pytest.raises(EmbedError):
        EmbedClient("", MODEL)
    with pytest.raises(EmbedError):
        EmbedClient(ENDPOINT, "")
    assert EmbedClient(ENDPOINT + "/", MODEL).endpoint == ENDPOINT


# ----------------------------------------------------------- response shape --

def test_a_reply_is_read_in_the_servers_own_index_order():
    """Nothing promises `data` is serialised in order, and a batch's vectors have to
    line up with the texts that produced them."""
    reply = parse(_response([[1.0, 0.0], [0.0, 1.0]], shuffle=True))
    assert reply.error is None
    assert reply.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert reply.dimensions == 2
    assert reply.tokens == 12
    assert reply.model == MODEL


def test_total_tokens_stands_in_when_prompt_tokens_is_missing():
    data = _response([[1.0]])
    data["usage"] = {"total_tokens": 9}
    assert parse(data).tokens == 9


def test_a_response_with_no_usage_reports_no_tokens_rather_than_zero():
    data = _response([[1.0]])
    del data["usage"]
    assert parse(data).tokens is None


@pytest.mark.parametrize("payload, why", [
    ("not an object", "a bare string"),
    ({"object": "list"}, "no data list"),
    ({"data": []}, "an empty data list"),
    ({"data": ["nope"]}, "an entry that is not an object"),
    ({"data": [{"index": 0}]}, "an entry with no embedding"),
    ({"data": [{"index": 0, "embedding": []}]}, "an empty embedding"),
])
def test_an_unreadable_response_is_an_error_and_never_an_empty_success(payload, why):
    reply = parse(payload)
    assert reply.error, why
    assert reply.vectors == []


# ------------------------------------------------------------------ metrics --

def test_a_percentile_is_interpolated_not_nearest_rank():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10], 0.95) == 10
    assert percentile([], 0.5) is None
    # The p95 of 0..99 sits between 94 and 95, not on either.
    assert percentile(list(range(100)), 0.95) == pytest.approx(94.05)


def test_the_latency_block_counts_failures_out_of_the_percentiles():
    block = latency_block([10.0, 20.0, 30.0], errors=2)
    assert (block["n"], block["answered"], block["errors"]) == (5, 3, 2)
    assert block["p50_ms"] == 20.0
    assert block["min_ms"] == 10.0 and block["max_ms"] == 30.0
    assert block["mean_ms"] == 20.0


def test_an_all_failed_latency_block_reports_null_and_not_zero():
    block = latency_block([], errors=4)
    assert block["answered"] == 0
    assert block["p50_ms"] is None and block["mean_ms"] is None


def test_a_throughput_row_divides_by_the_wall_time_of_the_sweep():
    row = throughput_row(16, requests=4, texts=64, seconds=0.5, tokens=800)
    assert row["texts_per_s"] == 128.0
    assert row["tokens_per_s"] == 1600.0


def test_a_throughput_row_with_no_reported_tokens_says_so():
    row = throughput_row(1, requests=1, texts=1, seconds=0.1, tokens=None)
    assert row["tokens"] is None and row["tokens_per_s"] is None
    assert row["texts_per_s"] == 10.0


def test_a_throughput_row_that_measured_nothing_is_null_and_not_zero():
    row = throughput_row(64, requests=1, texts=0, seconds=0.0, tokens=None, errors=1)
    assert row["texts_per_s"] is None and row["tokens_per_s"] is None
    assert row["errors"] == 1


def test_cosine_normalises_and_refuses_a_degenerate_vector():
    assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) is None
    assert cosine([1.0], [1.0, 2.0]) is None
    assert cosine([], []) is None


def test_the_quality_verdict_is_the_ordering_and_not_a_threshold():
    pairs = [{"id": "r", "related": True, "a": "x", "b": "y"},
             {"id": "u", "related": False, "a": "x", "b": "z"}]
    vectors = {"x": [1.0, 0.0], "y": [0.9, 0.1], "z": [0.0, 1.0]}
    block = quality_block(pairs, vectors)
    assert block["ordered"] is True
    assert block["margin"] > 0
    assert [p["id"] for p in block["pairs"]] == ["r", "u"]

    # The same pairs with the meanings swapped: related below unrelated is a FAIL,
    # and the margin goes negative so a reader sees by how much.
    swapped = quality_block(pairs, {"x": [1.0, 0.0], "y": [0.0, 1.0],
                                    "z": [0.9, 0.1]})
    assert swapped["ordered"] is False
    assert swapped["margin"] < 0


def test_a_pair_missing_a_vector_gives_no_verdict_rather_than_a_failure():
    pairs = [{"id": "r", "related": True, "a": "x", "b": "missing"},
             {"id": "u", "related": False, "a": "x", "b": "z"}]
    block = quality_block(pairs, {"x": [1.0, 0.0], "z": [0.0, 1.0]})
    assert block["ordered"] is None and block["margin"] is None


# ------------------------------------------------------------------- corpus --

def test_the_corpus_is_50_distinct_short_texts():
    assert len(LATENCY_TEXTS) == 50
    assert len(set(LATENCY_TEXTS)) == 50


def test_the_pairs_are_three_related_and_three_not():
    assert len(PAIRS) == 6
    assert sum(1 for p in PAIRS if p["related"]) == 3
    assert sum(1 for p in PAIRS if not p["related"]) == 3
    assert len({p["id"] for p in PAIRS}) == 6
    # Every side of every pair is in the one request the quality check sends.
    for pair in PAIRS:
        assert pair["a"] in PAIR_TEXTS and pair["b"] in PAIR_TEXTS


def test_a_batch_wraps_the_corpus_rather_than_repeating_one_text():
    """A batch of 64 copies of one string measures the prefix cache, not the engine."""
    texts = cycle_texts(64)
    assert len(texts) == 64
    assert len(set(texts)) == 50
    assert cycle_texts(0) == []


def test_the_corpus_block_states_what_a_reader_needs_to_compare_two_runs():
    block = corpus_block()
    assert block["id"] == "embed-50" and block["version"] >= 1
    assert block["latency_texts"] == 50
    assert block["pairs"] == 6 and block["related_pairs"] == 3


# -------------------------------------------------------------- the run ------

def test_the_latency_stage_sends_one_text_per_request_and_measures_the_floor():
    client = _FakeClient()
    block, dims, errors = run_latency(client, texts=LATENCY_TEXTS[:5])
    assert [len(call) for call in client.calls] == [1, 1, 1, 1, 1]
    assert block["n"] == 5 and block["answered"] == 5 and errors == []
    assert block["p50_ms"] == 10.0
    assert block["transport_floor_ms"] == 7.5
    assert dims == 4


def test_a_failed_request_is_an_error_row_and_out_of_the_percentiles():
    client = _FakeClient(fail_on={2})
    block, _dims, errors = run_latency(client, texts=LATENCY_TEXTS[:4])
    assert block["n"] == 4 and block["answered"] == 3 and block["errors"] == 1
    assert [e["index"] for e in errors] == [2]
    assert "503" in errors[0]["error"]


def test_the_batch_sweep_pushes_the_same_texts_through_every_size():
    client = _FakeClient()
    rows, errors = run_throughput(client, sizes=(1, 16, 64), per_size=64)
    assert errors == []
    assert [r["batch"] for r in rows] == [1, 16, 64]
    assert [r["requests"] for r in rows] == [64, 4, 1]
    assert [r["texts"] for r in rows] == [64, 64, 64]
    # Tokens come from the reported usage, three per text here.
    assert [r["tokens"] for r in rows] == [192, 192, 192]
    assert [len(call) for call in client.calls[:64]] == [1] * 64
    assert [len(call) for call in client.calls[64:68]] == [16] * 4
    assert len(client.calls[68]) == 64


def test_a_batch_size_that_does_not_divide_the_total_rounds_up_the_requests():
    client = _FakeClient()
    rows, _errors = run_throughput(client, sizes=(10,), per_size=25)
    assert rows[0]["requests"] == 3 and rows[0]["texts"] == 30


def test_the_quality_stage_embeds_every_side_in_one_request():
    client = _FakeClient(dims=8)
    block, errors = run_quality(client)
    assert errors == []
    assert len(client.calls) == 1
    assert client.calls[0] == list(PAIR_TEXTS)
    assert len(block["pairs"]) == 6
    assert block["ordered"] in (True, False)  # the fake's vectors are arbitrary


def test_a_failed_quality_request_gives_no_verdict_and_names_the_error():
    client = _FakeClient(fail_on={1})
    block, errors = run_quality(client)
    assert block["ordered"] is None
    assert errors and errors[0]["stage"] == "quality"


# ------------------------------------------------------------- the record ----

def _built(fail_on=()):
    client = _FakeClient(fail_on=fail_on)
    latency, dims, latency_errors = run_latency(client, texts=LATENCY_TEXTS[:6])
    throughput, throughput_errors = run_throughput(client, sizes=(1, 16),
                                                   per_size=16)
    quality, quality_errors = run_quality(client)
    errors = list(latency_errors) + list(throughput_errors) + list(quality_errors)
    block = build_embed_block(client, latency, throughput, quality, dims, errors, 4.2)
    notes = build_notes(client, block, 4.2)
    return client, block, build_record("a label", {"id": MODEL, "name": "Qwen3"},
                                       {"node": "Spark-4-GX10"}, block,
                                       {"endpoint": ENDPOINT}, notes,
                                       "20260919-220011")


def test_the_record_carries_an_embed_block_and_no_results_block():
    _client, _block, record = _built()
    assert record["schema"] == 1
    assert record["source"] == "scripts/ainode-bench.py embed"
    assert "embed" in record and "results" not in record
    assert set(record) == {"schema", "stamp", "label", "model", "placement",
                           "settings", "embed", "notes", "source"}


def test_the_embed_block_holds_every_section_the_schema_documents():
    _client, block, _rec = _built()
    assert set(block) == {"endpoint", "model_reported", "dimensions", "corpus",
                          "protocol", "latency", "throughput", "quality", "errors",
                          "seconds"}
    assert block["dimensions"] == 4
    assert block["model_reported"] == MODEL
    assert set(block["latency"]) >= {"n", "answered", "errors", "p50_ms", "p95_ms",
                                     "transport_floor_ms"}
    assert [r["batch"] for r in block["throughput"]] == [1, 16]


def test_the_record_is_json_and_carries_no_credential():
    _client, _block, record = _built()
    text = json.dumps(record)
    assert "Authorization" not in text and "Bearer" not in text
    assert json.loads(text) == record


def test_the_notes_explain_the_transport_floor_and_what_the_run_did_not_do():
    _client, _block, record = _built()
    notes = " ".join(record["notes"]).lower()
    assert "transport_floor_ms" in notes
    assert "nothing was loaded, unloaded or restarted" in notes
    assert "quality check" in notes
    assert "prefix cache" in notes
    assert "usage.prompt_tokens" in notes


@pytest.mark.parametrize("ordered, wanted", [
    (True, "not a retrieval benchmark"),
    (False, "suspect the pooling mode"),
    (None, "did not score every pair"),
])
def test_the_quality_note_says_which_of_the_three_verdicts_it_is(ordered, wanted):
    """Each verdict gets its own sentence: a pass says what the check does NOT prove,
    a fail names the three things to suspect, and no verdict says nothing failed."""
    client = _FakeClient()
    block = build_embed_block(
        client, latency_block([10.0]), [throughput_row(64, 1, 64, 0.5, 100)],
        {"ordered": ordered, "related_min": 0.8, "unrelated_max": 0.3,
         "margin": 0.5, "pairs": []}, 4, [], 1.0)
    assert wanted in " ".join(build_notes(client, block, 1.0))


def test_the_notes_count_failed_requests_separately():
    _client, _block, record = _built(fail_on={2})
    assert any("failed on transport" in note for note in record["notes"])


def test_the_record_filename_follows_the_slug_helper(tmp_path):
    path = record_path(tmp_path, "20260919-220011", MODEL,
                       "Spark-4 stacked beside Nemotron")
    assert path.name == ("20260919-220011-qwen3-embedding-0_6b-"
                         "spark-4-stacked-beside-nemotron-embed.json")


def test_the_settings_block_names_the_corpus_and_the_sweep():
    args = build_parser().parse_args(["--endpoint", ENDPOINT, "--model", MODEL,
                                     "--label", "x"])
    settings = settings_for(args, _FakeClient(), list(BATCH_SIZES))
    assert settings["corpus"] == "embed-50"
    assert settings["batches"] == [1, 16, 64]
    assert settings["latency_texts"] == 50
    assert "api_key" not in settings


# ----------------------------------------------------------------- the CLI ---

def test_a_dry_run_writes_nothing_and_requests_nothing(tmp_path, capsys):
    rc = main(["--endpoint", ENDPOINT, "--model", MODEL, "--dry-run"],
              out_dir=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "nothing was requested and no file was written" in out
    assert "64 request(s) of 1 text(s)" in out
    assert list(tmp_path.glob("*.json")) == []


def test_a_real_run_needs_a_label(tmp_path):
    with pytest.raises(SystemExit):
        main(["--endpoint", ENDPOINT, "--model", MODEL], out_dir=tmp_path)


def test_bad_batch_sizes_are_refused_before_anything_is_measured(tmp_path):
    for bad in ("0", "-4", "one"):
        with pytest.raises(SystemExit):
            main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "x",
                  "--batches", bad, "--dry-run"], out_dir=tmp_path)


def test_the_bench_cli_dispatches_the_embed_subcommand(tmp_path):
    """`scripts/ainode-bench.py embed ...` reaches this package's parser."""
    from ainode.bench.cli import main as bench_main

    rc = bench_main(["embed", "--endpoint", ENDPOINT, "--model", MODEL, "--dry-run"],
                    out_dir=tmp_path)
    assert rc == 0


# ------------------------------------------------- the committed real record --

REPO = pathlib.Path(__file__).resolve().parent.parent
RECORD = (REPO / "bench" / "results" /
          "20260919-220011-qwen3-embedding-0_6b-spark-4-stacked-beside-nemotron-embed.json")


def test_the_committed_record_is_the_shape_the_catalog_points_at():
    """The entry `qwen3-embedding-0.6b` is `verified=True` on the strength of this
    file, so its shape is part of the catalog's provenance and not just an example."""
    data = json.loads(RECORD.read_text())
    block = data["embed"]
    assert data["model"]["id"] == MODEL
    assert data["placement"]["node"] == "Spark-4-GX10"
    assert block["dimensions"] == 1024
    assert block["latency"]["answered"] == 50
    assert block["latency"]["transport_floor_ms"] is not None
    assert [r["batch"] for r in block["throughput"]] == [1, 16, 64]
    assert block["quality"]["ordered"] is True
    assert block["errors"] == []
    assert "results" not in data
