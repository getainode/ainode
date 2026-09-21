"""The speech bench: multipart body, response shapes, the normaliser, WER, the record.

Every backend in `ainode/bench` is split `request()` / `parse()` so this file can pin
both halves with no server anywhere, and the metrics are plain functions of plain values
so the same is true of every number in a record. Nothing here touches the network; the
one client that would is replaced by a fake whose transcripts the test writes.

The clips are committed audio, so this file also asserts on the real WAVs: the manifest
and the directory have to agree, or a run reports a rate over fewer clips than its
header claims.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ainode.bench.speech import (
    BOUNDARY,
    CLIPS,
    CLIPS_ID,
    ClipError,
    SpeechClient,
    SpeechError,
    accuracy_block,
    build_notes,
    build_record,
    build_speech_block,
    clip_row,
    clips_block,
    edit_counts,
    encode_multipart,
    latency_block,
    load_clips,
    normalize,
    parse,
    percentile,
    real_time_factor,
    request_for,
    rtf_block,
    run_clips,
    run_transport_floor,
)
from ainode.bench.speech.cli import build_parser, main, record_path, settings_for
from ainode.bench.speech.clips import clips_dir, generate_clips

MODEL = "openai/whisper-large-v3-turbo"
ENDPOINT = "http://100.122.26.9:3000/v1"
REPO = pathlib.Path(__file__).resolve().parent.parent


def _clip(clip_id="clip-01", text="The train arrives at platform 9", seconds=3.5,
          voice="Samantha", path=None):
    return {"id": clip_id, "voice": voice, "locale": "en_US", "text": text,
            "path": path or pathlib.Path(f"/nowhere/{clip_id}.wav"),
            "bytes": 1000, "seconds": seconds,
            "shape": {"channels": 1, "sample_rate": 16000, "sample_width_bytes": 2,
                      "frames": int(seconds * 16000), "seconds": seconds}}


class _FakeClient:
    """A SpeechClient with the network removed: every transcript is scripted."""

    def __init__(self, transcripts=None, fail_on=(), wall_ms=700.0, floor=31.5):
        self.endpoint = ENDPOINT
        self.model = MODEL
        self.reported_model = ""
        self.path_name = "transcriptions"
        self.transcripts = transcripts or {}
        self.fail_on = set(fail_on)
        self.wall_ms = wall_ms
        self.floor = floor
        self.calls: list = []

    def read_audio(self, clip):
        return b"RIFF----WAVEfake"

    def transcribe(self, clip, audio=None):
        from ainode.bench.speech.client import Reply

        self.calls.append((clip["id"], audio))
        if clip["id"] in self.fail_on:
            return Reply(wall_ms=self.wall_ms, error="HTTP 503: engine loading")
        return Reply(text=self.transcripts.get(clip["id"], clip["text"]),
                     wall_ms=self.wall_ms)

    def ping(self):
        return self.floor

    def protocol(self):
        return {"path": "POST /v1/audio/transcriptions", "endpoint": self.endpoint}


# ------------------------------------------------------------- request shape --

def test_the_body_is_multipart_with_the_model_as_a_form_field():
    req = request_for(ENDPOINT, MODEL, "/tmp/clip-01.wav", b"AUDIOBYTES",
                      api_key="sekret")
    assert req.url == "http://100.122.26.9:3000/v1/audio/transcriptions"
    assert req.content_type == f"multipart/form-data; boundary={BOUNDARY}"
    assert req.headers == {"Authorization": "Bearer sekret"}
    body = req.body.decode("latin-1")
    assert 'name="model"' in body
    assert MODEL in body
    assert 'name="file"; filename="clip-01.wav"' in body
    assert "AUDIOBYTES" in body


def test_the_file_part_comes_last_so_a_field_reader_has_to_step_over_it():
    """AINode's proxy reads the model field out of a buffered body and has to skip the
    file part to find it; a body whose fields all preceded the file would never test
    that."""
    body = encode_multipart({"model": MODEL, "response_format": "json"},
                            "clip-01.wav", b"AUDIO").decode("latin-1")
    assert body.index('name="model"') < body.index('name="file"')
    assert body.index('name="response_format"') < body.index('name="file"')


def test_the_multipart_body_uses_crlf_and_closes_its_boundary():
    body = encode_multipart({"model": MODEL}, "c.wav", b"A")
    assert body.startswith(f"--{BOUNDARY}\r\n".encode())
    assert body.endswith(f"--{BOUNDARY}--\r\n".encode())
    assert b"\n" not in body.replace(b"\r\n", b"")


def test_the_audio_bytes_survive_the_encoding_unchanged():
    """A WAV carries every byte value, newlines and boundary-looking runs included."""
    audio = bytes(range(256)) * 4
    body = encode_multipart({"model": MODEL}, "c.wav", audio)
    assert audio in body


def test_a_language_hint_is_only_sent_when_a_run_asks_for_it():
    plain = request_for(ENDPOINT, MODEL, "c.wav", b"A")
    assert "language" not in plain.fields
    hinted = request_for(ENDPOINT, MODEL, "c.wav", b"A", language="en")
    assert hinted.fields["language"] == "en"


def test_a_translation_run_posts_to_the_other_audio_path():
    req = request_for(ENDPOINT, MODEL, "c.wav", b"A", path_name="translations")
    assert req.url.endswith("/v1/audio/translations")


def test_a_shown_request_names_no_key_and_no_wall_of_bytes():
    req = request_for(ENDPOINT, MODEL, "/tmp/clip-01.wav", b"A" * 200_000,
                      api_key="sekret")
    shown = req.curl_safe()
    assert "sekret" not in shown
    assert "-F file=@clip-01.wav" in shown
    assert "AAAA" not in shown


def test_a_client_refuses_to_exist_without_an_endpoint_or_a_model():
    with pytest.raises(SpeechError):
        SpeechClient("", MODEL)
    with pytest.raises(SpeechError):
        SpeechClient(ENDPOINT, "")
    assert SpeechClient(ENDPOINT + "/", MODEL).endpoint == ENDPOINT


def test_translate_picks_the_translations_path_on_the_client():
    assert SpeechClient(ENDPOINT, MODEL).path_name == "transcriptions"
    assert SpeechClient(ENDPOINT, MODEL, translate=True).path_name == "translations"


# ------------------------------------------------------------ response shape --

def test_a_reply_is_the_text_field_stripped():
    reply = parse({"text": "  The train arrives.  "})
    assert reply.error is None
    assert reply.text == "The train arrives."


def test_a_body_with_no_text_field_is_an_error_not_an_empty_transcript():
    """An empty transcript scores as every word deleted, which would read as a model
    that heard silence rather than as a broken response."""
    assert parse({"object": "list"}).error is not None
    assert parse({"text": 7}).error is not None
    assert parse(["nope"]).error is not None


def test_a_plain_text_body_is_accepted_as_the_transcript():
    reply = parse("The train arrives.")
    assert reply.error is None
    assert reply.text == "The train arrives."
    assert parse("   ").error is not None


# -------------------------------------------------------------- normalisation --

def test_the_normaliser_folds_case_and_punctuation():
    assert normalize("The Train, arrives!") == ["the", "train", "arrives"]


def test_number_words_fold_to_digits_so_spelling_is_not_a_hearing_error():
    spoken = "The train arrives at platform nine at half past six"
    written = "The train arrives at platform 9 at half past 6."
    assert normalize(spoken) == normalize(written)


def test_a_composed_number_folds_to_the_digits_it_spells():
    assert normalize("four hundred and eighty-seven gigabytes") == ["487", "gigabytes"]
    assert normalize("two thousand twenty six") == ["2026"]
    assert normalize("one hundred and twenty eight") == ["128"]


def test_a_conjunction_outside_a_number_run_survives():
    """"salt and 3 eggs" keeps its "and"; only an "and" INSIDE a spelled number goes."""
    assert normalize("salt and 3 eggs") == ["salt", "and", "3", "eggs"]


def test_ordinals_fold_to_their_cardinals():
    assert normalize("September 19th") == ["september", "19"]
    assert normalize("the third of May") == ["the", "3", "of", "may"]
    assert normalize("the fortieth map") == ["the", "40", "map"]


def test_the_orthographic_normaliser_leaves_numbers_alone():
    assert normalize("platform nine", numbers=False) == ["platform", "nine"]
    assert normalize("platform 9", numbers=False) == ["platform", "9"]


def test_an_apostrophe_stays_inside_a_word():
    assert normalize("don't stop") == ["dont" if False else "don't", "stop"]


# ------------------------------------------------------------- word error rate --

def test_a_perfect_transcript_has_no_edits():
    counts = edit_counts(normalize("a b c"), normalize("a b c"))
    assert counts["edits"] == 0
    assert (counts["substitutions"], counts["deletions"], counts["insertions"]) == (
        0, 0, 0)


def test_the_three_edit_kinds_are_kept_apart():
    """A model that drops words and one that invents them both score badly, and only
    the breakdown tells them apart."""
    dropped = edit_counts(["a", "b", "c", "d"], ["a", "b"])
    assert (dropped["deletions"], dropped["insertions"], dropped["substitutions"]) == (
        2, 0, 0)
    invented = edit_counts(["a", "b"], ["a", "b", "c", "d"])
    assert (invented["insertions"], invented["deletions"]) == (2, 0)
    swapped = edit_counts(["a", "b", "c"], ["a", "x", "c"])
    assert (swapped["substitutions"], swapped["edits"]) == (1, 1)


def test_a_rate_is_pooled_over_words_not_averaged_over_clips():
    """A mean of per-clip rates weights a short clip like a long one; the pooled rate
    is the standard definition and the one the README column reports."""
    rows = [
        clip_row(_clip("a", "one two"), "one three", 100.0),
        clip_row(_clip("b", "three four five six seven eight seven six"),
                 "three four five six seven eight seven six", 100.0),
    ]
    block = accuracy_block(rows)
    assert block["reference_words"] == 10
    assert block["edits"] == 1
    assert block["wer"] == pytest.approx(0.1)
    # The per-clip mean is much worse, which is exactly why it is not the headline.
    assert block["wer_per_clip_mean"] == pytest.approx(0.25)
    assert block["clips_exact"] == 1


def test_both_rates_are_reported_and_the_gap_is_the_spelling():
    row = clip_row(_clip("a", "platform 9 at half past 6"),
                   "platform nine at half past six", 100.0)
    assert row["wer"] == 0.0
    assert row["wer_orthographic"] == pytest.approx(2 / 6)


def test_a_failed_clip_carries_its_error_and_no_numbers():
    """Never folded in as a 100 percent error rate: a transport failure inside a figure
    a reader takes as the model's is the one mistake this section can make."""
    row = clip_row(_clip("a", "one two three"), None, 900.0,
                   error="HTTP 503: engine loading")
    assert row["error"] == "HTTP 503: engine loading"
    for key in ("wer", "wer_orthographic", "rtf", "edits", "insertions"):
        assert row[key] is None
    assert row["transcript"] is None
    assert row["reference_words"] == 3


def test_a_failed_clip_is_counted_out_of_every_aggregate():
    rows = [clip_row(_clip("a", "one two"), "one two", 100.0),
            clip_row(_clip("b", "three four"), None, 900.0, error="boom")]
    accuracy = accuracy_block(rows)
    assert (accuracy["clips"], accuracy["scored"]) == (2, 1)
    assert accuracy["reference_words"] == 2
    assert accuracy["wer"] == 0.0
    latency = latency_block(rows)
    assert (latency["n"], latency["answered"], latency["errors"]) == (2, 1, 1)
    assert rtf_block(rows)["scored"] == 1


def test_a_run_where_nothing_answered_reports_no_rate_rather_than_a_perfect_one():
    rows = [clip_row(_clip("a", "one two"), None, 900.0, error="boom")]
    assert accuracy_block(rows)["wer"] is None
    assert latency_block(rows)["p50_ms"] is None
    assert rtf_block(rows)["pooled"] is None


# ------------------------------------------------------- latency and real time --

def test_a_percentile_is_interpolated_not_nearest_rank():
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([], 0.5) is None
    assert percentile([5], 0.95) == 5


def test_real_time_factor_is_wall_over_audio_and_absent_without_both():
    assert real_time_factor(1000.0, 4.0) == pytest.approx(0.25)
    assert real_time_factor(None, 4.0) is None
    assert real_time_factor(1000.0, None) is None
    assert real_time_factor(1000.0, 0) is None


def test_the_pooled_real_time_factor_is_total_wall_over_total_audio():
    rows = [clip_row(_clip("a", "one", seconds=2.0), "one", 1000.0),
            clip_row(_clip("b", "two", seconds=8.0), "two", 1000.0)]
    block = rtf_block(rows)
    assert block["audio_seconds"] == pytest.approx(10.0)
    assert block["wall_seconds"] == pytest.approx(2.0)
    assert block["pooled"] == pytest.approx(0.2)
    assert block["max"] == pytest.approx(0.5)


# ------------------------------------------------------------------ the runner --

def test_the_runner_walks_the_clips_in_manifest_order_one_at_a_time():
    clips = [_clip("clip-01", "one two"), _clip("clip-02", "three four")]
    client = _FakeClient()
    rows = run_clips(client, clips)
    assert [c[0] for c in client.calls] == ["clip-01", "clip-02"]
    assert [r["id"] for r in rows] == ["clip-01", "clip-02"]
    assert all(r["error"] is None for r in rows)


def test_the_transport_floor_is_the_median_of_the_probes():
    assert run_transport_floor(_FakeClient(floor=42.0), probes=3) == 42.0


def test_a_floor_nobody_measured_is_absent():
    class _Dead(_FakeClient):
        def ping(self):
            return None

    assert run_transport_floor(_Dead()) is None


def test_the_block_carries_the_clip_set_the_normaliser_and_the_rows():
    clips = [_clip("clip-01", "one two"), _clip("clip-02", "three four")]
    client = _FakeClient(transcripts={"clip-02": "three five"})
    rows = run_clips(client, clips)
    block = build_speech_block(client, clips, rows, 8.1, floor_ms=31.5)
    assert block["path"] == "POST /v1/audio/transcriptions"
    assert block["clips"]["id"] == CLIPS_ID
    assert block["clips"]["clips"] == 2
    assert block["normalizer"]["version"] >= 1
    assert block["latency"]["transport_floor_ms"] == 31.5
    assert block["accuracy"]["edits"] == 1
    assert [r["id"] for r in block["rows"]] == ["clip-01", "clip-02"]
    assert block["errors"] == []


def test_the_block_lists_every_failed_clip_by_id():
    clips = [_clip("clip-01"), _clip("clip-02")]
    client = _FakeClient(fail_on=("clip-02",))
    rows = run_clips(client, clips)
    block = build_speech_block(client, clips, rows, 8.1)
    assert [e["id"] for e in block["errors"]] == ["clip-02"]


def test_a_record_carries_a_speech_block_and_no_results_block():
    clips = [_clip("clip-01", "one two")]
    client = _FakeClient()
    rows = run_clips(client, clips)
    block = build_speech_block(client, clips, rows, 8.1, floor_ms=31.5)
    record = build_record("a label", {"id": MODEL, "name": "Whisper"},
                          {"node": "Spark-4-GX10"}, block, {"timeout_s": 120},
                          ["note"], "20260921-013000")
    assert record["schema"] == 1
    assert record["source"] == "scripts/ainode-bench.py speech"
    assert "speech" in record and "results" not in record
    assert record["placement"]["node"] == "Spark-4-GX10"


def test_the_notes_name_both_rates_the_floor_and_every_failure():
    clips = [_clip("clip-01", "platform 9"), _clip("clip-02", "three four")]
    client = _FakeClient(transcripts={"clip-01": "platform nine"},
                         fail_on=("clip-02",))
    rows = run_clips(client, clips)
    block = build_speech_block(client, clips, rows, 8.1, floor_ms=31.5)
    notes = " ".join(build_notes(client, block, 8.1))
    assert "nothing was loaded, unloaded or restarted" in notes
    assert "orthographic" in notes
    assert "31.5 ms" in notes
    assert "clip-02" in notes
    assert "never folded in as a" in notes


def test_the_notes_say_when_the_floor_could_not_be_measured():
    clips = [_clip("clip-01")]
    client = _FakeClient()
    rows = run_clips(client, clips)
    block = build_speech_block(client, clips, rows, 8.1, floor_ms=None)
    notes = " ".join(build_notes(client, block, 8.1))
    assert "transport floor could not be measured" in notes


# ------------------------------------------------------------- the committed clips --

def test_every_manifest_entry_has_a_committed_wav():
    """Strict loading: a manifest entry whose WAV is missing is a load error, not a
    clip quietly dropped out of a rate whose header still says ten."""
    clips = load_clips()
    assert len(clips) == len(CLIPS)
    assert [c["id"] for c in clips] == [entry[0] for entry in CLIPS]
    for clip in clips:
        assert clip["bytes"] > 1000
        assert clip["shape"]["sample_rate"] == 16000
        assert clip["shape"]["channels"] == 1
        assert clip["shape"]["sample_width_bytes"] == 2
        assert 1.0 < clip["seconds"] < 20.0


def test_the_clip_set_is_ten_clips_in_six_voices_across_six_locales():
    block = clips_block(load_clips())
    assert block["clips"] == 10
    assert len(block["voices"]) == 6
    assert len(block["locales"]) == 6
    assert block["sample_rates"] == [16000]
    assert block["directory"] == "bench/speech/clips"


def test_every_reference_line_is_between_eight_and_twenty_words_and_carries_a_digit():
    for _id, _voice, _locale, text in CLIPS:
        words = text.split()
        assert 8 <= len(words) <= 20, text
        assert any(ch.isdigit() for ch in text), text


def test_the_committed_audio_stays_under_two_megabytes():
    """Committed rather than synthesised per run, because a word error rate is only
    comparable over the same bytes; small enough that committing it is honest."""
    total = sum(clip["bytes"] for clip in load_clips())
    assert total < 2 * 1024 * 1024


def test_a_missing_clip_directory_is_a_load_error(tmp_path):
    with pytest.raises(ClipError):
        load_clips(tmp_path / "nope")


def test_a_clip_file_that_is_not_there_is_a_load_error(tmp_path):
    (tmp_path / "clip-01.wav").write_bytes(b"x" * 2000)
    with pytest.raises(ClipError):
        load_clips(tmp_path)


def test_an_empty_wav_is_a_load_error(tmp_path):
    for entry in CLIPS:
        (tmp_path / f"{entry[0]}.wav").write_bytes(b"RIFF")
    with pytest.raises(ClipError):
        load_clips(tmp_path)


def test_the_repo_set_is_what_a_run_finds_by_default():
    assert clips_dir().name == "clips"
    assert clips_dir().parent.name == "speech"


def test_an_env_override_moves_the_clip_directory(monkeypatch, tmp_path):
    from ainode.bench.speech.clips import ENV_CLIPS_DIR

    monkeypatch.setenv(ENV_CLIPS_DIR, str(tmp_path))
    assert clips_dir() == tmp_path
    assert clips_dir("/explicit") == pathlib.Path("/explicit")


def test_generating_clips_shells_out_to_say_then_afconvert(tmp_path):
    """Pinned argv, so the committed set can be rebuilt exactly and a test needs no
    synthesiser on the machine."""
    seen: list = []

    def fake(argv):
        seen.append(list(argv))
        if argv[0] == "afconvert":
            pathlib.Path(argv[-1]).write_bytes(b"RIFF" + b"\0" * 2000)

    written = generate_clips(tmp_path, manifest=CLIPS[:1], say=fake)
    assert seen[0][:3] == ["say", "-v", "Samantha"]
    assert seen[0][3] == "-o"
    assert seen[1][:6] == ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c"]
    assert seen[1][6] == "1"
    assert written == [tmp_path / "clip-01.wav"]


# ------------------------------------------------------------------ the CLI ---

def test_the_record_filename_follows_the_slug_helper(tmp_path):
    path = record_path(tmp_path, "20260921-013000", MODEL,
                       "Spark-4 stacked beside Nemotron")
    assert path.name == ("20260921-013000-whisper-large-v3-turbo-"
                         "spark-4-stacked-beside-nemotron-speech.json")


def test_the_settings_block_names_the_clip_set_and_the_path():
    args = build_parser().parse_args(["--endpoint", ENDPOINT, "--model", MODEL,
                                      "--label", "x"])
    settings = settings_for(args, _FakeClient(), load_clips())
    assert settings["clips"] == CLIPS_ID
    assert settings["clip_count"] == 10
    assert settings["path"] == "POST /v1/audio/transcriptions"
    assert "api_key" not in settings


def test_a_dry_run_writes_nothing_and_requests_nothing(tmp_path, capsys):
    rc = main(["--endpoint", ENDPOINT, "--model", MODEL, "--dry-run"],
              out_dir=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "nothing was requested and no file was written" in out
    assert "say-10 v1" in out
    assert list(tmp_path.glob("*.json")) == []


def test_a_real_run_needs_a_label(tmp_path):
    with pytest.raises(SystemExit):
        main(["--endpoint", ENDPOINT, "--model", MODEL], out_dir=tmp_path)


def test_a_bad_timeout_is_refused_before_anything_is_measured(tmp_path):
    with pytest.raises(SystemExit):
        main(["--endpoint", ENDPOINT, "--model", MODEL, "--label", "x",
              "--timeout", "0", "--dry-run"], out_dir=tmp_path)


def test_the_bench_cli_dispatches_the_speech_subcommand(tmp_path):
    """`scripts/ainode-bench.py speech ...` reaches this package's parser."""
    from ainode.bench.cli import main as bench_main

    rc = bench_main(["speech", "--endpoint", ENDPOINT, "--model", MODEL, "--dry-run"],
                    out_dir=tmp_path)
    assert rc == 0


# ------------------------------------------------------------- the speech table --

def test_the_speech_table_renders_a_record_and_keeps_it_out_of_the_speed_one():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "render_bench_table", REPO / "scripts" / "render-bench-table.py")
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)

    run = {"_file": "x-speech.json", "stamp": "20260921-013000",
           "label": "Spark-4 via the fleet endpoint",
           "model": {"id": MODEL, "name": "Whisper Large v3 Turbo"},
           "placement": {"node": "Spark-4-GX10", "tp": 1},
           "speech": {"endpoint": ENDPOINT, "path": "POST /v1/audio/transcriptions",
                      "accuracy": {"clips": 10, "scored": 10, "wer": 0.032,
                                   "wer_orthographic": 0.048, "clips_exact": 7},
                      "latency": {"p50_ms": 701.5},
                      "rtf": {"pooled": 0.1715}}}
    assert render.is_speech_run(run) is True
    assert render.throughput_runs([run]) == []
    row = render.row_for_speech(run, "http://base")
    assert row[0] == "Whisper Large v3 Turbo"
    assert row[2] == "10"
    assert row[3] == "3.2% (4.8% raw)"
    assert row[4] == "7 of 10"
    assert row[6] == "0.172"
    table = render.render_speech_table([run], "http://base")
    assert "Whisper Large v3 Turbo" in table
    assert table.startswith("| Model |")


def test_a_run_that_lost_a_clip_says_so_in_the_clips_column():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "render_bench_table", REPO / "scripts" / "render-bench-table.py")
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)

    speech = {"accuracy": {"clips": 10, "scored": 8, "wer": None,
                           "wer_orthographic": None, "clips_exact": None},
              "latency": {}, "rtf": {}}
    assert render.fmt_speech_clips(speech) == "8 of 10"
    assert render.fmt_wer(speech) == render.NOT_MEASURED
    assert render.fmt_exact(speech) == render.NOT_MEASURED
    assert render.fmt_speech_latency(speech) == render.NOT_MEASURED
    assert render.fmt_rtf(speech) == render.NOT_MEASURED


# --------------------------------------------- the committed real record ------

RECORD_DIR = REPO / "bench" / "results"


def _speech_records():
    return [p for p in sorted(RECORD_DIR.glob("*-speech.json"))]


def test_every_committed_speech_record_is_the_shape_the_schema_documents():
    """A catalog entry flipped to `verified=True` names one of these files, so their
    shape is part of the catalog's provenance and not just an example."""
    records = _speech_records()
    assert records, "no speech record is committed yet"
    for path in records:
        data = json.loads(path.read_text())
        block = data["speech"]
        assert data["schema"] == 1
        assert "results" not in data
        assert data["source"] == "scripts/ainode-bench.py speech"
        assert block["clips"]["id"] == CLIPS_ID
        assert block["clips"]["clips"] == len(CLIPS)
        assert block["normalizer"]["id"]
        assert block["accuracy"]["reference_words"] > 0
        assert block["accuracy"]["wer"] is not None
        assert block["latency"]["answered"] == block["latency"]["n"]
        assert block["rtf"]["pooled"] is not None
        assert [r["id"] for r in block["rows"]] == [e[0] for e in CLIPS]
        for row in block["rows"]:
            assert row["transcript"] is not None
