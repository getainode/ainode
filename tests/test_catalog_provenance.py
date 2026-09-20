"""Provenance for `verified=True`, and the measured load times beside it.

`verified=True` on a catalog entry used to be a claim with nothing behind it: no
date, no node, no record. Some entries carried it from eras before the bench
existed while the ones proved this week had records nobody could find from the
entry. So the flag comes with `verified_on` (ISO date) and `verified_record` (a
filename under `bench/results/`), and both are required with it.

The guard used to be circular (issue #201): it only asserted on entries that
already named a record, so an entry claiming verification while naming nothing
passed, and six did. It reads the other way now. Every `verified=True` entry in
the catalog MUST name a record, that record MUST exist on disk, and it MUST carry
the date, asserted over the whole catalog rather than over the subset that already
complies. The reverse direction still holds too: a curated entry the bench has a
record for has to be marked verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ainode.models.registry import CURATED_CLUSTER_MODELS, FALLBACK_CATALOG, ModelInfo

RESULTS = Path(__file__).resolve().parents[1] / "bench" / "results"


def _all_entries():
    out = dict(FALLBACK_CATALOG)
    out.update(CURATED_CLUSTER_MODELS)
    return out


def _record_repos() -> dict:
    """{hf repo id: [record filenames]} over every bench record on disk."""
    by_repo: dict = {}
    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:  # pragma: no cover - a corrupt record is its own bug
            continue
        repo = ((data.get("model") or {}).get("id") or "").strip()
        if repo:
            by_repo.setdefault(repo, []).append(path.name)
    return by_repo


# ------------------------------------------------------------- the two fields --

def test_model_info_carries_verification_provenance():
    info = ModelInfo(id="x", name="X", hf_repo="o/x", size_gb=1.0, description="")
    assert info.verified is False
    assert info.verified_on == ""
    assert info.verified_record == ""
    assert info.typical_ready_minutes is None


@pytest.mark.parametrize("model_id", sorted(_all_entries()))
def test_every_verified_record_names_a_file_that_exists(model_id):
    info = _all_entries()[model_id]
    if not info.verified_record:
        return
    assert (RESULTS / info.verified_record).exists(), (
        f"{model_id} points at bench/results/{info.verified_record}, which is not there")
    assert info.verified, f"{model_id} names a bench record but is not marked verified"
    assert info.verified_on, f"{model_id} names a bench record but has no verified_on"


@pytest.mark.parametrize("model_id", sorted(_all_entries()))
def test_a_verified_entry_names_a_record_that_exists(model_id):
    """The direction the old guard could not check (#201). "Verified" is how a user
    decides whether a recipe will come up on their hardware, so the flag without a
    record on disk is a claim nobody can check and this fails on it."""
    info = _all_entries()[model_id]
    if not info.verified:
        return
    assert info.verified_record, (
        f"{model_id} is marked verified and names no bench record. Attach the record "
        f"that proves it, or set verified=False")
    assert (RESULTS / info.verified_record).exists(), (
        f"{model_id} is marked verified by bench/results/{info.verified_record}, "
        f"which is not there")
    assert info.verified_on, (
        f"{model_id} is marked verified with no verified_on date")


@pytest.mark.parametrize("model_id", sorted(_all_entries()))
def test_a_curated_model_the_bench_has_a_record_for_is_marked_verified(model_id):
    """The other direction: a record proves the model served on this hardware, so
    the entry must not still read as unproven."""
    info = _all_entries()[model_id]
    if not info.curated or info.hf_repo not in _record_repos():
        return
    assert info.verified, (
        f"{model_id} has bench records ({_record_repos()[info.hf_repo]}) "
        f"but verified is False")


def test_verified_on_is_an_iso_date_or_empty():
    import re
    for model_id, info in _all_entries().items():
        if not info.verified_on:
            continue
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", info.verified_on), \
            f"{model_id}: {info.verified_on!r} is not an ISO date"


def test_no_entry_is_verified_without_a_record():
    """The whole catalog in one assertion, because the shape "verified, no record"
    is not a shape any more (#201). Six entries used to be in it, wearing the badge
    a user reads before trusting a recipe."""
    unbacked = sorted(model_id for model_id, info in _all_entries().items()
                      if info.verified and not info.verified_record)
    assert unbacked == [], (
        f"marked verified with no bench record: {unbacked}. Attach the record or "
        f"set verified=False")


def test_the_six_entries_with_no_record_are_unverified():
    """The ones #201 found. Named here so a later edit cannot quietly flip one back
    without a record: the test above would catch it, this says which entries the
    decision was about."""
    for model_id in ("qwen3.5-9b-awq", "qwen3.5-4b-awq", "qwen3.5-35b-a3b-awq",
                     "llama-3.1-8b-nvfp4", "nemotron-cascade-2-30b-a3b-nvfp4",
                     "llama-3.3-70b-nvfp4"):
        info = _all_entries()[model_id]
        assert not info.verified, f"{model_id} is verified again with no record"
        assert info.verified_record == "" and info.verified_on == ""


def test_a_description_quoting_throughput_names_the_record_behind_it():
    """Measured numbers live in bench/results/ and nowhere else. A catalog
    description is read as a promise about the reader's own hardware, and six of
    them carried tok/s figures no record contains, so a figure in a description now
    requires the entry to name its record."""
    import re

    pattern = re.compile(r"tok/s|tokens/s|texts/s|\bt/s\b")
    offenders = sorted(model_id for model_id, info in _all_entries().items()
                       if pattern.search(info.description or "")
                       and not info.verified_record)
    assert offenders == [], (
        f"throughput in the description with no bench record named: {offenders}")


def test_the_entries_proved_this_week_all_name_their_record():
    """The five launches that were measured, plus the frontier MoE from June."""
    want = {
        "ornith-1.5-35b-a3b-nvfp4": "2026-09-13",
        "nemotron-3.5-lightning-nvfp4": "2026-09-13",
        "qwen3.8-27b-nvfp4": "2026-08-15",
        "deepseek-v4-flash-dspark": "2026-09-16",
        "qwen3.8-flash-next-nvfp4": "2026-09-16",
        "qwen3-235b-a22b-nvfp4": "2026-06-17",
    }
    for model_id, on in want.items():
        info = CURATED_CLUSTER_MODELS[model_id]
        assert info.verified_on == on, model_id
        assert (RESULTS / info.verified_record).exists(), model_id


# ------------------------------------------------------ the load-time seeds --

def test_the_measured_launches_seed_a_typical_load_time():
    """The numbers from this week's launches, so the interface has something to
    say on a node that has never run the model itself."""
    want = {
        "qwen3.8-27b-nvfp4": 12.0,          # solo on a GB10
        "ornith-1.5-35b-a3b-nvfp4": 12.0,   # stacked on a GB10
        "nemotron-3.5-lightning-nvfp4": 10.0,  # on the GX10
        "deepseek-v4-flash-dspark": 7.0,    # TP=2 on two GB10s, custom image
        "qwen3.8-flash-next-nvfp4": 11.0,   # TP=2 on two GB10s, autotune off
    }
    for model_id, minutes in want.items():
        assert CURATED_CLUSTER_MODELS[model_id].typical_ready_minutes == minutes, model_id


def test_a_model_nobody_has_launched_states_no_load_time():
    """Never filled in from the weight size: the seeds are timed launches or
    nothing, same rule the bench records live by."""
    for model_id in ("qwen3.5-397b-a17b-nvfp4", "llama-3.1-405b-nvfp4", "glm-5.1"):
        assert CURATED_CLUSTER_MODELS[model_id].typical_ready_minutes is None, model_id


def test_the_catalog_cache_round_trips_the_new_fields():
    """The on-disk catalog cache is `ModelInfo(**dict)`, and a cache written before
    these fields existed still has to load."""
    info = CURATED_CLUSTER_MODELS["qwen3.8-27b-nvfp4"]
    same = ModelInfo(**info.to_dict())
    assert (same.verified_on, same.verified_record, same.typical_ready_minutes) == \
        (info.verified_on, info.verified_record, info.typical_ready_minutes)

    old = {k: v for k, v in info.to_dict().items()
           if k not in ("verified_on", "verified_record", "typical_ready_minutes")}
    stale = ModelInfo(**old)
    assert stale.verified_on == "" and stale.typical_ready_minutes is None


# ------------------------------------------------- the embedding entry ------
#
# The one catalog entry whose capability is not a chat capability. It is checked
# here rather than in a file of its own because everything worth pinning about it is
# provenance: the record behind its verified flag, and the recipe that record was
# taken on.

def test_the_embedding_entry_states_the_recipe_its_record_was_taken_on():
    """Nothing here is guessable and the launch fails without it: a pooling runner
    (the checkpoint has no LM head to sample from), a batch the engine will accept,
    and a gmu small enough to stack beside a chat model on the same GB10."""
    info = CURATED_CLUSTER_MODELS["qwen3-embedding-0.6b"]
    assert info.hf_repo == "Qwen/Qwen3-Embedding-0.6B"
    assert info.capabilities == ["embedding"]
    assert info.extra_vllm_args == ["--runner", "pooling",
                                    "--max-num-seqs", "64",
                                    "--enable-prefix-caching"]
    assert info.recommended_gmu == 0.06
    assert info.max_model_len == 8192
    # Fleet default image on purpose: the proven launch ran on the node's own engine
    # image, and stating a pinned one here would be a claim no record backs.
    assert info.engine_image == ""
    assert info.curated and info.recommended
    assert info.typical_ready_minutes == 1.5


def test_the_embedding_entry_is_verified_by_an_embed_bench_record():
    """Its record is an `embed` record, not a throughput one: the model generates no
    tokens, so there is no tok/s for one to contain."""
    import json

    info = CURATED_CLUSTER_MODELS["qwen3-embedding-0.6b"]
    assert info.verified and info.verified_on == "2026-09-19"
    record = RESULTS / info.verified_record
    assert record.exists()
    data = json.loads(record.read_text())
    assert data["model"]["id"] == info.hf_repo
    assert "embed" in data and "results" not in data
    assert data["embed"]["dimensions"] == 1024
    assert data["embed"]["quality"]["ordered"] is True


def test_embedding_is_the_only_capability_an_entry_may_carry_alone():
    """A catalog entry marked `embedding` must not also claim chat capabilities: the
    interface reads that one word to hide the chat controls and to keep the model out
    of the chat picker, so a mixed list would put a dead "Use in New Chat" button on
    the card."""
    chat_caps = {"tool_use", "reasoning", "code", "vision"}
    for model_id, info in _all_entries().items():
        caps = set(info.capabilities or [])
        if "embedding" in caps:
            assert not (caps & chat_caps), (
                f"{model_id} claims embedding and {sorted(caps & chat_caps)}")
