"""Tests for consume_start_clean — the closet #310 one-shot 'start clean' knob.

Skips replaying persisted models on a boot so an operator can `restart` to free
a node. Triggered by env AINODE_START_CLEAN or a consumed sentinel file.
"""

import os
from pathlib import Path

from ainode.core.config import AINODE_HOME
from ainode.models.api_routes import consume_start_clean

SENTINEL = Path(AINODE_HOME) / ".start-clean"


def _reset():
    os.environ.pop("AINODE_START_CLEAN", None)
    if SENTINEL.exists():
        SENTINEL.unlink()


def test_neither_is_false():
    _reset()
    assert consume_start_clean() is False


def test_env_triggers():
    _reset()
    os.environ["AINODE_START_CLEAN"] = "1"
    try:
        assert consume_start_clean() is True
    finally:
        os.environ.pop("AINODE_START_CLEAN", None)


def test_sentinel_triggers_and_is_consumed():
    _reset()
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text("")
    assert consume_start_clean() is True
    assert not SENTINEL.exists()  # single-use


if __name__ == "__main__":
    test_neither_is_false()
    test_env_triggers()
    test_sentinel_triggers_and_is_consumed()
    print("ok")
