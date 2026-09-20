"""Tests for the training runner's numerics contract (_run_training.py).

The runner had no tests at all, which is how a LoRA run shipped an adapter that
was 100 percent NaN and reported COMPLETED (Spark-4, 2026-07-06, job
681b658ad647). These pin the three pieces that let that happen: how labels are
built, what happens when a metric goes non-finite, and the read-back of the
saved weights.

The module imports cleanly without torch (everything heavy is imported inside
main()), so these run on any dev box.
"""

from __future__ import annotations

import math
import sys
import types

import pytest

from ainode.training import _run_training as runner


# ---------------------------------------------------------------------------
# Tokenization: truncation only, labels -100 on pad
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Records how it was called and encodes each word as one token id."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        max_length = kwargs.get("max_length")
        input_ids, attention_mask = [], []
        for text in texts:
            ids = [ord(w[0]) for w in text.split()]
            if kwargs.get("truncation") and max_length:
                ids = ids[:max_length]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_tokenize_does_not_pad_to_max_length():
    """padding="max_length" is what made the model predict pad at ~90 percent of
    every position. The collator pads per batch instead."""
    tok = FakeTokenizer()
    fn = runner.make_tokenize_fn(tok, 256)
    out = fn({"text": ["a b c", "d e"]})

    assert tok.calls[0].get("padding") is None, "tokenizer must not pad"
    assert tok.calls[0]["truncation"] is True
    assert tok.calls[0]["max_length"] == 256
    # Rows keep their own lengths: no padding to a common width here.
    assert [len(ids) for ids in out["input_ids"]] == [3, 2]


def test_tokenize_truncates_at_max_seq_length():
    fn = runner.make_tokenize_fn(FakeTokenizer(), 3)
    out = fn({"text": ["a b c d e f"]})
    assert len(out["input_ids"][0]) == 3
    assert len(out["labels"][0]) == 3


def test_labels_are_minus_100_where_attention_mask_is_zero():
    """A row that arrives already padded must not be trained on its pad."""

    class PrePaddedTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": [[11, 12, 13, 0, 0]],
                "attention_mask": [[1, 1, 1, 0, 0]],
            }

    fn = runner.make_tokenize_fn(PrePaddedTokenizer(), 256)
    out = fn({"text": ["whatever"]})
    assert out["labels"] == [[11, 12, 13, -100, -100]]


def test_labels_mirror_input_ids_on_real_tokens():
    fn = runner.make_tokenize_fn(FakeTokenizer(), 256)
    out = fn({"prompt": ["a b "], "completion": ["c"]})
    assert out["labels"] == out["input_ids"]
    assert -100 not in out["labels"][0]


def test_prompt_texts_cover_the_supported_dataset_shapes():
    assert runner.build_prompt_texts({"text": ["one", "two"]}) == ["one", "two"]
    assert runner.build_prompt_texts(
        {"instruction": ["do it"], "output": ["done"]}
    ) == ["### Instruction:\ndo it\n\n### Response:\ndone"]
    assert runner.build_prompt_texts(
        {"prompt": ["q: "], "completion": ["a"]}
    ) == ["q: a"]


# ---------------------------------------------------------------------------
# Conversations: the format AutoData writes (issue #194)
# ---------------------------------------------------------------------------

class ChatTokenizer(FakeTokenizer):
    """FakeTokenizer with a canned chat template, the way a real chat model has one."""

    def __init__(self, template: bool = True):
        super().__init__()
        self.template = template
        self.rendered: list[list[dict]] = []

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if not self.template:
            raise ValueError("cannot use chat template: tokenizer has no template")
        assert tokenize is False, "the runner renders to text and tokenizes itself"
        self.rendered.append(messages)
        return "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages
        )


AUTODATA_ROW = [
    {"from": "human", "value": "who made you"},
    {"from": "gpt", "value": "AINode did"},
]


def test_autodata_rows_map_from_value_onto_role_content():
    assert runner.normalize_conversation(AUTODATA_ROW) == [
        {"role": "user", "content": "who made you"},
        {"role": "assistant", "content": "AINode did"},
    ]


def test_openai_style_rows_pass_through():
    assert runner.normalize_conversation(
        [{"role": "system", "content": "be terse"}, {"role": "assistant", "content": "ok"}]
    ) == [
        {"role": "system", "content": "be terse"},
        {"role": "assistant", "content": "ok"},
    ]


def test_turns_with_no_content_are_dropped_not_guessed():
    assert runner.normalize_conversation(
        [{"from": "human"}, "not a dict", {"from": "gpt", "value": "kept"}]
    ) == [{"role": "assistant", "content": "kept"}]


def test_conversations_go_through_the_chat_template():
    """The documented AutoData handoff: {"conversations": [...]} used to raise
    IndexError in dataset.map because the runner had no branch for it."""
    tok = ChatTokenizer()
    fn = runner.make_tokenize_fn(tok, 256)
    out = fn({"conversations": [AUTODATA_ROW]})

    assert tok.rendered == [[
        {"role": "user", "content": "who made you"},
        {"role": "assistant", "content": "AINode did"},
    ]]
    assert out["input_ids"] and out["labels"]
    assert len(out["labels"][0]) == len(out["input_ids"][0])
    # Padding still never happens here, and no pad position is a label.
    assert tok.calls[0].get("padding") is None
    assert -100 not in out["labels"][0]


def test_a_messages_column_is_treated_as_a_conversation():
    tok = ChatTokenizer()
    fn = runner.make_tokenize_fn(tok, 256)
    fn({"messages": [[{"role": "user", "content": "hi"}]]})
    assert tok.rendered == [[{"role": "user", "content": "hi"}]]


def test_a_messages_column_of_plain_strings_is_still_text():
    """Only a list of turns is a conversation; a string column is text."""
    assert runner.conversation_column({"messages": ["hello", "there"]}) is None
    assert runner.conversation_column({"conversations": [AUTODATA_ROW]}) == "conversations"


def test_a_tokenizer_with_no_chat_template_falls_back_to_plain_text():
    """A base model without a chat template must still train, not crash."""
    tok = ChatTokenizer(template=False)
    fn = runner.make_tokenize_fn(tok, 256)
    out = fn({"conversations": [AUTODATA_ROW]})
    assert out["input_ids"][0], "the row still tokenized"
    assert runner.plain_chat_text(runner.normalize_conversation(AUTODATA_ROW)) == (
        "### User:\nwho made you\n\n### Assistant:\nAINode did"
    )


def test_an_empty_conversation_renders_to_empty_text():
    tok = ChatTokenizer()
    assert runner.render_conversations(tok, [[]]) == [""]


# ---------------------------------------------------------------------------
# The NaN guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "logs",
    [
        {"loss": float("nan")},
        {"loss": 2.0, "grad_norm": float("nan")},
        {"loss": float("inf")},
        {"loss": 2.0, "grad_norm": float("-inf")},
        {"eval_loss": float("nan")},
        {"train_loss": float("nan")},
    ],
)
def test_non_finite_metric_aborts_the_run(logs):
    with pytest.raises(runner.NonFiniteLoss):
        runner.assert_finite_metrics(logs, step=7)


def test_nan_guard_names_the_metric_and_step():
    with pytest.raises(runner.NonFiniteLoss) as exc:
        runner.assert_finite_metrics({"grad_norm": float("nan")}, step=3)
    assert "grad_norm" in str(exc.value)
    assert "step 3" in str(exc.value)


@pytest.mark.parametrize(
    "logs",
    [
        {},
        {"loss": 2.5, "grad_norm": 3.3e6},  # huge but finite: not this guard's call
        {"loss": 0.0, "grad_norm": 0.0},
        {"learning_rate": 0.0002, "epoch": 1.0},
        {"loss": 1.0, "some_string": "n/a"},  # non-numeric values are ignored
    ],
)
def test_finite_metrics_pass(logs):
    runner.assert_finite_metrics(logs, step=1)  # must not raise


def test_nan_guard_is_a_runtime_error_subclass():
    """main() catches NonFiniteLoss before its generic RuntimeError handler, so
    the ordering of those excepts matters; keep the relationship pinned."""
    assert issubclass(runner.NonFiniteLoss, RuntimeError)


# ---------------------------------------------------------------------------
# Read-back of the saved weights
# ---------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, values, floating=True):
        self.values = list(values)
        self._floating = floating

    def is_floating_point(self):
        return self._floating


class _Mask:
    """Result of torch.isfinite(t) / its inversion: enough surface for the scan."""

    def __init__(self, flags):
        self.flags = list(flags)

    def all(self):
        return all(self.flags)

    def sum(self):
        return sum(1 for f in self.flags if f)

    def __invert__(self):
        return _Mask([not f for f in self.flags])


@pytest.fixture
def fake_torch_stack(monkeypatch):
    """Install stub torch + safetensors.torch modules and return the file map the
    stub load_file() serves, so the scan can be tested without torch."""
    files: dict[str, dict] = {}

    fake_torch = types.ModuleType("torch")
    fake_torch.isfinite = lambda t: _Mask(math.isfinite(v) for v in t.values)

    fake_st = types.ModuleType("safetensors")
    fake_st_torch = types.ModuleType("safetensors.torch")
    fake_st_torch.load_file = lambda path: files[str(path)]
    fake_st.torch = fake_st_torch

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)
    return files


def test_scan_reports_non_finite_tensors(tmp_path, fake_torch_stack):
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(b"")
    fake_torch_stack[str(path)] = {
        "lora_A.weight": FakeTensor([0.1, 0.2]),
        "lora_B.weight": FakeTensor([float("nan"), 0.3, float("nan")]),
    }

    bad = runner._scan_saved_weights(tmp_path)

    assert len(bad) == 1
    assert "lora_B.weight" in bad[0]
    assert "2 non-finite" in bad[0]


def test_scan_passes_a_clean_adapter(tmp_path, fake_torch_stack):
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(b"")
    fake_torch_stack[str(path)] = {"lora_A.weight": FakeTensor([0.1, -0.2, 3.0])}
    assert runner._scan_saved_weights(tmp_path) == []


def test_scan_skips_integer_tensors(tmp_path, fake_torch_stack):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"")
    fake_torch_stack[str(path)] = {
        "counts": FakeTensor([float("nan")], floating=False),
    }
    assert runner._scan_saved_weights(tmp_path) == []


def test_scan_without_torch_is_a_no_op(tmp_path, monkeypatch):
    """The scan is a guard, not a dependency: with no torch it reports nothing
    rather than failing a run that is otherwise fine."""
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    assert runner._scan_saved_weights(tmp_path) == []


def test_warm_logging_window_is_twenty_steps():
    """The UI's loss curve is drawn from log events; the first steps are where a
    run goes wrong, so they are logged one by one."""
    assert runner.WARM_LOGGING_STEPS == 20
