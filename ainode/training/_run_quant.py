"""Internal quantization runner — launched as a subprocess INSIDE a GPU
container (the NVIDIA vLLM image + llm-compressor), not the slim orchestrator.

Reads a config JSON and runs llm-compressor one-shot PTQ, producing a
compressed-tensors checkpoint vLLM serves natively. Emits structured progress
lines (``AINODE_PROGRESS:{json}``) the parent process parses — same protocol as
``_run_training.py``.

Schemes (Phase 1 — dense text models):
  awq    — W4A16 (4-bit weights, asym) via AWQModifier; serves as awq_marlin on GB10
  nvfp4  — NVFP4 (4-bit float, Blackwell-native) via QuantizationModifier

Single-node, single-GPU: a 9B fits one 122GB GB10 with room to spare. The HF
push (if any) is done by the orchestrator AFTER this job exits — this runner
only produces the on-disk checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ponytail: dense text models only (AutoModelForCausalLM, ignore=["lm_head"]).
# Multimodal (Qwen3.5-VL / Ornith) needs AutoProcessor + a vision ignore-list and
# is Phase 2 — see the locked contract.
_DEFAULT_IGNORE = ["lm_head"]


def _progress(phase: str, pct: float, msg: str = "") -> None:
    print("AINODE_PROGRESS:" + json.dumps({"phase": phase, "pct": round(pct, 1), "msg": msg}), flush=True)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _resolve_model(base_model: str) -> str:
    """A mounted on-disk dir wins over the HF repo id (offline + reproducible).

    The orchestrator mounts ~/.ainode/models at /ainode-models, so an installed
    model lives at /ainode-models/<org--name>. Fall back to the bare repo id
    (llm-compressor/transformers will pull it via HF) if not on disk.
    """
    slug = base_model.replace("/", "--")
    local = Path("/ainode-models") / slug
    if (local / "config.json").exists():
        _log(f"Using on-disk weights: {local}")
        return str(local)
    _log(f"On-disk weights not found at {local} — resolving '{base_model}' via HF")
    return base_model


def _build_calibration(dataset_id: str, tokenizer, n_samples: int, max_seq_length: int):
    from datasets import load_dataset

    # ultrachat_200k is the llm-compressor reference calibration set. Its split
    # is 'train_sft'; arbitrary text datasets fall back to 'train'.
    split = "train_sft" if "ultrachat" in dataset_id.lower() else "train"
    _log(f"Loading calibration dataset {dataset_id} [{split}], {n_samples} samples")
    ds = load_dataset(dataset_id, split=f"{split}[:{n_samples}]")
    ds = ds.shuffle(seed=42)

    def _preprocess(example):
        # Prefer a chat 'messages' column; else fall back to a 'text' column.
        if "messages" in example and example["messages"]:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        else:
            text = example.get("text") or example.get("content") or ""
        return {"text": text}

    ds = ds.map(_preprocess, remove_columns=[c for c in ds.column_names if c != "text"])

    def _tokenize(example):
        return tokenizer(
            example["text"], padding=False, max_length=max_seq_length,
            truncation=True, add_special_tokens=False,
        )

    return ds.map(_tokenize, remove_columns=["text"])


def _build_recipe(scheme: str, ignore: list):
    """Return an llm-compressor recipe (modifier or list) for the scheme."""
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if scheme == "nvfp4":
        return QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)
    if scheme == "awq":
        # Current (llm-compressor 0.9+) API: AWQModifier lives under
        # .transform.awq and pairs with a QuantizationModifier. The legacy
        # llmcompressor.modifiers.awq is a deprecated shim whose no-arg call left
        # the QuantizationModifier under-specified ("requires quantization fields").
        from llmcompressor.modifiers.transform.awq import AWQModifier
        return [
            AWQModifier(duo_scaling="both"),
            QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=ignore),
        ]
    raise ValueError(f"unsupported scheme '{scheme}' (expected awq|nvfp4)")


def main() -> None:
    parser = argparse.ArgumentParser(description="AINode quantization runner")
    parser.add_argument("--config", required=True, help="Path to quant config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    base_model = config["base_model"]
    scheme = (config.get("scheme") or "awq").strip().lower()
    # Output lands in the RW-mounted model store at /ainode-models/<out-slug> →
    # host ~/.ainode/models/<out-slug>, where the serve/catalog path auto-finds it.
    out_slug = config.get("out_slug") or (base_model.replace("/", "--") + "-" + scheme)
    output_dir = config.get("output_dir") or f"/ainode-models/{out_slug}"
    calib_dataset = config.get("calib_dataset") or "HuggingFaceH4/ultrachat_200k"
    n_samples = int(config.get("calib_samples", 256))
    max_seq_length = int(config.get("max_seq_length", 2048))
    ignore = config.get("ignore") or _DEFAULT_IGNORE

    hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    _progress("starting", 2, f"quantize {base_model} -> {scheme}")
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
    except ImportError as exc:
        print(
            f"Missing quantization dependency: {exc}. This runner must execute in a "
            "torch+CUDA container with llm-compressor (the derived vLLM image), NOT "
            "the slim orchestrator.",
            file=sys.stderr,
        )
        sys.exit(1)

    model_src = _resolve_model(base_model)

    _progress("loading_weights", 10, "loading base model (bf16)")
    model = AutoModelForCausalLM.from_pretrained(model_src, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_src)

    _progress("calibrating", 30, f"building {n_samples} calibration samples")
    calib = _build_calibration(calib_dataset, tokenizer, n_samples, max_seq_length)

    _progress("quantizing", 45, f"llm-compressor one-shot ({scheme})")
    recipe = _build_recipe(scheme, ignore)
    oneshot(
        model=model,
        dataset=calib,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=n_samples,
    )

    _progress("saving", 90, f"writing checkpoint -> {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    # Marker the orchestrator polls to confirm a real artifact (not an empty
    # --rm container layer — see the AINODE_HOST_HOME tripwire in the contract).
    cfgs = list(Path(output_dir).glob("*.safetensors"))
    if not cfgs:
        print("ERROR: no .safetensors written — output mount likely not host-backed "
              "(check AINODE_HOST_HOME).", file=sys.stderr)
        sys.exit(2)
    _progress("ready", 100, f"done — {len(cfgs)} shard(s) in {output_dir}")


if __name__ == "__main__":
    main()
