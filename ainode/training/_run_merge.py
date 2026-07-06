"""Internal adapter-merge runner — launched as a subprocess INSIDE a GPU
container (TRAIN_IMAGE + a peft pip-shim), not the slim orchestrator.

Reads a config JSON ({base_model, adapter_dir, output_dir, hf_token}), applies a
LoRA/QLoRA adapter onto its base model, merges the weights, and writes a
standalone servable checkpoint. Emits ``AINODE_PROGRESS:{json}`` lines the parent
process parses — same protocol as _run_training.py / _run_quant.py.

Self-contained: imports nothing from the ainode package (it isn't installed in
the container image).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _progress(pct: float, msg: str = "") -> None:
    print("AINODE_PROGRESS:" + json.dumps({"pct": round(pct, 1), "msg": msg}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AINode adapter-merge runner")
    parser.add_argument("--config", required=True, help="Path to merge config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(config_path.read_text())
    base_model = cfg["base_model"]
    adapter_dir = cfg["adapter_dir"]
    output_dir = cfg["output_dir"]

    token = cfg.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        print(
            f"Missing merge dependency: {exc}. "
            "Install with: pip install torch transformers peft",
            file=sys.stderr,
        )
        sys.exit(1)

    _progress(5, f"loading base model {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    _progress(40, f"applying adapter {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()

    _progress(75, f"saving merged model to {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    _progress(100, "merge complete")


if __name__ == "__main__":
    main()
