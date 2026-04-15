"""Internal training script — launched as a subprocess by TrainingJob.

Reads a config JSON and runs HuggingFace Transformers + PEFT training.
Emits structured progress lines (``AINODE_PROGRESS:{json}``) for the
parent process to parse.

Supports three methods:

  lora   — PEFT LoRA adapters on top of the full-precision model
  qlora  — bitsandbytes 4-bit NF4 quantised base + PEFT LoRA adapters
  full   — standard full fine-tune (no PEFT)

All three methods are DDP-aware. When launched via ``torchrun``, the
``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK`` env vars are honoured by
HuggingFace ``Trainer`` automatically — this script just has to avoid
duplicate setup on non-rank-zero workers (logging, final save, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _is_main_process() -> bool:
    """Rank-0 check that works both inside and outside torchrun."""
    rank = os.environ.get("RANK")
    return rank is None or rank == "0"


def _log(msg: str) -> None:
    """Print only from rank-0 to avoid N duplicate lines in multi-GPU runs."""
    if _is_main_process():
        print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AINode training runner")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())

    # Resolve relative dataset paths to ~/.ainode/datasets/
    ds = config.get("dataset_path", "")
    if ds and not ds.startswith("/") and not ds.startswith("~"):
        from ainode.core.config import AINODE_HOME
        resolved = AINODE_HOME / "datasets" / ds
        if resolved.exists():
            config["dataset_path"] = str(resolved)

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            TrainerCallback,
        )
        from datasets import load_dataset
    except ImportError as exc:
        print(
            f"Missing training dependency: {exc}. "
            "Install with: pip install torch transformers datasets peft",
            file=sys.stderr,
        )
        sys.exit(1)

    base_model = config["base_model"]
    dataset_path = config["dataset_path"]
    output_dir = config.get("output_dir", "./output")
    method = config.get("method", "lora")
    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("batch_size", 4)
    learning_rate = config.get("learning_rate", 2e-4)
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 32)
    max_seq_length = config.get("max_seq_length", 2048)
    gradient_accumulation_steps = max(
        1, int(config.get("gradient_accumulation_steps", 8 // max(1, batch_size)))
    )
    warmup_steps = int(config.get("warmup_steps", 0))
    weight_decay = float(config.get("weight_decay", 0.0))
    use_gradient_checkpointing = bool(config.get("use_gradient_checkpointing", False))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _log(f"Method: {method} · world_size={world_size} · rank={os.environ.get('RANK', '0')}")

    _log(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Load model — dispatch per method
    # ------------------------------------------------------------------
    if method == "qlora":
        # QLoRA = 4-bit NF4 base + LoRA adapters in bf16.
        try:
            import bitsandbytes  # noqa: F401 — presence check only
            from transformers import BitsAndBytesConfig
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError as exc:
            print(
                f"QLoRA requires bitsandbytes + peft: {exc}. "
                "Install with: pip install bitsandbytes peft",
                file=sys.stderr,
            )
            sys.exit(1)

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        _log(f"Loading model (4-bit NF4): {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_cfg,
            device_map={"": int(os.environ.get("LOCAL_RANK", "0"))} if world_size > 1 else "auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        if _is_main_process():
            model.print_trainable_parameters()

    elif method == "lora":
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            print("PEFT is required for LoRA training: pip install peft", file=sys.stderr)
            sys.exit(1)

        _log(f"Loading model (bf16): {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK", "0"))} if world_size > 1 else "auto",
            trust_remote_code=True,
        )
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        if _is_main_process():
            model.print_trainable_parameters()

    elif method == "full":
        _log(f"Loading model for full fine-tune (bf16): {base_model}")
        # Under DDP, let Trainer place the model per LOCAL_RANK.
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None if world_size > 1 else "auto",
        )
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

    else:
        print(f"Unknown training method: {method}", file=sys.stderr)
        sys.exit(2)

    # ------------------------------------------------------------------
    # Dataset (supports JSON, JSONL, CSV, or HF dataset name)
    # ------------------------------------------------------------------
    _log(f"Loading dataset: {dataset_path}")
    if Path(dataset_path).exists():
        ext = Path(dataset_path).suffix.lower()
        if ext == ".csv":
            dataset = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            # JSON / JSONL / anything else we try to parse as JSONL
            dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")

    def tokenize_fn(examples):
        if "text" in examples:
            texts = examples["text"]
        elif "instruction" in examples and "output" in examples:
            texts = [
                f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                for inst, out in zip(examples["instruction"], examples["output"])
            ]
        elif "prompt" in examples and "completion" in examples:
            texts = [f"{p}{c}" for p, c in zip(examples["prompt"], examples["completion"])]
        else:
            keys = [k for k in examples.keys() if isinstance(examples[k][0], str)]
            texts = [
                " ".join(examples[k][i] for k in keys)
                for i in range(len(examples[keys[0]]))
            ]

        return tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]})

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    class ProgressCallback(TrainerCallback):
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs

        def on_log(self, _args, state, control, logs=None, **kwargs):
            # Only rank-0 emits progress so the parent process doesn't
            # see N copies per step.
            if not _is_main_process():
                return
            if logs and "loss" in logs:
                epoch = state.epoch or 0
                progress = (epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0
                payload = {
                    "epoch": int(epoch),
                    "loss": round(logs["loss"], 4),
                    "progress": round(progress, 1),
                    "step": state.global_step,
                }
                print(f"AINODE_PROGRESS:{json.dumps(payload)}", flush=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=0.03 if warmup_steps == 0 else 0.0,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        optim=("paged_adamw_8bit" if method == "qlora" else "adamw_torch"),
        gradient_checkpointing=use_gradient_checkpointing,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProgressCallback(num_epochs)],
    )

    _log("Starting training...")
    trainer.train()

    # Only rank-0 writes artifacts.
    if _is_main_process():
        _log(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(
            f"AINODE_PROGRESS:{json.dumps({'epoch': num_epochs, 'loss': 0, 'progress': 100.0})}",
            flush=True,
        )
        _log("Training complete.")


if __name__ == "__main__":
    main()
