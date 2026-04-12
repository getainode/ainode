"""Internal training script — launched as a subprocess by TrainingJob.

Reads a config JSON and runs HuggingFace Transformers + PEFT training,
emitting structured progress lines for the parent process to parse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if method == "lora":
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            print("PEFT is required for LoRA training: pip install peft", file=sys.stderr)
            sys.exit(1)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load dataset (supports JSON, JSONL, CSV, or HF dataset name)
    print(f"Loading dataset: {dataset_path}")
    if Path(dataset_path).exists():
        ext = Path(dataset_path).suffix.lower()
        if ext in (".json", ".jsonl"):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif ext == ".csv":
            dataset = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        # Treat as HuggingFace dataset name
        dataset = load_dataset(dataset_path, split="train")

    # Tokenize
    def tokenize_fn(examples):
        # Support common dataset formats
        if "text" in examples:
            texts = examples["text"]
        elif "instruction" in examples and "output" in examples:
            texts = [
                f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                for inst, out in zip(examples["instruction"], examples["output"])
            ]
        elif "prompt" in examples and "completion" in examples:
            texts = [
                f"{p}{c}" for p, c in zip(examples["prompt"], examples["completion"])
            ]
        else:
            # Fallback: concatenate all string fields
            keys = [k for k in examples.keys() if isinstance(examples[k][0], str)]
            texts = [" ".join(examples[k][i] for k in keys) for i in range(len(examples[keys[0]]))]

        return tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    # Set labels = input_ids for causal LM
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]})

    # Progress callback
    class ProgressCallback(TrainerCallback):
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs

        def on_log(self, _args, state, control, logs=None, **kwargs):
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
        gradient_accumulation_steps=max(1, 8 // batch_size),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProgressCallback(num_epochs)],
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"AINODE_PROGRESS:{json.dumps({'epoch': num_epochs, 'loss': 0, 'progress': 100.0})}")
    print("Training complete.")


if __name__ == "__main__":
    main()
