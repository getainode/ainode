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

Two numerics invariants live here, both paid for by the 2026-07-06 run that
"completed" with a 100 percent NaN adapter:

* Attention runs EAGER by default. The training image's memory-efficient SDPA
  kernels are built for sm80-sm100; on a GB10 (sm121) the cutlassF forward and
  the cutlassB backward both refuse to launch, the forward then returns zeros
  (loss lands exactly on ln(vocab) = 11.93) and the backward returns NaN.
* Pad positions are masked out of the labels (-100) and padding is dynamic per
  batch. Labels that copy the padded ``input_ids`` make the model predict pad
  for most of every sequence, which is a meaningless objective and a huge
  gradient.

A run whose loss or grad_norm goes non-finite now FAILS with
``AINODE_ERROR:NAN_LOSS`` instead of saving NaN weights and reporting success.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Log every step for this many steps before falling back to the configured
# cadence, so the loss curve the UI draws is real from the start.
WARM_LOGGING_STEPS = 20


class NonFiniteLoss(RuntimeError):
    """Raised from the training loop when loss or grad_norm goes non-finite."""


def _is_main_process() -> bool:
    """Rank-0 check that works both inside and outside torchrun."""
    rank = os.environ.get("RANK")
    return rank is None or rank == "0"


def _log(msg: str) -> None:
    """Print only from rank-0 to avoid N duplicate lines in multi-GPU runs."""
    if _is_main_process():
        print(msg, flush=True)


def _is_finite(value) -> bool:
    """True for a real finite number; True for anything that is not a number
    (so a callback never trips on a string or a dict in the log payload)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return math.isfinite(value)


def _non_finite_metrics(logs: dict) -> list[str]:
    """Names of the numeric metrics in ``logs`` that are NaN or inf."""
    watched = ("loss", "grad_norm", "eval_loss", "train_loss")
    return [k for k in watched if k in logs and not _is_finite(logs[k])]


def assert_finite_metrics(logs: dict, step: int) -> None:
    """Raise NonFiniteLoss if a logged metric is NaN or inf.

    One non-finite gradient is terminal for a run: Adam writes NaN into the
    adapter and every later step keeps it there. The 2026-07-06 job logged
    grad_norm 3.3e6 then NaN, kept going for 30 more steps, saved a 100 percent
    NaN adapter and reported COMPLETED."""
    bad = _non_finite_metrics(logs or {})
    if bad:
        detail = ", ".join(f"{k}={logs[k]}" for k in bad)
        raise NonFiniteLoss(f"non-finite {detail} at step {step}")


def build_prompt_texts(examples: dict) -> list[str]:
    """Flatten one batch of dataset rows into training strings."""
    if "text" in examples:
        return list(examples["text"])
    if "instruction" in examples and "output" in examples:
        return [
            f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            for inst, out in zip(examples["instruction"], examples["output"])
        ]
    if "prompt" in examples and "completion" in examples:
        return [f"{p}{c}" for p, c in zip(examples["prompt"], examples["completion"])]
    keys = [k for k in examples.keys() if isinstance(examples[k][0], str)]
    return [
        " ".join(examples[k][i] for k in keys)
        for i in range(len(examples[keys[0]]))
    ]


def make_tokenize_fn(tokenizer, max_seq_length: int):
    """Return the batched ``dataset.map`` function used for every method.

    Truncates but NEVER pads: the collator pads each batch to its own longest row
    and pads the labels with -100. Padding to ``max_length`` here and copying
    ``input_ids`` into ``labels`` (what this did until 0.5.26) asked the model to
    predict the pad token at ~90 percent of every position, which pinned the loss
    at ln(vocab) and blew the gradient up by six orders of magnitude."""

    def tokenize_fn(examples):
        texts = build_prompt_texts(examples)
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
        )
        masks = enc.get("attention_mask")
        labels = []
        for i, ids in enumerate(enc["input_ids"]):
            mask = masks[i] if masks is not None else None
            if mask is None:
                labels.append(list(ids))
            else:
                # -100 is the ignore_index of the causal-LM loss. Anything the
                # attention mask already excludes must not be a training target.
                labels.append([
                    (tok if m == 1 else -100) for tok, m in zip(ids, mask)
                ])
        enc["labels"] = labels
        return enc

    return tokenize_fn


def _scan_saved_weights(output_dir: Path) -> list[str]:
    """Read back every ``*.safetensors`` file just written and return a
    ``file:tensor`` description for each tensor holding a non-finite value.

    Cheap (the adapter is a few MB) and the only check that speaks for the
    artifact itself rather than for the loss curve."""
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError:
        return []

    bad: list[str] = []
    for path in sorted(output_dir.glob("*.safetensors")):
        try:
            tensors = load_file(str(path))
        except Exception as exc:  # unreadable is a different failure, not NaN
            _log(f"WARNING: could not scan {path.name} for NaN: {exc}")
            continue
        for name, tensor in tensors.items():
            if not tensor.is_floating_point():
                continue
            if not bool(torch.isfinite(tensor).all()):
                count = int((~torch.isfinite(tensor)).sum())
                bad.append(f"{path.name}:{name} ({count} non-finite)")
    return bad


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
        # The ainode package isn't installed in the spawned training container —
        # fall back to the AINODE_HOME env var (the container run sets it to /job).
        try:
            from ainode.core.config import AINODE_HOME as _ainode_home
            ainode_home = Path(_ainode_home)
        except Exception:
            ainode_home = Path(os.environ.get("AINODE_HOME", str(Path.home() / ".ainode")))
        resolved = ainode_home / "datasets" / ds
        if resolved.exists():
            config["dataset_path"] = str(resolved)

    # Resolve an on-disk base_model slug to a local directory. The container path
    # (engine._build_container_command) already rewrites this to /ainode-models/<slug>;
    # this keeps the host-venv path consistent so a slug like
    # "qwen--qwen2.5-0.5b-instruct" loads from the models store instead of choking
    # AutoTokenizer.from_pretrained on the '--' (HFValidationError). Only rewrites
    # when a matching directory exists, else the hub repo id passes through.
    bm = config.get("base_model", "")
    if bm and "/" not in bm and not bm.startswith("/") and not bm.startswith("~"):
        try:
            from ainode.core.config import AINODE_HOME as _bm_home
            _home = Path(_bm_home)
        except Exception:
            _home = Path(os.environ.get("AINODE_HOME", str(Path.home() / ".ainode")))
        _cand = _home / "models" / bm
        if _cand.is_dir():
            config["base_model"] = str(_cand)

    # Inject HF token if provided in config — needed for gated repos (Llama etc.)
    hf_token = config.get("hf_token") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        _log("HF token set — gated model access enabled")

    # DDP env var validation: fail fast with an actionable message instead of a
    # cryptic NCCL or socket timeout buried in torchrun output.
    world_size_str = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(world_size_str)
    except ValueError:
        world_size = 1

    if world_size > 1:
        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")
        if not master_addr:
            print(
                "ERROR: MASTER_ADDR is not set. Multi-node DDP requires MASTER_ADDR "
                "to be the IP of the head node (e.g. MASTER_ADDR=10.0.0.1). "
                "Set it before launching torchrun.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not master_port:
            os.environ["MASTER_PORT"] = "29500"
            _log("MASTER_PORT not set — defaulting to 29500")

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
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
    resume_from_checkpoint = config.get("_resume_from_checkpoint")  # set by resume endpoint
    eval_split = float(config.get("eval_split", 0.1))
    eval_steps = int(config.get("eval_steps", 0))
    wandb_project = config.get("wandb_project") or None
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
    logging_steps = max(1, int(config.get("logging_steps", 10)))
    # "eager" is the default for the reason in the module docstring. "auto" hands
    # the choice back to transformers; any other value is passed through as-is
    # (an operator overriding this owns the numerics).
    attn_implementation = (config.get("attn_implementation") or "eager").strip()
    attn_kwargs = {} if attn_implementation == "auto" else {
        "attn_implementation": attn_implementation
    }

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _log(f"Method: {method} · world_size={world_size} · rank={os.environ.get('RANK', '0')}")
    _log(
        f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"attn_implementation={attn_implementation}"
    )
    if torch.cuda.is_available():
        _log(
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"sm{''.join(str(n) for n in torch.cuda.get_device_capability(0))}"
        )
    else:
        _log("WARNING: CUDA is not available: training will run on CPU and be very slow.")

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
            **attn_kwargs,
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
            **attn_kwargs,
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
            **attn_kwargs,
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

    tokenize_fn = make_tokenize_fn(tokenizer, max_seq_length)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    if len(dataset) > 0:
        first = dataset[0]
        supervised = sum(1 for lab in first["labels"] if lab != -100)
        _log(
            f"Tokenized {len(dataset)} samples · first sample: "
            f"{supervised}/{len(first['labels'])} positions supervised "
            f"(dynamic padding, pad labels -100)"
        )

    # Split into train / eval if requested
    eval_dataset = None
    train_dataset = dataset
    if eval_split > 0 and len(dataset) > 10:
        split = dataset.train_test_split(test_size=min(eval_split, 0.2), seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        _log(f"Dataset split: {len(train_dataset)} train / {len(eval_dataset)} eval samples")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    class ProgressCallback(TrainerCallback):
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs

        def on_log(self, _args, state, control, logs=None, **kwargs):
            # A non-finite loss or gradient means the run is already dead: every
            # later step feeds NaN into Adam and the saved adapter is all NaN.
            # Abort here so the job FAILS instead of "completing" with junk.
            # Runs on every rank: a NaN on rank 3 is still a dead run.
            assert_finite_metrics(logs or {}, state.global_step)
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
                # Include eval metrics if present
                if "eval_loss" in logs:
                    payload["eval_loss"] = round(logs["eval_loss"], 4)
                if "eval_runtime" in logs:
                    payload["eval_samples_per_second"] = round(
                        logs.get("eval_samples_per_second", 0), 2
                    )
                print(f"AINODE_PROGRESS:{json.dumps(payload)}", flush=True)

    class WarmLoggingCallback(TrainerCallback):
        """Log every step for the first ``WARM_LOGGING_STEPS``, then drop to the
        configured cadence. The early steps are where a run goes wrong, and the
        browser's loss curve is drawn from exactly these log events."""

        def __init__(self, steady_steps: int):
            self.steady_steps = steady_steps

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step < WARM_LOGGING_STEPS:
                return
            # transformers 5.x reads state.logging_steps in DefaultFlowCallback;
            # older releases read args.logging_steps. Set both.
            if getattr(state, "logging_steps", None) == 1:
                state.logging_steps = self.steady_steps
            if getattr(args, "logging_steps", None) == 1:
                args.logging_steps = self.steady_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,  # WarmLoggingCallback raises this after WARM_LOGGING_STEPS
        # Trainer replaces a NaN/inf loss with the running mean by default, which
        # is exactly how a dead run reported a plausible-looking 0.0 loss and
        # "completed". Keep the real number so the NaN guard can see it.
        logging_nan_inf_filter=False,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=["wandb"] if wandb_project else ["none"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=0.03 if warmup_steps == 0 else 0.0,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        optim=("paged_adamw_8bit" if method == "qlora" else "adamw_torch"),
        gradient_checkpointing=use_gradient_checkpointing,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        # Evaluation
        eval_strategy="steps" if (eval_dataset and eval_steps > 0) else ("epoch" if eval_dataset else "no"),
        eval_steps=eval_steps if eval_steps > 0 else None,
        per_device_eval_batch_size=max(1, batch_size // 2),
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
    )

    # Dynamic padding per batch, with pad positions kept out of the loss. This is
    # the collator half of the label masking done in tokenize_fn: it pads the
    # labels with -100 rather than with the pad token.
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[ProgressCallback(num_epochs), WarmLoggingCallback(logging_steps)],
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # transformers < 4.46 spells it `tokenizer=`.
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    # Configure W&B if requested
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if config.get("run_name"):
            os.environ["WANDB_NAME"] = config["run_name"]
        _log(f"W&B logging enabled → project: {wandb_project}")

    _log("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    except NonFiniteLoss as exc:
        print(
            f"AINODE_ERROR:NAN_LOSS: training diverged: {exc}. Nothing was saved; "
            f"a NaN adapter is worse than no adapter. Most likely causes: a "
            f"learning rate too high for this model (currently {learning_rate}), "
            f"or an attention kernel that does not support this GPU. The default "
            f"attn_implementation is 'eager' for that reason, and this run used "
            f"'{attn_implementation}'.",
            file=sys.stderr, flush=True,
        )
        sys.exit(1)
    except RuntimeError as exc:
        err_str = str(exc)
        # Provide actionable messages for the most common GPU errors
        if "out of memory" in err_str.lower() or "cuda out of memory" in err_str.lower():
            print(
                f"AINODE_ERROR:CUDA_OOM — GPU ran out of memory. "
                f"Try: lower batch_size (currently {batch_size}), "
                f"enable gradient_checkpointing, or use QLoRA instead of LoRA/full. "
                f"Original error: {err_str}",
                file=sys.stderr, flush=True,
            )
        elif "cuda" in err_str.lower() or "nccl" in err_str.lower():
            print(
                f"AINODE_ERROR:CUDA_ERROR — GPU/NCCL error during training. "
                f"Check GPU health with nvidia-smi. "
                f"Original error: {err_str}",
                file=sys.stderr, flush=True,
            )
        elif "address already in use" in err_str.lower():
            print(
                f"AINODE_ERROR:DDP_PORT_CONFLICT — Port {os.environ.get('MASTER_PORT', '29500')} "
                f"is already in use. Another training job may be running. "
                f"Original error: {err_str}",
                file=sys.stderr, flush=True,
            )
        else:
            print(f"AINODE_ERROR:TRAINING_FAILED — {err_str}", file=sys.stderr, flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        _log("Training interrupted by user.")
        sys.exit(130)

    # Only rank-0 writes artifacts.
    if _is_main_process():
        _log(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Last line of defence: the 2026-07-06 run reported success with an
        # adapter that was 100 percent NaN. Refuse to call a run complete
        # without reading back what it wrote.
        bad_tensors = _scan_saved_weights(Path(output_dir))
        if bad_tensors:
            print(
                "AINODE_ERROR:NAN_WEIGHTS: the saved weights contain non-finite "
                "values, so this run produced nothing usable: "
                + "; ".join(bad_tensors[:5])
                + (f" (and {len(bad_tensors) - 5} more)" if len(bad_tensors) > 5 else ""),
                file=sys.stderr, flush=True,
            )
            sys.exit(1)

        print(
            f"AINODE_PROGRESS:{json.dumps({'epoch': num_epochs, 'loss': 0, 'progress': 100.0})}",
            flush=True,
        )
        _log("Training complete.")


if __name__ == "__main__":
    main()
