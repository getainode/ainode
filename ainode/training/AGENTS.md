# training/ AGENTS.md (edit contract)

Parent: `../../AGENTS.md` · State / "why" / history: Obsidian Vault → `AINode`.

Fine-tuning ran end to end exactly once before 0.5.26 and the adapter it produced
was 100 percent NaN, so the rules below are not preferences. Each one has a run
behind it (Spark-4 2026-07-06 for the failure, Spark-3 2026-09-19 for the fix).

## Numerics (dangerous, read before touching `_run_training.py`)

- **Training loads the model with `attn_implementation="eager"` by default.** The
  train image's memory-efficient SDPA kernels are compiled for sm80-sm100. On a
  GB10 (sm121) `fmha_cutlassF` and `fmha_cutlassB` refuse to launch, torch prints
  `FATAL: kernel ... is for sm80-sm100, but was built for sm121` per call, the
  forward returns zeros (loss lands exactly on ln(vocab)) and the backward
  returns NaN from step one. Do not "optimize" this back to SDPA or flash
  attention without a run on the target GPU whose loss descends and whose adapter
  scans clean. `attn_implementation` is a per-job config field for the operator
  who has checked; the default belongs to whatever is proven on sm121.
- **Never pad in the tokenizer, and never let a pad position be a label.**
  Tokenization truncates only; `DataCollatorForSeq2Seq` pads each batch with
  `label_pad_token_id=-100`, and labels are `-100` wherever the attention mask is
  0. `padding="max_length"` with `labels=input_ids` asks the model to predict pad
  at most positions: it pins the loss at ln(vocab) and inflates the gradient by
  orders of magnitude. Keep the tokenization in `make_tokenize_fn` so it stays
  testable without torch.
- **`logging_nan_inf_filter` stays False.** Trainer's default replaces a
  non-finite loss with the running mean, which is how a dead run logged
  `loss: 0.0` and reported `COMPLETED`.
- **A non-finite loss or grad_norm ends the run.** `assert_finite_metrics` raises
  `NonFiniteLoss` from `on_log`; `main()` catches it BEFORE its generic
  `RuntimeError` handler (ordering matters: `NonFiniteLoss` is a `RuntimeError`)
  and exits non-zero with `AINODE_ERROR:NAN_LOSS`. Nothing is saved.
- **The saved weights are read back before a run is called complete.**
  `_scan_saved_weights` scans every `*.safetensors` in the output dir and the run
  exits non-zero with `AINODE_ERROR:NAN_WEIGHTS` on any non-finite value. A guard
  on the artifact, not on the loss curve: keep it.
- **`_run_training.py` and `_run_merge.py` must stay self-contained.** They are
  copied into the job dir and executed inside the train image, which has no
  `ainode` package: every `ainode.*` import in them is inside a `try` with a
  working fallback. The image bakes only `_run_quant.py`
  (`scripts/Dockerfile.quant`), so a fix to the training runner reaches the
  container through that copy, and nothing needs rebuilding.

## Job lifecycle

- **A job becomes RUNNING only after `Popen` returns a process**, and
  `TrainingManager.start_next()` claims `_active_job_id` only after
  `job.start()` returns. Setting either earlier leaves a phantom RUNNING job with
  no process when the command cannot be built, and that phantom blocks the queue
  for every later job.
- **`POST /api/training/jobs` answers 400, never 500, for a job the engine
  refuses** (`handle_submit_job` catches `RuntimeError` from `start_next`).
- **Every spawned container is preflighted with `_assert_image_present`** before
  launch, with an error naming the image and its build. The train image is
  `TRAIN_IMAGE` (= `QUANT_IMAGE`), hand-built and not published by CI, so a fresh
  node does not have it.
- **A template exists only if the runner implements it.** `dpo-preference`
  (trains on the rejected answer through the SFT loop) and `distributed-ddp` (no
  launch path) were removed in 0.5.26 and are listed in `RETIRED_TEMPLATE_IDS`;
  `tests/test_training_command.py` fails if either is offered again. Re-add one
  together with its implementation and a proven run, not before.

## Secrets

- **The HF token never goes into a job file or a job log.** `GET
  /api/training/jobs/{id}/logs` serves the log verbatim and the job dir is read
  by the API, so: strip it from the config written to disk
  (`_config_without_secrets`), pass it with `--env-file` from
  `_write_token_env_file` (mode 0600, removed when the container exits), and log
  every launch line through `scrub_command`. A new `-e SOMETHING_TOKEN=` in a
  command is a leak; add the key to `SECRET_ENV_KEYS` if one is unavoidable.

## Tests

- `pytest tests/test_training_runner.py` covers the runner with no torch
  installed; `tests/test_training_gpu.py` is marked `gpu` and runs a real LoRA
  job (set `AINODE_SMOKE_BASE_MODEL` to a small local model dir). Run the GPU one
  inside the train image on a node with a GPU before changing numerics.
- Tests must not write under the operator's real `~/.ainode`: a `TrainingJob`
  mkdirs its job dir on construction, so keep the autouse fixtures that redirect
  `AINODE_HOME` and `JOBS_DIR`.
