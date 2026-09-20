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
- **A dataset shape the tokenizer has no branch for is a crash, not a fallback.**
  `conversation_column` / `render_conversations` handle a `conversations` or
  `messages` column through `tokenizer.apply_chat_template` (`from`/`value` and
  `role`/`content` both mapped by `ROLE_ALIASES`), because that is what AutoData
  writes and what `sharegpt-chat` advertises; without that branch
  `build_prompt_texts`'s generic join raised IndexError and the documented handoff
  could not run. The whole rendered conversation is supervised: assistant-only
  masking is NOT implemented, so no docstring, template or README line may claim
  it. A template's `sample_shape` is a promise the runner has to keep.
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
  refuses** (`handle_submit_job` catches `RuntimeError` from `start_next`), and so
  does `POST /api/training/jobs/{id}/resume`.
- **Every job's status is a file in its job dir, and the registry is the disk.**
  `TrainingJob.status` is a property whose setter writes `status.json`
  (`STATUS_FILENAME`). There is a setter because eight places move a job's status,
  and one of them forgetting is how the registry drifts from the disk again. `TrainingManager.__init__` rebuilds from `JOBS_DIR` through
  `load_jobs_from_disk`. Rules that hold there: a directory with neither a status
  file nor a config is NOT a job (the suite once left 6,000 empty ones); a
  recorded RUNNING or PENDING job comes back FAILED with a note, never adopted;
  a job dir with no status file gets a best-effort status with a `note` and
  `restored: True` and **no `start_time`**, because `stats()` counts GPU hours
  from it and a duration nobody recorded must not be invented. Secrets are
  stripped from `status.json` exactly as from `config.json`.
- **The queue advances from the monitor.** `TrainingJob._monitor` calls
  `_on_exit` (set by `start_next`) when the process exits, and a failure to start
  the next job is logged on the finished job, never raised out of the monitor
  task. Do not make the HTTP handlers the only thing that calls `start_next`
  again: that is how a second queued job waited for a human to submit a third.
- **Resume mounts the source job dir read-only at `/src` and rewrites the
  checkpoint path** (`_resume_mount_for`), and the resumed job gets its OWN
  `output_dir` (the API passes `output_dir: None`). Handing the container an
  orchestrator path with nothing mounted behind it is why "Can't find a valid
  checkpoint" was the only possible outcome before 0.5.27; reusing the source
  job's output dir is why neither run's output would have meant anything.
- **Every spawned container is preflighted, and the error names every candidate
  image.** `job_image_candidates` is the order (override → `ghcr_train_image()`
  → `LOCAL_TRAIN_IMAGE`), `resolve_job_image` picks what the node has for the
  argv and never raises, `assert_job_image_present` is the preflight and raises
  naming all of them. A broken docker is reported as a broken docker, not as a
  missing image. `ghcr.io/getainode/ainode-train:<version>` is published by
  `.github/workflows/publish-train-image.yml` on a `train-v*` tag: it is 22 GB
  and over an hour on a Spark, so nothing triggers it implicitly and no ordinary
  release rebuilds it.
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
- `tests/test_training_persistence.py` owns the registry rebuild, the queue
  advance and resume. **A test that pins an image in an argv patches
  `_image_present`**: resolution asks the docker daemon, and a Spark running the
  suite really does have the hand-built train image, so without that the same test
  passes on a Mac and fails on a node.
