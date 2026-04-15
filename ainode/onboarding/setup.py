"""First-run onboarding flow."""

import re
from ainode.core.config import NodeConfig
from ainode.core.gpu import detect_gpu


def run_onboarding(config: NodeConfig) -> NodeConfig:
    """Interactive onboarding for first-time users."""

    print("  Welcome! Let's set up your AI node.\n")

    # Step 1: GPU detection
    gpu = detect_gpu()
    if gpu:
        mem_gb = gpu.memory_total_mb / 1024
        print(f"  Detected: {gpu.name} ({mem_gb:.0f} GB)")

        # Suggest a model based on memory
        if mem_gb >= 80:
            suggestion_note = "70B (you have plenty of memory)"
        elif mem_gb >= 20:
            suggestion_note = "8B (good balance of quality and speed)"
        else:
            suggestion_note = "3B (fits your memory)"
        print(f"  Suggested model: {suggestion_note}\n")
    else:
        print("  No NVIDIA GPU detected. AINode will run in CPU mode.\n")

    # Step 2: Email (for updates and support)
    print("  Step 1/3 — Enter your email (for updates, optional)")
    email = input("  Email: ").strip()
    if email and _is_valid_email(email):
        config.email = email
    elif email:
        print("  Invalid email, skipping.\n")

    # Step 3: Model selection
    # All defaults are open-access (no Hugging Face login required).
    # Llama variants require a gated HF token — we don't prompt for that
    # here so we default to Qwen 2.5 which is fully open.
    print("\n  Step 2/3 — Choose your model")
    print("  [1] Qwen 2.5 1.5B  (quick start, ~3 GB, no login required)")
    print("  [2] Qwen 2.5 7B    (recommended, ~15 GB, no login required)")
    print("  [3] Qwen 2.5 72B   (advanced, ~40 GB AWQ, no login required)")
    print("  [4] Custom model   (Hugging Face repo ID — gated repos need HF_TOKEN)")
    print()

    choice = input("  Choice [2]: ").strip() or "2"

    models = {
        "1": "Qwen/Qwen2.5-1.5B-Instruct",
        "2": "Qwen/Qwen2.5-7B-Instruct",
        "3": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    }

    if choice == "4":
        custom = input("  Model name (HuggingFace repo ID): ").strip()
        if custom:
            config.model = custom
            hf_token = input(
                "  HF token (leave blank if model is public): "
            ).strip()
            if hf_token:
                config.hf_token = hf_token
    elif choice in models:
        config.model = models[choice]
        if choice == "3":
            config.quantization = "awq"
    else:
        config.model = models["2"]

    # Step 4: Node name
    print("\n  Step 3/3 — Name this node (optional)")
    name = input("  Node name [my-ainode]: ").strip() or "my-ainode"
    config.node_name = name

    # Save
    config.onboarded = True
    config.save()

    print(f"\n  Setup complete! Starting {config.model.split('/')[-1]}...\n")
    return config


def _is_valid_email(email: str) -> bool:
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))
