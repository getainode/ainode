# CLAUDE.md — AINode Product

## Project Overview

AINode — Turn any NVIDIA GPU into a local AI platform. Inference + fine-tuning in your browser. One command to start, automatic clustering.

- **Product repo:** https://github.com/getainode/ainode
- **Marketing site repo:** https://github.com/getainode/ainode.dev
- **Live site:** https://ainode.dev
- **Powered by:** argentos.ai
- **Public docs:** https://docs.argentos.ai (repo: /Users/sem/code/mintlify-docs)
- **License:** Apache 2.0

## Tech Stack

- Python 3.10+
- vLLM (inference engine)
- aiohttp (API server + web UI serving)
- pynvml + psutil (GPU detection)
- Rich (terminal UI)

## Key Commands

```bash
pip install -e .            # Install in dev mode
pip install -e ".[engine]"  # Install with vLLM
pip install -e ".[dev]"     # Install with test deps
ainode start                # Start AINode
ainode status               # Show cluster status
ainode models               # List models
pytest tests/               # Run tests
```

## Architecture

```
ainode/
├── core/          # Config, GPU detection
├── engine/        # vLLM wrapper
├── api/           # API proxy (OpenAI-compatible)
├── web/           # Embedded chat UI
├── discovery/     # UDP node discovery
├── cli/           # CLI entry point
├── onboarding/    # First-run setup
└── training/      # Fine-tuning (future)
```

## Working Conventions

- Follow ops-approved workflow (see ops/)
- All work on `codex/*` branches
- PRs required — never push directly to main
- Handoffs use the threadmaster-handoff runbook
- Test on real GPU hardware when possible

## Target Hardware

- NVIDIA DGX Spark (GB10, 128 GB unified memory)
- ASUS/Dell/HP GB10 variants
- Any Linux system with NVIDIA GPU + CUDA

## Brand

- "Powered by argentos.ai" in all CLI output and web UI footer
- Product name: AINode (capital A, capital I, capital N)
