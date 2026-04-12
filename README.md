# AINode

### Turn any NVIDIA GPU into a local AI platform.

**Inference + fine-tuning in your browser. One command to start. Add a second GPU, they find each other automatically.**

```bash
curl -fsSL https://ainode.dev/install | bash
```

```
  ┌──────────────────────────────────────┐
  │         Welcome to AINode            │
  │   Your local AI platform for NVIDIA  │
  ├──────────────────────────────────────┤
  │                                      │
  │   Detected: NVIDIA GB10 · 128 GB     │
  │                                      │
  │   ✓ vLLM engine ready                │
  │   ✓ API server on :8000              │
  │   ✓ Web UI on :3000                  │
  │                                      │
  │   Open: http://localhost:3000         │
  │                                      │
  └──────────────────────────────────────┘
```

---

## What Is This?

AINode is a free, open-source tool that turns NVIDIA GPU systems into a complete local AI platform — chat, inference API, and model fine-tuning — all from your browser.

**Built for:**
- NVIDIA DGX Spark
- ASUS AI Server (GB10)
- Dell AI Factory
- HP AI Workstations
- Any system with an NVIDIA GPU and Linux

**No Linux expertise required.** Install, open your browser, start using AI.

---

## Features

| Feature | Status |
|---|---|
| One-command install | ✅ |
| Auto-detect GPU and memory | ✅ |
| Chat UI in your browser | ✅ |
| OpenAI-compatible API | ✅ |
| Model fine-tuning from browser | 🔜 |
| Multi-node auto-discovery | 🔜 |
| Automatic model sharding across nodes | 🔜 |

---

## Quick Start

### Install

```bash
curl -fsSL https://ainode.dev/install | bash
```

### Start

```bash
ainode start
```

Opens your browser to the AINode dashboard. Pick a model, start chatting.

### Add a Second Node

On your second NVIDIA box:

```bash
curl -fsSL https://ainode.dev/install | bash
ainode start
```

AINode discovers the first node automatically. Models too large for one GPU are sharded across both.

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  AINode                                              │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Web UI      │  │  REST API │  │  CLI          │  │
│  │  Chat +      │  │  OpenAI-  │  │  ainode start │  │
│  │  Fine-tune   │  │  compat   │  │  ainode train │  │
│  └──────┬──────┘  └────┬─────┘  └───────┬───────┘  │
│         │              │                │            │
│  ┌──────▼──────────────▼────────────────▼───────┐   │
│  │  Orchestrator                                 │   │
│  │  Node discovery · Model routing · Sharding    │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                            │
│  ┌──────────────────────▼───────────────────────┐   │
│  │  vLLM Engine                                  │   │
│  │  PagedAttention · Continuous batching · CUDA  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## Supported Models

AINode works with any model vLLM supports. Some popular choices:

| Model | Memory Needed | Nodes |
|---|---|---|
| Llama 3.2 3B | ~6 GB | 1 |
| Llama 3.1 8B | ~16 GB | 1 |
| Llama 3.1 70B (4-bit) | ~35 GB | 1 |
| Qwen 2.5 72B | ~40 GB | 1 |
| Llama 3.1 405B (4-bit) | ~200 GB | 2 |
| DeepSeek V3 (671B) | ~350 GB | 2+ |

---

## CLI Reference

```bash
ainode start              # Start AINode (inference + web UI)
ainode stop               # Stop AINode
ainode status             # Show cluster status
ainode models             # List available models
ainode train              # Launch fine-tuning UI
ainode join               # Join an existing cluster
ainode config             # Edit configuration
```

---

## API

AINode exposes an OpenAI-compatible API. Use it with any tool that speaks OpenAI:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Works with Open WebUI, LiteLLM, LangChain, and anything else that uses the OpenAI SDK.

---

## Requirements

- **OS:** Ubuntu 22.04+ (including DGX Spark OS)
- **GPU:** Any NVIDIA GPU with CUDA support
- **Memory:** 8 GB+ GPU memory (128 GB recommended for large models)
- **Disk:** 20 GB+ for models
- **Python:** 3.10+

---

## Why AINode?

| | Cloud AI (OpenAI, etc.) | AINode |
|---|---|---|
| Monthly cost | $100-10,000+ | $0 (you own the hardware) |
| Data privacy | Your data on their servers | Your data stays local |
| Rate limits | Yes | No |
| Latency | 200-2000ms | 10-50ms |
| Fine-tuning | Limited, expensive | Unlimited, free |
| Internet required | Yes | No |
| Models available | Their choice | Your choice |

---

## Roadmap

- [x] Core CLI and installer
- [x] vLLM integration
- [x] Web UI (chat)
- [ ] Browser-based fine-tuning
- [ ] Multi-node auto-discovery
- [ ] Automatic model sharding
- [ ] Training job dashboard
- [ ] Model management (download, delete, organize)
- [ ] Prometheus metrics endpoint
- [ ] Mobile-friendly UI

---

## Contributing

AINode is open source and welcomes contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0 — use it however you want.

---

<p align="center">
  <sub>Powered by <a href="https://argentos.ai">argentos.ai</a></sub>
</p>
