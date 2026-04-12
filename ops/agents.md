# Agent Registry — AINode Product

## Threadmaster

- **Role:** Claude Opus 4.6 (main exo window) — owns the product build
- **Session:** Main Claude Code session in /Users/sem/code/ainode
- **Responsibilities:** Slice planning, implementation, PR management, handoffs to operator

## Agent Teams

Created by Threadmaster as needed. Current team:

### Core Team
| Agent | Focus | Session |
|-------|-------|---------|
| Threadmaster | Engine, API, CLI — P0 slices | Main session |

### Future Teams (when P0 is done)
| Agent | Focus | Session |
|-------|-------|---------|
| Web Agent | Chat UI, onboarding UI | `tmux: ainode-webui` |
| Training Agent | Fine-tuning backend + UI | `tmux: ainode-training` |
| QA Agent | Testing on real DGX Spark hardware | `tmux: ainode-qa` |

## Agent Rules

1. Each agent owns one slice at a time
2. Work on `codex/*` branches only
3. Hand off via structured packet (see runbooks/threadmaster-handoff.md)
4. Never push to main or dev directly
5. Always run validation before handoff
6. Report blockers to Threadmaster immediately
