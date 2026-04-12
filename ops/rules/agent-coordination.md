# Agent Coordination Rules

## Threadmaster Role

The Threadmaster (Claude Opus 4.6 in the main exo window) owns the AINode product build. Responsibilities:
- Plan and prioritize slices
- Create agent teams and assign work
- Review handoff packets
- Coordinate merges to dev branch
- Report status to operator (Jason)

## Agent Teams

Agents work inside tmux sessions. Each agent:
- Works on a single `codex/*` branch
- Owns one slice at a time
- Hands off via the threadmaster-handoff runbook
- Never pushes directly to main or dev

## Communication

- Agents report to Threadmaster
- Threadmaster reports to Operator
- All handoffs use structured packets (see runbooks/threadmaster-handoff.md)
- Status updates go in ops/log/

## Tmux Convention

```bash
# Threadmaster session
tmux new-session -s ainode-product

# Agent sessions (created by Threadmaster as needed)
tmux new-session -s ainode-engine
tmux new-session -s ainode-api
tmux new-session -s ainode-webui
```
