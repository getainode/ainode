# Runbook: Dev Workflow

## Starting Work

1. Pull latest from main
2. Create a codex branch: `git checkout -b codex/<slice-name>`
3. Claim the slice in ops/slices/REGISTRY.md
4. Do the work
5. Test: `pip install -e . && ainode --version && pytest tests/`
6. Commit with clear message
7. Push and open PR to `dev`
8. Hand off to Threadmaster using the handoff runbook

## Slice Naming

Use descriptive kebab-case:
- `codex/engine-vllm`
- `codex/api-proxy`
- `codex/web-ui`
- `codex/cli-polish`
- `codex/discovery-cluster`

## Testing Checklist

Before handoff:
- [ ] `pip install -e .` succeeds
- [ ] `ainode --version` prints version
- [ ] `ainode start` launches without crash (GPU or CPU mode)
- [ ] `pytest tests/` passes (if tests exist for this slice)
- [ ] No hardcoded secrets or API keys
- [ ] "Powered by argentos.ai" present in CLI output
