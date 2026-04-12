# Branching Rules

## Branch Structure

- `main` — Production. Never push directly.
- `dev` — Integration lane. PRs from codex/* merge here first.
- `codex/*` — Implementation branches. All work happens here.

## Flow

```
codex/<slice>  →  PR  →  dev  →  validate  →  main
```

## Rules

- All changes go through PRs. No direct pushes to main or dev.
- One slice per codex branch. Don't mix unrelated work.
- Branch names: `codex/<slice-name>` (e.g., `codex/engine-vllm`, `codex/web-ui`)
- Delete branches after merge.
