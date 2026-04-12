# Runbook: Threadmaster Handoff

## When to Use

Use this runbook whenever:
- An agent wants the Threadmaster to merge or release work
- An agent is ending a session with work that needs continuation
- A change touches the docs repo at /Users/sem/code/mintlify-docs

## The Core Rule

Use commit/file truth, not branch-name truth.

## Required Handoff Packet

```markdown
MERGE REQUEST — <short label>

Repo: getainode/ainode
Linear: <ID or "unavailable">
Author: <agent id>

Source of truth:
- Branch: `<branch>`
- Commit: `<short sha>`
- Tree state: CLEAN | DIRTY

Intended files:
- `path/to/file-a`
- `path/to/file-b`

Explicit exclusions:
- `ops/**`

Validation:
- `pip install -e .` — PASS | FAIL
- `ainode --version` — PASS | FAIL
- `pytest tests/` — PASS | FAIL | NOT RUN
- Manual GPU test — PASS | FAIL | NOT RUN

Public docs:
- Repo: /Users/sem/code/mintlify-docs
- Status: UPDATED | NOT NEEDED | BLOCKED
- Files: `docs/...` or `none`

Risks:
- <risk 1>

Action required from Threadmaster:
1. Reconstruct in clean dev lane
2. Re-run validation
3. Present to operator for sign-off
4. Merge to dev, then promote to main after approval
```
