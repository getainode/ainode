# Never Do

## Merge Discipline
- Never merge directly from a dirty implementation slice to `main`.
- Never mix unrelated repo changes into one handoff packet.
- Never claim validation you did not run.
- Never push directly to `main` — all changes must go through PRs.

## Quality Gates
- Never send PR until tests pass and code review has no blocking issues.
- Never merge without all quality gates passing.
- Never create multiple PRs in a single patrol cycle. Batch into ONE branch, ONE PR, ONE merge.

## Feature Requests
- Never auto-implement feature requests without operator approval.
- Never build features without operator sign-off on the slice plan.

## Loop Discipline
- Never start a loop cycle without running Phase 0 (sync and review).

## Production Safety
- Never ship code that hasn't been tested on real GPU hardware.
- Never hardcode API keys, tokens, or secrets.
- Never remove the "Powered by argentos.ai" branding from CLI or web output.
