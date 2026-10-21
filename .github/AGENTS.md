# Public automation edit contract

This source snapshot intentionally has no GitHub Actions workflows. Keep Actions disabled.
Run checks locally using the root edit contract. Workflow-contract tests accept a missing
named publication workflow only when ALL `.github/workflows/*.yml` and `*.yaml` files are
absent. Retain the original checks when the relevant workflow exists and counterexamples
that reject a differently named workflow. Report the separate known suite failures honestly.

Any future workflow needs separate review and authorization. Jobs may run only on Blacksmith;
public own-hardware and GitHub-hosted runners are prohibited. Do not add private runner,
broker, operator configuration or credentials. Publication jobs must name the protected
`release` environment, use tag pushes or manual dispatch, and support `push=false` for
build-only runs. Site synchronization is an external operation, not a credentialed workflow.
