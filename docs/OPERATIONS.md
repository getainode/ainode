# Operating a local AINode

Use the installed host wrapper for service operations. Inspect `scripts/install.sh` and
`scripts/uninstall.sh` before changing an installation. A source checkout and an existing
container image are distinct artifacts; confirm the image version with `ainode doctor`
and the keyless `/api/health` response.

- Restrict dashboard, API, engine and discovery access to intended networks. Engine ports
  bypass AINode authentication. Distributed launches need peer SSH and compatible fabric.
- Keep fresh-install authentication enabled. Create the first admin on the node with
  `ainode auth user add <name> --admin`. Use scoped keys for programs, revoke unused keys,
  and remember account replication does not replicate browser sessions.
- Enable TLS before sending passwords across an untrusted network. `ainode tls enable`
  configures certificates; `ainode tls enable --tailscale` obtains a host certificate.
  Check `ainode tls status`; enabling TLS leaves HTTP available, so enforce network policy.
- Keep credentials outside the checkout. Treat the whole AINode home as sensitive:
  config, auth, account and secrets files can carry credentials or credential material.
  Never upload a raw support bundle, shell environment or command transcript publicly.
- Review model code, image recipes, mounts and datasets before launch or training.
  Host controls require operator authority. Extra mount roots are local operator policy.
- Set client rate limits through the dashboard/API for exposed inference use. Inspect
  `ainode/ratelimit/` for the current schema; management routes are outside that limiter.
- Monitor `ainode doctor`, service logs, `/api/health` and authenticated `/metrics`.
  Missing telemetry is not a measured zero. Scrub logs before sharing them.
- Back up the AINode home and any separately configured model/dataset stores with access
  restrictions that preserve private file permissions. Stop writes while taking a consistent
  backup. Protect backups as credentials; they are not public export artifacts.
- Restore into a stopped test installation first, preserving paths and permissions, then
  start it and verify health, authentication, model inventory and a sample inference request.
  Keep a known-working backup and image pin until the replacement is verified.
- `ainode update` pulls and verifies an installed image. To roll back, use an intentionally
  retained compatible version with `ainode update <version>`, then check health and doctor.
  Review config compatibility first; a code rollback does not undo data or credential changes.

Public CI and publication workflows are absent in this source snapshot. Local test results
are not an approval to publish an image or expose an installation publicly. Report suspected
vulnerabilities using `SECURITY.md`.
