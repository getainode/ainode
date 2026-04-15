# Runbook: Release Flow

How AINode is packaged and distributed, and the step-by-step procedure for
shipping any kind of change. Written after the v0.4.0 container-native
cutover; anchor this when cutting a new release.

---

## 1. Distribution model — what lives where

AINode has **three distinct distribution surfaces**, each with its own
deploy mechanism and blast radius. Confusing them is the #1 way to break
things, so keep this chart straight.

| Surface | Git source | Artifact | Cache TTL | Who pulls it | Trigger |
|---|---|---|---|---|---|
| **Marketing site** | `getainode/ainode.dev` | Next.js site on Railway | CDN 5-min | Anyone visiting ainode.dev | `git push main` → Railway GitHub App (sometimes needs a nudge) |
| **Installer script** | `getainode/ainode` (`scripts/install.sh`) | Raw file on GitHub | none | New installers via `curl | bash` | `git push main` — anyone curling after push gets new version |
| **Runtime image** | `getainode/ainode` (full repo) | `ghcr.io/getainode/ainode:<tag>` | frozen per tag | Every running node on `docker pull` | Explicit build + push (§3) |

The container is the *only* surface that serves running users. Changing
`install.sh` or the marketing page does nothing for someone already
installed — they keep running whatever image tag their systemd unit
references.

### What's baked into the container

- All Python code (`ainode/` package, including `web/static/*`)
- vLLM + Ray + NCCL (via eugr base)
- eugr `launch-cluster.sh` for distributed mode
- Docker CLI (for cross-node orchestration)
- OpenSSH client (for peer SSH in distributed mode)

### What is *not* in the container (lives on host)

- `~/.ainode/config.json` — node config, peer IPs, model choice
- `~/.ainode/models/` — downloaded model weights
- `~/.ainode/logs/` — vLLM + distributed logs
- `~/.ainode/datasets/`, `~/.ainode/training/` — fine-tuning artifacts
- `/usr/local/bin/ainode` — host wrapper installed by `install.sh`
- `/etc/systemd/system/ainode.service` — unit file

This separation is why `ainode update` is safe: pulling a new image never
touches user data.

---

## 2. Decision tree — which flow do I need?

```
What did I change?
│
├── install.sh only                      → §4 (push, done)
├── marketing site                       → §5 (push, verify Railway)
├── README / docs                        → §4 (push, done — README is
│                                               rendered from GitHub)
├── Python code (ainode/api, engine,...) → §3 (rebuild + push image)
├── Web UI (ainode/web/static/*)         → §3 (rebuild + push image)
├── vLLM / Ray / base OS                 → §6 (rebuild BASE too, then §3)
└── Adding a new release (version bump)  → §3 + create GitHub release
```

---

## 3. Release flow — Python or Web UI change

This is the common case. Almost every "ship a fix" goes through here.

### 3a. Local work

```bash
# 1. Branch, edit, test
git checkout -b codex/my-fix
# ... edits in ainode/web/static/js/app.js, ainode/api/, etc. ...
pip install -e . && pytest tests/
git commit -am "fix(ui): ..." && git push origin codex/my-fix
# → open PR → merge to main
```

### 3b. Version bump

For any user-visible change, bump the patch version. Two files:

```bash
# pyproject.toml
version = "0.4.1"

# ainode/service/systemd.py
AINODE_IMAGE_TAG = "0.4.1"
```

Commit: `chore: bump to 0.4.1`. Push to main.

> The `AINODE_IMAGE_TAG` constant is what fresh installs pin to. Keeping
> it in lockstep with `pyproject.toml` is the invariant.

### 3c. Build on Spark 1 (aarch64 host)

The image has to be built on aarch64 because the base image is aarch64.
There is no x86 build. Spark 1 is the canonical build host.

```bash
ssh sem@spark1
cd ~/ainode
git pull
docker build \
  -f scripts/Dockerfile.ainode \
  --build-arg BASE_IMAGE=ghcr.io/getainode/ainode-base:c026c92 \
  -t ghcr.io/getainode/ainode:0.4.1 \
  -t ghcr.io/getainode/ainode:latest \
  .
```

Smoke test before pushing:

```bash
docker run --rm ghcr.io/getainode/ainode:0.4.1 ainode --version
# → ainode 0.4.1
```

### 3d. Push to GHCR

```bash
# Login once per session — token from `gh auth token` on your laptop,
# piped over SSH so it's never written to disk permanently.
# (From your laptop:)
gh auth token | ssh sem@spark1 \
  "docker login ghcr.io -u webdevtodayjason --password-stdin"

# Then on Spark 1:
docker push ghcr.io/getainode/ainode:0.4.1
docker push ghcr.io/getainode/ainode:latest
```

Verify the new tag is public:

```bash
docker logout ghcr.io
docker manifest inspect ghcr.io/getainode/ainode:0.4.1   # should 200
```

If it 401s: new tags inherit the package's visibility setting, so as
long as the package itself is public, new tags are too. If the package
ever flips private, re-flip via
https://github.com/orgs/getainode/packages/container/ainode/settings.

### 3e. Upgrade existing nodes

```bash
# On any node:
ainode update
```

That does `docker pull ghcr.io/getainode/ainode:latest` + restart. The
wrapper was installed by `install.sh` at install time.

### 3f. Tag + release on GitHub

```bash
# On your laptop:
git tag v0.4.1 && git push origin v0.4.1
gh release create v0.4.1 --generate-notes
```

Marketing site v-badge + install.sh auto-pick up the new version.

---

## 4. install.sh / README / docs change

Zero-ceremony. The script lives on `main`; the ainode.dev/install
redirect serves raw-from-main.

```bash
# Edit, commit, push
git commit -am "install: ..." && git push origin main
# That's it. Next curl gets the new script.
```

README renders live on GitHub. No deploy step.

---

## 5. Marketing site (ainode.dev)

Repo: `getainode/ainode.dev`. Hosted on Railway, auto-deployed via the
Railway GitHub App.

```bash
cd ~/code/ainode.dev
# edits…
git commit -am "feat: ..." && git push origin main
```

Railway should rebuild within ~60 seconds. If it doesn't:

- Touch any file and re-push (webhook occasionally misses)
- Or log in to Railway dashboard and hit "Redeploy"

Cache TTL is 5 min on Next.js ISR. Do a cache-busted curl to verify:

```bash
curl -s "https://ainode.dev/?v=$(date +%s)" | grep 'v0.4'
```

---

## 6. Base image rebuild (eugr vLLM)

Rare. Only when bumping vLLM, Ray, CUDA, or NCCL.

1. Bump the pinned eugr commit in `scripts/build-base-image.sh`
2. Run on Spark 1:
   ```bash
   scripts/build-base-image.sh
   docker push ghcr.io/getainode/ainode-base:<new-sha>
   ```
3. Update `BASE_IMAGE` build-arg in the §3c build command
4. Update `scripts/Dockerfile.ainode` default FROM line
5. Then follow §3 for a new AINode tag

Expect ~12 minutes of base build time. Cached layers short-circuit on
subsequent runs.

---

## 7. Rollback

Worst case: a release breaks production. Users run:

```bash
AINODE_IMAGE=ghcr.io/getainode/ainode:0.4.0 ainode update
```

That pulls the previous tag and restarts. To make it stick across
reboots, edit the systemd unit:

```bash
sudo sed -i 's|ainode:latest|ainode:0.4.0|' /etc/systemd/system/ainode.service
sudo systemctl daemon-reload
sudo systemctl restart ainode
```

Or re-run `install.sh` with `AINODE_VERSION=0.4.0 curl ... | bash`.

---

## 8. Dev-loop shortcuts (not for release, just iteration)

For fast UI iteration without rebuilding:

```bash
# Mount the host source into the running container
docker run --rm -it --network=host --gpus all \
  -v $(pwd)/ainode:/app/ainode \
  -v ~/.ainode:/root/.ainode \
  ghcr.io/getainode/ainode:latest \
  ainode start --in-container
```

Edits to `ainode/web/static/*` reflect on browser refresh. Python
changes need the container restart. Do *not* ship this way — always
bake changes into the image per §3 before releasing.

---

## 9. Quick reference — command cheat sheet

| Task | Command |
|---|---|
| Ship a UI fix | edits → push → Spark 1 `docker build && docker push` → bump `AINODE_IMAGE_TAG` |
| User upgrades | `ainode update` |
| User rollbacks | `AINODE_IMAGE=ghcr.io/getainode/ainode:0.4.0 ainode update` |
| Verify GHCR public | `docker logout ghcr.io && docker manifest inspect ghcr.io/getainode/ainode:latest` |
| Bust ainode.dev cache | `curl -s "https://ainode.dev/?v=$(date +%s)"` |
| Force Railway redeploy | Touch any file, push, or hit dashboard redeploy |

---

## 10. Known gotchas

- **Never build the image on your Mac.** Apple Silicon is aarch64 but
  CUDA + NCCL won't install — the base image only works on Linux
  aarch64. Build on Spark 1.
- **`ainode update` needs the wrapper.** If a user installed before
  v0.4.1, they don't have `/usr/local/bin/ainode`. They need to
  re-run `install.sh` once, then `ainode update` works forever after.
- **GHCR package visibility is org-scoped.** If the `getainode` org
  policy ever flips "public packages" off, every pushed tag goes
  private immediately. The setting lives at
  https://github.com/organizations/getainode/settings/packages.
- **Marketing site CDN is Fastly on Railway.** `s-maxage` is long.
  Use cache-busted curl to verify instead of browser.
