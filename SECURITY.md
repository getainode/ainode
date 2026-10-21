# Security Policy

## Reporting a vulnerability

Mail **hello@argentos.ai** with "AINode security" in the subject. Do not open a
GitHub issue, a pull request or a Discussion for a vulnerability: those are public
the moment you press the button, and AINode runs on hardware people keep their own
data on.

Tell us what you can of:

- What the issue is, and what an attacker gets out of it.
- The AINode version (`ainode --version`, or the tag in `~/.ainode/image.env`).
- How you reached it: solo node or cluster, whether API-key auth was on, which
  ports were exposed and to what network.
- Steps to reproduce, and a proof of concept if you have one.

We will confirm we received it, tell you what we found, and say when a fix is
released. If the report stands, the release notes describe the issue and credit you
by whatever name you ask for, or nobody if you prefer.

There is **no bug bounty**. This is an Apache-2.0 project with no security budget,
so we have nothing to pay and would rather say so than imply otherwise.

## Supported versions

The latest minor release line, and in practice the newest release on it. AINode
ships often and the upgrade is one command (`ainode update`), so a fix lands in a
new release rather than a patch to an older one.

| Version | Supported |
|---|---|
| Latest 0.5.x release | Yes |
| Anything older | No, upgrade first |

If you are not on the newest release, run `ainode update` before reporting: the
thing you found may already be fixed, and the answer to a report against an old
release is going to be "upgrade" either way.

## What is in scope

The code in this repository: the AINode orchestrator container, its API and web UI,
the installer (`scripts/install.sh`) and the systemd unit it writes, the host
wrapper at `/usr/local/bin/ainode`, and how AINode launches and talks to engine and
job containers.

Out of scope, because they are somebody else's code even though AINode runs them:
vLLM and the engine images (report those upstream), model weights and what a model
says, Docker and the NVIDIA Container Toolkit, and the operating system.

## Things that are deliberate, not vulnerabilities

Report them anyway if you think we have the trade wrong, but know that they are
decisions and are documented:

- **A node listens on `0.0.0.0`; a fresh installer run enables API-key auth by
  default.** When neither `config.json` nor `auth.json` exists, the installer mints
  one key, stores its SHA-256 hash and prints the key once. `AINODE_AUTH=off` opts a
  fresh install into open access; an update preserves the installed access control.
  If no supported SHA-256 tool is available, the installer warns and leaves auth off.
  Running the application directly without an auth store also starts with auth off.
  Open inference access is a choice for a network you control, not a hostile one.
- **People sign in; programs present API keys.** Create the first dashboard admin
  on the node with `ainode auth user add <name> --admin`. The dashboard reads
  `/api/auth/me` before showing the sign-in screen or shell; API-key entry remains
  available in **Config, API access**. With auth enabled, the middleware requires
  a login session, an API key or the fleet credential, except for the static shell,
  `/api/health`, `/api/auth/status`, `/api/auth/login`, `/api/auth/me`,
  `/api/cluster/endpoint` and `POST /api/cluster/join` (which requires its join
  token). Browser preflight requests are keyless. Scoped keys retain their route
  restrictions even on otherwise open routes. The browser onboarding wizard and
  its `/api/onboarding/*` routes have been removed.
  Cookie-authenticated writes require `X-AINode-Client: dashboard`; Bearer keys do
  not. Account, key and auth-switch administration requires an admin session, an
  operator key or the fleet key; a member session cannot administer them. In open
  mode, a caller without credentials may administer only over a loopback transport,
  with that client header for writes. Host-control fields also require admin or
  operator authority with auth on, or loopback with the client header for writes
  when auth is off; fleet forwarding separately checks the original caller's
  authority. Account replication accepts the fleet key alone.
- **`trust_remote_code` is a host control, including on a node running open.**
  Setting it means a later load executes the model repository's own `modeling_*.py` inside the engine
  container, which mounts the host HF cache and the host SSH directory, so
  `PATCH /api/config` requires the host-control authority above to set it. A curated
  catalog entry whose recipe already declares it is the other way in, which is how the models that
  genuinely need it stay one click.
- **The engine port (8000, and 8001 upward for stacked models) is the vLLM container
  itself**, with no AINode routing, no auth and no rate limit in front of it.
- **A distributed head SSHes into its peers** with the install user's key, and starts
  containers there. That is how a multi-node launch works.
- **Training, quantization and adapter merge run model-supplied code in a spawned GPU
  container**, and a catalog recipe can set `trust_remote_code`. Loading a model is
  running its code.
- **The bench harness and the agentic rubric execute code a model wrote**, in a
  temporary directory with a timeout, on the machine driving the run.

## Secrets

Credentials live in `~/.ainode/secrets.json`, mode 0600, obfuscated at rest rather
than encrypted: a root user or anyone with that file can read them. Treat the file
as the secret. A Hugging Face token travels to a job container in a 0600 env file
that is deleted when the container exits, and every logged command line is scrubbed,
so if you find a token in a log, a job record or an API response, that IS a bug and
we want to hear about it.
