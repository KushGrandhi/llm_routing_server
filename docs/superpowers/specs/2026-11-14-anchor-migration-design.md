# Anchor migration: design

Date: 2026-11-14
Status: approved by Kush

## Intent

Move the LLM gateway off Railway and onto Kush's Lightsail server `anchor`, next to aurora and the other services already there. Remove the Railway dependency completely. Keep the public hosted endpoint working at `https://llm.kgai.pro`.

## Goals

- The gateway runs on `anchor` under Docker Compose, with TLS through Caddy, at `llm.kgai.pro`.
- The gateway serves the same API as today: `/v1/chat/completions`, `/v1/models`, `/health`, `/v1/usage`, `/metrics`.
- Nothing in the repo or its docs refers to Railway as a deploy target.
- Multi-worker correctness: rate limits and cache are shared across gunicorn workers, not per process.

## Non-goals

- No rewrite of the gateway or its routing logic.
- No migration of usage data. Railway had none, so the new SQLite database starts empty.
- No change to the aws-anchor stack (LiteLLM proxy, Open WebUI, Postgres) or to aurora. They keep running as they are.
- No new public endpoints beyond `llm.kgai.pro`.

## Anti-goals

- The gateway must not become unreachable if one of the other services on `anchor` restarts or fails.
- Provider API keys must never be committed or written into the image.

## Constraints

- `anchor` is a Lightsail instance with 2 vCPU and 1.9 GB RAM, shared with other services.
- Ports 80 and 443 are the only public ports. Caddy must own them, so anything else on `anchor` that listens on 80 or 443 needs to be checked before cutover.
- The domain `kgai.pro` is managed at Namecheap. DNS for `llm.kgai.pro` is an A record to the `anchor` public IP.

## Approach (chosen)

Docker Compose on `anchor`, in a `deploy/` folder that mirrors aurora's layout (`deploy/docker-compose.yml`, `deploy/Caddyfile`, `deploy/.env.example`, `deploy/README.md`).

Services:
- `gateway`: built from the repo's `Dockerfile`, gunicorn with workers and threads, no host port published (reachable only inside the compose network).
- `redis`: shared cache and rate-limit storage for all gunicorn workers.
- `caddy`: the only service that publishes ports, 80 and 443, with automatic TLS for `llm.kgai.pro` and reverse proxy to `gateway:5000`.

Volumes: `gateway-data` for the SQLite usage database, `redis-data`, and Caddy's certificate volume.

Rejected alternatives:
- A bare process under systemd: no isolation, harder to rebuild, and it would not match aurora's pattern.
- Tailnet-only: breaks the public `llm.kgai.pro` endpoint in the README.

## Required repo changes

1. **Stable rate-limit keys.** `app/middleware/rate_limiter.py` uses Python's `hash()` for API keys, which is randomized per process. Replace it with a stable digest (for example `hashlib.sha256`).
2. **Shared storage.** Rate-limit storage and the response cache use Redis when `RATE_LIMIT_STORAGE_URI` and `CACHE_TYPE` point at it. The compose file sets both. Confirm the code paths actually read these settings.
3. **Pins.** Remove the obsolete `version:` key from the compose file. Resolve the litellm "Python 3.9" pin against the 3.11 image in the Dockerfile.
4. **Railway removal.** Remove Railway mentions from `README.md` and any other doc.
5. **Tests.** The suite must pass before and after. Add a test for the stable key digest.

## Settings

Secrets live only in `deploy/.env` on `anchor`, which is git-ignored. The file holds `GATEWAY_API_KEYS`, the provider keys, `DOMAIN`, and any other values. `deploy/.env.example` lists every variable with a placeholder.

## Resource limits

Each service gets a memory limit in compose so the gateway cannot starve the other services on `anchor`. Limits are checked against actual headroom on `anchor` before cutover.

## Cutover

1. Confirm ports 80 and 443 on `anchor` are free, or that the existing reverse proxy can be moved behind Caddy.
2. Bring the stack up on `anchor` with `llm.kgai.pro` DNS pointing at it, and run a smoke test.
3. Switch clients to `llm.kgai.pro`.
4. Shut down the Railway service. Remove it only after step 3 is confirmed.

## Calls I made (not agreed with Kush)

- Container names prefixed `gateway-`, to avoid clashing with the aws-anchor containers.
- Caddy is per-stack, not shared. If aws-anchor already has a Caddy, it is reused and the Caddyfile is merged instead.
- Smoke test is a `scripts/smoke.sh` script that hits `/health`, `/v1/models`, and one chat completion with a test model.

## Left to the builder

- Exact memory limits and worker counts, measured on `anchor`.
- Whether the aws-anchor stack already has a reverse proxy on 80 and 443.
