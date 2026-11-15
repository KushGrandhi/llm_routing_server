# Anchor Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the LLM gateway on the Lightsail server `anchor` behind Caddy at `https://llm.kgai.pro`, with Railway removed as a dependency.

**Architecture:** A `deploy/` folder holds a Docker Compose stack (gateway, Redis, Caddy) modeled on aurora's deploy. The gateway's rate-limit keys become stable across processes, and rate limits and cache use the shared Redis. Gunicorn's worker count is configurable so memory on `anchor` can be controlled.

**Tech Stack:** Python 3.11, Flask 3, gunicorn, litellm, Redis 7, Caddy 2, Docker Compose, pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-anchor-migration-design.md`

## Global Constraints

- Python image and local venv: Python 3.11 (`Dockerfile` uses `python:3.11-slim`).
- Public ports: only 80 and 443, owned by Caddy. The gateway publishes no host port.
- Domain: `llm.kgai.pro`. Caddy serves `llm.$DOMAIN`, with `DOMAIN=kgai.pro`.
- Secrets live only in `deploy/.env` on `anchor`, which is git-ignored. Never committed, never pasted into chat.
- Memory limits: gateway 1g, Redis 128m, Caddy 128m. Gateway runs `GUNICORN_WORKERS=2` by default. Measured before cutover (Task 9).
- Commits go on branch `feat/anchor-migration`, not `main`.

## Review Focus

Five inputs the spec implies, none covered by existing tests. Each has a test in the owning task.

1. **Empty bearer token** (`Authorization: Bearer ` with nothing after): must not share one rate-limit bucket with other empty-token requests. Owner: Task 1.
2. **Two different API keys** must not share one rate-limit bucket. Owner: Task 1.
3. **Non-JSON body** to `/v1/chat/completions`: must return 4xx, not 500. Owner: Task 3.
4. **Missing `messages` field**: must return 4xx, not 500. Owner: Task 3.
5. **Unknown model name**: must return 4xx, not 500. Owner: Task 3.

---

### Task 0: Working environment and test baseline

**Files:**
- Create: `.venv/` (git-ignored, already covered by `.gitignore`)

**Interfaces:**
- Consumes: nothing
- Produces: a working `.venv` used by every later test run

- [ ] **Step 1: Create a git branch**

Run: `cd /Users/kush/Documents/projects/llm_routing_server && git checkout -b feat/anchor-migration`
Expected: `Switched to a new branch 'feat/anchor-migration'`

- [ ] **Step 2: Install Python 3.11 if missing**

Run: `brew install python@3.11`
Expected: `python3.11 --version` prints `Python 3.11.x`

- [ ] **Step 3: Create venv and install requirements**

Run: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`
Expected: install completes with no resolver errors.

- [ ] **Step 4: Record the baseline test result**

Run: `.venv/bin/python -m pytest -q`
Expected: the suite runs. Record the pass and fail counts in the task notes. Any failure present before any change is pre-existing and must be listed, not fixed silently.

- [ ] **Step 5: Commit**

Nothing to commit yet. Continue to Task 1.

---

### Task 1: Stable rate-limit keys

**Files:**
- Modify: `app/middleware/rate_limiter.py:14-25`
- Test: `tests/test_rate_limiter.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `_get_rate_limit_key() -> str`. Returns `apikey:<32 hex chars>` for a non-empty Bearer token (sha256 of the token), otherwise `ip:<remote address>`. Used by `create_rate_limiter()` as `key_func`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rate_limiter.py
import hashlib
import subprocess
import sys
import os
import pytest
from flask import Flask

from app.middleware.rate_limiter import _get_rate_limit_key


def _key_for_header(header_value):
    app = Flask(__name__)
    headers = {} if header_value is None else {"Authorization": header_value}
    with app.test_request_context("/v1/chat/completions", headers=headers):
        return _get_rate_limit_key()


def test_api_key_key_is_sha256_digest():
    expected = "apikey:" + hashlib.sha256(b"abc").hexdigest()[:32]
    assert _key_for_header("Bearer abc") == expected


def test_key_is_stable_across_python_processes():
    # Python's hash() is randomized per process; the key must not be.
    code = (
        "from flask import Flask\n"
        "from app.middleware.rate_limiter import _get_rate_limit_key\n"
        "app = Flask('x')\n"
        "with app.test_request_context('/', headers={'Authorization': 'Bearer abc'}):\n"
        "    print(_get_rate_limit_key())\n"
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs = set()
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=project_root)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


def test_different_api_keys_get_different_keys():
    assert _key_for_header("Bearer key-one") != _key_for_header("Bearer key-two")


def test_empty_bearer_token_falls_back_to_ip():
    assert _key_for_header("Bearer ").startswith("ip:")


def test_missing_header_uses_ip():
    assert _key_for_header(None).startswith("ip:")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rate_limiter.py -v`
Expected: `test_api_key_key_is_sha256_digest`, `test_key_is_stable_across_python_processes`, and `test_empty_bearer_token_falls_back_to_ip` FAIL (the current code uses `hash()`, and an empty token currently produces an `apikey:` key).

- [ ] **Step 3: Implement the stable key**

In `_get_rate_limit_key()` in `app/middleware/rate_limiter.py`: add `import hashlib`. When the Authorization header starts with `"Bearer "` and the token after it is non-empty, return `"apikey:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]`. In every other case, return `f"ip:{get_remote_address()}"`. Update the docstring to match.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rate_limiter.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/middleware/rate_limiter.py tests/test_rate_limiter.py
git commit -m "Use stable sha256 rate-limit keys and fall back to IP for empty tokens"
```

---

### Task 2: Shared Redis for rate limits and cache

**Files:**
- Test: `tests/test_shared_storage.py` (create)
- No code change expected. If a test fails, fix the failing setting path in `app/config/settings.py` or `app/services/llm_router.py:_setup_caching`.

**Interfaces:**
- Consumes: `GatewaySettings` from `app/config/settings.py`, `LLMRouterService._setup_caching()` from `app/services/llm_router.py`.
- Produces: confirmation that `RATE_LIMIT_STORAGE_URI=redis://redis:6379` and `CACHE_TYPE=redis` (with `REDIS_HOST=redis`) are read from the environment and reach the storage and cache constructors.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shared_storage.py
import litellm
import pytest

from app.config.settings import get_gateway_settings, GatewaySettings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_gateway_settings.cache_clear()
    yield
    get_gateway_settings.cache_clear()


def test_rate_limit_storage_uri_comes_from_env(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_STORAGE_URI", "redis://redis:6379")
    assert GatewaySettings().rate_limit_storage_uri == "redis://redis:6379"


def test_redis_cache_is_configured_from_env(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("CACHE_TYPE", "redis")
    monkeypatch.setenv("REDIS_HOST", "redis")
    monkeypatch.setenv("REDIS_PORT", "6379")

    calls = []
    real_cache = litellm.Cache

    def recording_cache(**kwargs):
        calls.append(kwargs)
        return real_cache(type="local")

    monkeypatch.setattr(litellm, "Cache", recording_cache)

    from app.services.llm_router import LLMRouterService
    service = LLMRouterService.__new__(LLMRouterService)
    service.gateway_settings = get_gateway_settings()
    service._setup_caching()

    assert calls and calls[0]["type"] == "redis"
    assert calls[0]["host"] == "redis"
    assert calls[0]["port"] == 6379
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_shared_storage.py -v`
Expected: both PASS if the settings already read the environment. If `test_redis_cache_is_configured_from_env` fails with `calls` empty, `_setup_caching` is not reached with `cache_type="redis"`; fix the call in `_setup_caching` and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_shared_storage.py
git commit -m "Test that rate-limit and cache storage read Redis settings from env"
```

---

### Task 3: Input robustness on chat completions

**Files:**
- Test: `tests/test_input_robustness.py` (create)
- Modify, only if a test fails: `app/routes/chat_routes.py` (route at line 57)

**Interfaces:**
- Consumes: `create_flask_application()` from `app/__init__.py`, the `/v1/chat/completions` route in `chat_routes.py`.
- Produces: `/v1/chat/completions` returns a status below 500 for malformed input.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_input_robustness.py
import pytest

from app import create_flask_application


@pytest.fixture()
def client():
    application = create_flask_application()
    application.config["TESTING"] = True
    return application.test_client()


def test_non_json_body_is_client_error(client):
    response = client.post(
        "/v1/chat/completions",
        data="this is not json",
        content_type="application/json",
    )
    assert 400 <= response.status_code < 500


def test_missing_messages_is_client_error(client):
    response = client.post("/v1/chat/completions", json={"model": "gpt-4o"})
    assert 400 <= response.status_code < 500


def test_unknown_model_is_client_error(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "no-such-model-xyz", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert 400 <= response.status_code < 500
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_input_robustness.py -v`
Expected: PASS if the route already rejects these inputs. Any test that returns 500 is a real bug: read the traceback, add an explicit check in `chat_routes.py` that returns 400 with the same JSON error shape the other error responses use, and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_input_robustness.py app/routes/chat_routes.py
git commit -m "Test and handle malformed chat completion inputs as 4xx"
```

(If `chat_routes.py` was not changed, omit it from `git add`.)

---

### Task 4: Dockerfile, pins, and compose cleanup

**Files:**
- Modify: `Dockerfile:37-42` (the `CMD` block)
- Modify: `requirements.txt:5` (the litellm pin comment only)
- Modify: `docker-compose.yaml:1-3` (remove the `version:` key)

**Interfaces:**
- Consumes: nothing
- Produces: a `Dockerfile` whose gunicorn worker count comes from `GUNICORN_WORKERS` (default 2), and a root `docker-compose.yaml` that validates without warnings.

- [ ] **Step 1: Write the check**

Run: `docker compose -f docker-compose.yaml config > /dev/null`
Expected (currently): a warning that the `version` key is obsolete.

- [ ] **Step 2: Change the Dockerfile CMD to read the worker count**

Replace the exec-form `CMD` with shell form so `GUNICORN_WORKERS` expands, and default it to 2:

```dockerfile
ENV GUNICORN_WORKERS=2
CMD exec gunicorn run:flask_application \
    --bind 0.0.0.0:5000 \
    --workers ${GUNICORN_WORKERS} \
    --threads 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
```

- [ ] **Step 3: Fix the litellm pin comment**

Run `.venv/bin/pip install -r requirements.txt` on Python 3.11 (already done in Task 0). It resolves, so the pin stays. Change only the comment on the litellm line to `# Upper bound checked on Python 3.11; not a 3.9 requirement`. Do not change the version range.

- [ ] **Step 4: Remove the obsolete key**

Delete the `version: '3.8'` line from `docker-compose.yaml`.

- [ ] **Step 5: Verify**

Run: `docker compose -f docker-compose.yaml config > /dev/null && echo OK`
Expected: `OK` with no warnings.
Run: `docker build -t gateway-check . && docker run --rm gateway-check gunicorn --version`
Expected: a gunicorn version line, and the image builds.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile requirements.txt docker-compose.yaml
git commit -m "Make gunicorn worker count configurable and drop obsolete compose key"
```

---

### Task 5: The `deploy/` stack

**Files:**
- Create: `deploy/docker-compose.yml`
- Create: `deploy/Caddyfile`
- Create: `deploy/.env.example`
- Modify: `.gitignore` (add `deploy/.env`)

**Interfaces:**
- Consumes: the `Dockerfile` (Task 4), the env vars in `app/config/settings.py`.
- Produces: a stack where Caddy is the only service with published ports, and the gateway listens on `gateway:5000` inside the compose network.

- [ ] **Step 1: Write the compose file**

```yaml
# deploy/docker-compose.yml
services:
  gateway:
    build: ..
    restart: unless-stopped
    env_file: .env
    environment:
      CACHE_TYPE: redis
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      RATE_LIMIT_STORAGE_URI: redis://redis:6379
      GUNICORN_WORKERS: ${GUNICORN_WORKERS:-2}
    volumes:
      - gateway-data:/app/data
      - ../config:/app/config:ro
    expose:
      - "5000"
    depends_on:
      redis:
        condition: service_healthy
    mem_limit: 1g
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: ["redis-server", "--save", "", "--appendonly", "no", "--maxmemory", "96mb", "--maxmemory-policy", "allkeys-lru"]
    mem_limit: 128m
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  caddy:
    image: caddy:2
    restart: unless-stopped
    depends_on:
      - gateway
    env_file: .env
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data
      - caddy-config:/config
    mem_limit: 128m

volumes:
  gateway-data:
  caddy-data:
  caddy-config:
```

- [ ] **Step 2: Write the Caddyfile**

```
llm.{$DOMAIN} {
	reverse_proxy gateway:5000
}
```

- [ ] **Step 3: Write `deploy/.env.example`**

List every variable from `env.example` at the repo root, with placeholder values only: `GATEWAY_API_KEYS`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `HUGGINGFACE_API_KEY`, `CACHE_ENABLED=true`, `CACHE_DEFAULT_TTL_SECONDS=3600`, `RATE_LIMIT_ENABLED=true`, `RATE_LIMIT_DEFAULT=100 per minute`, `REQUEST_LOGGING_ENABLED=true`, `METRICS_ENABLED=true`, `LOG_LEVEL=INFO`, `DEFAULT_TIMEOUT_SECONDS=60`, `DEFAULT_MAX_RETRIES=2`, `DOMAIN=kgai.pro`, `GUNICORN_WORKERS=2`. Do not include `REDIS_PASSWORD`, `CACHE_TYPE`, `REDIS_HOST`, `REDIS_PORT`, or `RATE_LIMIT_STORAGE_URI`: the compose file sets those.

- [ ] **Step 4: Git-ignore the real env file**

Append `deploy/.env` to `.gitignore`.

- [ ] **Step 5: Verify the stack parses**

Run: `cp deploy/.env.example deploy/.env && docker compose -f deploy/docker-compose.yml config > /dev/null && echo OK; rm deploy/.env`
Expected: `OK`. Then confirm `git status` does not list `deploy/.env`.

- [ ] **Step 6: Commit**

```bash
git add deploy/docker-compose.yml deploy/Caddyfile deploy/.env.example .gitignore
git commit -m "Add deploy stack: gateway, Redis, and Caddy with TLS"
```

---

### Task 6: Smoke test script

**Files:**
- Create: `scripts/smoke.sh`

**Interfaces:**
- Consumes: a base URL argument and an optional `GATEWAY_KEY` env var.
- Produces: exit 0 only when `/health` returns 200, `/v1/models` returns 200 with a non-empty `data` array, and, when `SMOKE_MODEL` is set, one chat completion returns 200.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Usage: scripts/smoke.sh https://llm.kgai.pro
set -euo pipefail
base_url="${1:?usage: smoke.sh <base-url>}"
auth_header=()
if [[ -n "${GATEWAY_KEY:-}" ]]; then auth_header=(-H "Authorization: Bearer ${GATEWAY_KEY}"); fi

health_code=$(curl -s -o /dev/null -w '%{http_code}' "$base_url/health")
[[ "$health_code" == "200" ]] || { echo "FAIL /health $health_code"; exit 1; }

models_body=$(curl -s "${auth_header[@]}" "$base_url/v1/models")
echo "$models_body" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("data") else 1)' \
  || { echo "FAIL /v1/models has no data"; exit 1; }

if [[ -n "${SMOKE_MODEL:-}" ]]; then
  chat_code=$(curl -s -o /dev/null -w '%{http_code}' "${auth_header[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SMOKE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":5}" \
    "$base_url/v1/chat/completions")
  [[ "$chat_code" == "200" ]] || { echo "FAIL chat $chat_code"; exit 1; }
fi
echo "OK"
```

- [ ] **Step 2: Make it executable and run it locally**

Run: `chmod +x scripts/smoke.sh && python run.py & sleep 3 && scripts/smoke.sh http://localhost:5000; kill %1`
Expected: `OK`. Models listing works without a chat call because `SMOKE_MODEL` is unset.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke.sh
git commit -m "Add smoke test script for health, models, and optional chat"
```

---

### Task 7: Local container check

**Files:** none changed. Verification only.

**Interfaces:**
- Consumes: the stack from Task 5 and the smoke script from Task 6.
- Produces: confirmation the full stack starts and passes the smoke test before anything touches `anchor`.

- [ ] **Step 1: Start the stack with a local env file**

Run: `cp deploy/.env.example deploy/.env && sed -i '' 's/DOMAIN=kgai.pro/DOMAIN=localhost/' deploy/.env && docker compose -f deploy/docker-compose.yml up -d --build`
Expected: three containers start. `docker compose -f deploy/docker-compose.yml ps` shows all three healthy or running.

- [ ] **Step 2: Run the smoke test through the gateway container**

Run: `docker compose -f deploy/docker-compose.yml exec gateway python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:5000/health').status)"`
Expected: `200`. (Caddy's TLS needs a real domain, so Step 3 covers the proxy path.)

- [ ] **Step 3: Confirm Caddy proxies to the gateway**

Run: `curl -sk --resolve llm.localhost:443:127.0.0.1 https://llm.localhost/health`
Expected: `{"service": "llm-gateway", "status": "healthy"}` or the health JSON. If Caddy refuses without a real domain, record that as expected and rely on Step 2 plus the Task 9 public check.

- [ ] **Step 4: Check gateway memory and worker count**

Run: `docker stats --no-stream`
Expected: gateway memory well under 1g at idle. Record the figure for Task 9.

- [ ] **Step 5: Tear down**

Run: `docker compose -f deploy/docker-compose.yml down && rm deploy/.env`
Expected: containers stop and the local `deploy/.env` is removed.

No commit: nothing changed.

---

### Task 8: Remove Railway references and update the README

**Files:**
- Modify: `README.md` (add a "Deploy" section; no Railway text exists in the repo, verified with grep)

**Interfaces:**
- Consumes: the `deploy/` stack from Task 5.
- Produces: README instructions for running the stack on a server, with `llm.kgai.pro` as the hosted endpoint.

- [ ] **Step 1: Confirm no Railway references remain outside docs**

Run: `grep -rni railway --exclude-dir=.git --exclude-dir=.venv --exclude-dir=docs .`
Expected: no output. If output appears, remove each reference and re-run.

- [ ] **Step 2: Add the deploy section to the README**

Under "Option 2: Self-Host", add a short block: copy `deploy/.env.example` to `deploy/.env`, set `DOMAIN` and keys, then `docker compose -f deploy/docker-compose.yml up -d --build`. Mention `scripts/smoke.sh https://llm.<your-domain>`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document the deploy stack and smoke test in README"
```

---

### Task 9: Deploy to `anchor` and cut over

This task needs Kush's action for each step marked **[Kush]**. Claude cannot do those.

**Files:** none in the repo. Server state only.

**Interfaces:**
- Consumes: the merged branch, `deploy/.env` on the server, DNS for `llm.kgai.pro`.
- Produces: `https://llm.kgai.pro/health` returning healthy, and Railway shut down.

- [ ] **Step 1: [Kush] Confirm ports on anchor**

Run on `anchor`: `sudo ss -ltnp | grep -E ':(80|443)\s'`
Expected: nothing listening, or only a proxy that Caddy can replace. If something listens, stop and report what it is before continuing.

- [ ] **Step 2: [Kush] Add the DNS record**

At Namecheap, add an A record: host `llm`, value = `anchor`'s public IP. Confirm with `dig +short llm.kgai.pro`.

- [ ] **Step 3: [Kush] Create `deploy/.env` on anchor**

Copy the repo to `anchor` over Tailscale, then on `anchor` create `deploy/.env` from `deploy/.env.example` and fill in real keys. Keys are entered directly on the server, not sent through chat.

- [ ] **Step 4: Start the stack and check memory**

Run on `anchor`: `cd <repo> && docker compose -f deploy/docker-compose.yml up -d --build && docker stats --no-stream`
Expected: all three services running. Gateway memory is under 1g and total headroom is still comfortable alongside the aws-anchor services. If memory is tight, set `GUNICORN_WORKERS=1` in `deploy/.env`, restart, and re-check.

- [ ] **Step 5: Smoke test the public endpoint**

Run from the Mac: `GATEWAY_KEY=<key> SMOKE_MODEL=<a model from config/models.yaml> scripts/smoke.sh https://llm.kgai.pro`
Expected: `OK`. Certificate issuance can take a minute on first start.

- [ ] **Step 6: [Kush] Point clients at the new endpoint and shut down Railway**

Switch any client using the old URL to `https://llm.kgai.pro`. Once a day of normal use is clean, shut down the Railway service in the Railway dashboard. This removes the Railway dependency.

- [ ] **Step 7: Record the outcome**

Write the smoke output, memory figures, and the Railway shutdown date into the PR description. Open the PR from `feat/anchor-migration` into `main` only when Kush asks for it.

---

## Notes for the executor

- `python3.11` is not on this Mac by default. Task 0 installs it with Homebrew.
- Docker must be running for Tasks 4, 5, and 7.
- The Railway shutdown in Task 9 Step 6 is a dashboard action Kush takes. Do not attempt it.
- Keys: never write real provider keys into any file in the repo, and never print them.
