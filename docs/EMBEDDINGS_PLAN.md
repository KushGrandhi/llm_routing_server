# Embeddings upgrade plan

Goal: serve `POST /v1/embeddings` (OpenAI-compatible) through the same gateway,
with the same auth, rate limits, usage log, metrics and model registry as chat,
for any provider, including at least one route that runs on open-weight models.

## Acceptance criterion

1. `POST /v1/embeddings` with `{"model": <name from config>, "input": <str | [str] | token ids>}`
   returns `{"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [...]}], "model", "usage"}`.
2. The official `openai` Python client (`client.embeddings.create(...)` against `base_url`) works unchanged.
3. An embedding model is declared in `config/models.yaml` the same way as a chat model; the
   embeddings code path contains no provider names.
4. A self-hosted OpenAI-compatible server (Ollama, vLLM, TEI, llama.cpp) serving an open model
   (Qwen3-Embedding, BGE-M3, ...) works end to end.
5. Missing/invalid auth gives 401; unknown model, chat model, malformed input give 400.
6. Requests are rate-limited, logged to the usage DB and counted in Prometheus metrics like chat.

## Current architecture (as read from the code)

- `app/__init__.py`: Flask factory. Creates one `LLMRouterService` and one `UsageTrackerService`,
  attaches them to the app, registers blueprints under `/v1`, and calls `init_rate_limiter`.
- `app/middleware/rate_limiter.py`: Flask-Limiter with `default_limits` set from settings and
  `init_app` on the whole app, so every route (current and future) is limited per API key or IP.
- `app/middleware/auth_middleware.py`: `@require_api_key_authentication` checks `Bearer` against
  `GATEWAY_API_KEYS` (open when unset, i.e. dev mode).
- `app/services/llm_router.py`: loads `config/models.yaml` into `available_models`
  (name -> dict). `_resolve_model_identifier` maps a gateway name to a LiteLLM model string
  (`custom_openai` -> `openai/<model_id>` plus `api_base`). `_build_completion_kwargs` adds
  timeout, retries, cache TTL, `api_base`/`api_key`, passthrough params, and strips
  `unsupported_params`. Fallbacks = per-model list + `global_fallbacks`. Cost via
  `litellm.completion_cost`. Hot reload via `POST /v1/models/reload`.
- `app/routes/chat_routes.py`: validates body, resolves the model up front (400 `invalid_model`),
  tracks active requests, calls the router, and logs via `_log_successful_request` /
  `_log_error_request` (usage DB + metrics), keyed by `_get_api_key_hash`.
- `app/routes/models_routes.py`: lists the registry. `admin_routes.py`: usage, metrics, health.
- `app/config/settings.py`: pydantic settings (timeouts, retries, cache, limits, provider keys).
  Provider keys reach LiteLLM through the environment (`OPENAI_API_KEY`, `GEMINI_API_KEY`, ...).
- Pinned LiteLLM is 1.69.3 (`requirements.txt`: `>=1.40,<1.70`).

## Options compared

### Option A: extend the existing LiteLLM router (chosen)

Add `generate_embeddings()` to `LLMRouterService` calling `litellm.embedding`, plus a small
`embeddings_routes.py` blueprint. Models get one new optional config key, `mode: embedding`.

Evidence:
- `litellm.embedding` exists in the pinned version (`litellm/main.py:3355`) with the same
  `model`, `api_base`, `api_key`, `timeout`, `dimensions`, `user` arguments the router already
  builds for chat, and branches for `openai` and OpenAI-compatible `api_base`
  (`main.py:3547`), `gemini` (`main.py:3736`), `ollama` (`main.py:3844`), `voyage`
  (`main.py:3914`), plus cohere, bedrock, vertex, mistral, huggingface and others.
  Provider choice is driven entirely by the `model_id` string from config, exactly as for chat.
- `litellm.completion_cost` handles `EmbeddingResponse` (`litellm/cost_calculator.py:548`), so
  the existing `_calculate_request_cost` works unchanged.
- The cache kwarg is handled by LiteLLM's `@client` wrapper for every call type, so
  per-model `cache_enabled`/`cache_ttl_seconds` apply as-is.
- Registry, hot reload, `/v1/models`, the custom_openai `api_base` handling and the usage/metrics
  helpers in `chat_routes.py` are reused, not copied.
- Caveat found in the source: LiteLLM's `num_retries` retry loop runs only for
  `completion` (`litellm/utils.py:1228`, `if call_type == CallTypes.completion.value`). For
  embeddings the router passes the model's `max_retries` as LiteLLM's `max_retries`, which is
  forwarded to the OpenAI-compatible client (`main.py`, `max_retries = kwargs.get("max_retries")`).

### Option B: separate embeddings module with its own registry and HTTP client

A new `embeddings.yaml` and a service that POSTs directly to `<api_base>/embeddings` with the
`openai` SDK or `requests`.

Evidence against:
- Only OpenAI-shaped APIs would work. Gemini (`batchEmbedContents`), Voyage, Cohere and Bedrock
  have different request shapes, so supporting them means writing per-vendor adapters with
  provider names in the embeddings path, which violates the neutrality requirement and
  duplicates what LiteLLM already does for chat.
- Duplicates config loading, hot reload (`reload_configuration` only re-reads `models.yaml`),
  model listing, timeouts, caching and cost tracking.

### Option C: run the LiteLLM Proxy next to the gateway

Rejected without prototyping: it is a second server and framework with its own auth, DB and
config, which the requirements exclude ("no new frameworks").

**Decision: Option A.** It meets criteria 1-6 with one new route file and a few dozen lines in
the router, keeps provider selection in config, and adds no dependency.

## Design

### Endpoint

`POST /v1/embeddings` (new file `app/routes/embeddings_routes.py`, `@require_api_key_authentication`).

Request (OpenAI shape):

| field | rule |
|---|---|
| `model` | required string, must be a configured model with `mode: embedding` |
| `input` | required: non-empty string, non-empty list of strings, list of ints (token ids), or list of lists of ints; at most 2048 items (OpenAI's limit) |
| `encoding_format` | optional, `"float"` (default) or `"base64"` |
| `dimensions` | optional positive int, passed to the provider |
| `user` | optional string, passed to the provider |

Response:

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.01, ...]}],
 "model": "<gateway name>",
 "usage": {"prompt_tokens": 5, "total_tokens": 5, "cost_usd": 0.0000001},
 "gateway_metadata": {"latency_ms": 42, "provider_model": "...", "cached": false}}
```

`encoding_format`: the gateway always asks the provider for floats and, when the client asked
for `base64`, encodes the floats itself (little-endian float32, which is what OpenAI returns).
This keeps the behaviour identical across providers. The `openai` Python client asks for base64
by default and decodes it, and it also accepts float lists.

Errors: 400 `invalid_request_error` for a bad body, input or param; 400 `invalid_model` for an
unknown model or a chat model; 401 from the auth decorator; a LiteLLM `BadRequestError` from the
provider becomes 400; anything else 500 `internal_error` (same as chat).

### Config

One new optional key per model, `mode`, using LiteLLM's own vocabulary (`chat` | `embedding`,
default `chat`, so existing entries are unchanged). Everything else (`provider`, `model_id`,
`api_base`, `api_key`, `timeout_seconds`, `max_retries`, `cache_enabled`, `cache_ttl_seconds`,
`unsupported_params`, `description`) means the same thing as for chat.

```yaml
  - name: "text-embedding-3-small"          # closed provider
    mode: embedding
    provider: "openai"
    model_id: "text-embedding-3-small"

  - name: "qwen3-embedding"                  # open weights, self-hosted
    mode: embedding
    provider: "custom_openai"
    model_id: "qwen3-embedding:0.6b"
    api_base: "http://localhost:11434/v1"    # Ollama, vLLM, TEI, llama.cpp ...
```

Provider support is whatever LiteLLM supports for embeddings: `openai`, `gemini/...`,
`voyage/...`, `cohere/...`, `mistral/...`, `bedrock/...`, `ollama/...`, and any
OpenAI-compatible server via `custom_openai`. Anthropic and Groq do not sell an embeddings API;
a deployment that only holds those keys uses the open-weight route (or e.g. Voyage).

### Provider selection

Same as chat: the gateway name resolves through `_resolve_model_identifier` to the LiteLLM model
string from config; `custom_openai` (already handled for chat) adds `api_base`/`api_key`. The
resolver gains a `mode` argument, so chat rejects embedding models and embeddings reject chat
models with a clear 400. Chat fallback lists skip embedding models.

No fallbacks for embeddings: vectors from different models live in different spaces, so silently
answering from another model would corrupt a client's index. `fallbacks` and `global_fallbacks`
are ignored for embedding models (documented in the config file).

### Usage, rate limits, metrics

- Rate limiting: automatic (app-wide Flask-Limiter default limits, same key function).
- Auth: same decorator.
- Usage DB and Prometheus: the route reuses `_log_successful_request` / `_log_error_request` and
  `metrics_service.increment/decrement_active_requests` from the chat route. Embedding token
  counts go in `prompt_tokens`, `completion_tokens` is 0, cost comes from `completion_cost`.
  No schema change.
- `/v1/models` gains a `mode` field per model so clients can tell the two kinds apart.

### Code changes (small)

- `app/services/llm_router.py`: `mode` on resolve/list/fallbacks; extract the shared
  timeout/cache/`api_base` kwargs into `_base_call_kwargs` (chat behaviour unchanged);
  add `generate_embeddings()` and `_format_embedding_response()`.
- `app/routes/embeddings_routes.py`: new blueprint (validation, logging, base64).
- `app/__init__.py`: register the blueprint; list the endpoint on `/`.
- `app/routes/chat_routes.py`: nothing beyond the resolver default (`mode="chat"`).
- `config/models.yaml`: example embedding entries (OpenAI, Gemini, Ollama/open-weight).
- `README.md`: usage with the `openai` client.

## Test plan

New `tests/test_embeddings.py`, same style as the existing tests (Flask test client, pytest,
`unittest.mock.patch` on `litellm`), no real provider calls:

1. Validation: missing body, non-object body, missing/empty `input`, wrong `input` types,
   too many items, bad `encoding_format`, bad `dimensions`, missing/non-string `model` -> 400.
2. Unknown model -> 400; chat model on `/v1/embeddings` -> 400; embedding model on
   `/v1/chat/completions` -> 400.
3. Routing: with `litellm.embedding` mocked, a closed-provider model is called with the
   `model_id` from config, timeout and `dimensions`; response shape and usage are correct.
4. Open-weight route, no mocks: a local fake OpenAI-compatible server in a thread serves
   `/v1/embeddings`; a `custom_openai` model pointing at it goes through the real LiteLLM path.
5. The real `openai` client (a LiteLLM dependency, already installed) against the gateway served
   on a local port, through to the fake server, with default (base64) and float encoding.
6. Auth: with `GATEWAY_API_KEYS` set, no key / wrong key -> 401, right key -> 200.
7. Usage is logged with the embedding token count.
8. Existing suite still passes.
