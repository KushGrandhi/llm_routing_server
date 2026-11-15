#!/usr/bin/env bash
# Usage: scripts/smoke.sh https://llm.kgai.pro
# Optional env: GATEWAY_KEY (Bearer token), SMOKE_MODEL (runs one real chat call)
set -euo pipefail
base_url="${1:?usage: smoke.sh <base-url>}"
auth_header=()
if [[ -n "${GATEWAY_KEY:-}" ]]; then auth_header=(-H "Authorization: Bearer ${GATEWAY_KEY}"); fi

health_code=$(curl -s -o /dev/null -w '%{http_code}' "$base_url/health")
[[ "$health_code" == "200" ]] || { echo "FAIL /health $health_code"; exit 1; }

models_body=$(curl -s "${auth_header[@]+"${auth_header[@]}"}" "$base_url/v1/models")
echo "$models_body" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("data") else 1)' \
  || { echo "FAIL /v1/models has no data"; exit 1; }

if [[ -n "${SMOKE_MODEL:-}" ]]; then
  chat_code=$(curl -s -o /dev/null -w '%{http_code}' "${auth_header[@]+"${auth_header[@]}"}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SMOKE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":5}" \
    "$base_url/v1/chat/completions")
  [[ "$chat_code" == "200" ]] || { echo "FAIL chat $chat_code"; exit 1; }
fi
echo "OK"
