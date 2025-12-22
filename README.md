# LLM Gateway Server

A unified REST API gateway that routes requests to multiple LLM providers (OpenAI, Claude, Gemini, Groq, HuggingFace, and custom self-hosted models).

## Features

- **Multi-Provider Support**: OpenAI, Anthropic Claude, Google Gemini, Groq, HuggingFace, and custom OpenAI-compatible endpoints
- **Hot-Reload Config**: Add/remove models without restarting the server
- **Streaming Support**: Real-time token streaming via SSE
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API calls
- **API Key Auth**: Simple Bearer token authentication

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
python run.py
```

## Configuration

### Environment Variables (.env)

```bash
# Gateway API key (for your apps to authenticate)
GATEWAY_API_KEYS=your-secret-key

# Provider API keys (add the ones you need)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
GROQ_API_KEY=gsk_...
HUGGINGFACE_API_KEY=hf_...
```

### Model Registry (config/models.yaml)

Edit this file to add/remove models. Use `POST /v1/models/reload` to apply changes without restart.

```yaml
models:
  - name: "my-model"        # Name your apps will use
    provider: "openai"      # Provider type
    model_id: "gpt-4o"      # Actual model identifier

  # Custom self-hosted (OpenAI-compatible)
  - name: "local-llama"
    provider: "custom_openai"
    model_id: "llama-2"
    api_base: "http://localhost:8080/v1"

default_model: "gemini-flash"
```

## API Endpoints

### Chat Completions

```bash
POST /v1/chat/completions
```

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Authorization: Bearer your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### List Models

```bash
GET /v1/models
```

### Reload Config

```bash
POST /v1/models/reload
```

### Health Check

```bash
GET /health
```

## Supported Providers

| Provider | Config Key | Free Tier |
|----------|------------|-----------|
| OpenAI | `openai` | No |
| Anthropic | `anthropic` | No |
| Google Gemini | `gemini` | Yes (15 req/min) |
| Groq | `groq` | Yes (limited) |
| HuggingFace | `huggingface` | Yes (rate limited) |
| Custom/Self-hosted | `custom_openai` | Your infra |

## Production Deployment

```bash
gunicorn run:flask_application -w 4 -b 0.0.0.0:5000
```

## Adding Self-Hosted Models

For models hosted via vLLM, Ollama, or text-generation-inference:

```yaml
- name: "my-local-model"
  provider: "custom_openai"
  model_id: "model-name"
  api_base: "http://your-server:8000/v1"
```

