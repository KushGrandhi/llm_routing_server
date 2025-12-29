# LLM Gateway

A unified API for accessing multiple LLM providers (OpenAI, Claude, Gemini, Groq, and more) through a single, OpenAI-compatible interface.

## 🚀 Two Ways to Use

### Option 1: Use Our Hosted API

Get instant access without any setup:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://llm.kgai.pro/v1",  # Our hosted endpoint
    api_key="your-api-key"                     # Get yours at llmgateway.io
)

response = client.chat.completions.create(
    model="gpt-4o",  # or "claude-sonnet", "gemini-flash", etc.
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

**Benefits:**
- ✅ No infrastructure to manage
- ✅ Access to all major LLM providers
- ✅ Pay-as-you-go pricing
- ✅ Built-in caching saves you money
- ✅ Automatic fallbacks for reliability

[**Get Your API Key →**](https://kgai.pro)

---

### Option 2: Self-Host

Run your own gateway for full control:

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-gateway.git
cd llm-gateway

# Configure your provider API keys
cp env.example .env
# Edit .env with your OpenAI, Anthropic, etc. keys

# Run with Docker
docker-compose up -d

# Or run locally
pip install -r requirements.txt
python run.py
```

Your gateway is now running at `http://localhost:5000`

---

## 📖 Quick Start

### Basic Usage

```python
from openai import OpenAI

# Point to your gateway (hosted or self-hosted)
client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="your-gateway-key"
)

# Use any supported model
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:5000/v1',
  apiKey: 'your-gateway-key'
});

const response = await client.chat.completions.create({
  model: 'gemini-flash',
  messages: [{ role: 'user', content: 'Hello!' }]
});

console.log(response.choices[0].message.content);
```

### cURL

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Authorization: Bearer your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 🤖 Available Models

| Model | Provider | Best For |
|-------|----------|----------|
| `gpt-4o` | OpenAI | General purpose, high quality |
| `gpt-4o-mini` | OpenAI | Fast, cost-effective |
| `claude-sonnet` | Anthropic | Long context, coding |
| `claude-haiku` | Anthropic | Fast, lightweight |
| `gemini-flash` | Google | Very fast, free tier |
| `gemini-pro` | Google | High quality, long context |
| `llama-groq` | Groq | Ultra-fast inference |
| `reliable-gpt` | Multi | GPT-4o with auto-fallbacks |
| `reliable-fast` | Multi | Fast model with fallbacks |

---

## ✨ Features

### Unified API
One interface for all providers. Switch models by changing a single string.

### Automatic Fallbacks
Configure backup models. If GPT-4o fails, automatically try Claude, then Gemini.

### Response Caching
Identical prompts return cached responses instantly—saves time and money.

### Cost Tracking
Every response includes cost estimation:
```json
{
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "cost_usd": 0.00045
  }
}
```

### Rate Limiting
Built-in protection against runaway costs and abuse.

### OpenAI Compatible
Works with any OpenAI SDK—Python, JavaScript, Go, Rust, or raw HTTP.

---

## 📚 API Reference

### Chat Completions

```
POST /v1/chat/completions
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model name (e.g., "gpt-4o") |
| `messages` | array | Yes | Conversation messages |
| `temperature` | float | No | Randomness (0-2, default: 0.7) |
| `max_tokens` | int | No | Max response length |
| `stream` | bool | No | Enable streaming (default: false) |

### List Models

```
GET /v1/models
```

Returns all available models with their configurations.

### Health Check

```
GET /health
```

Returns gateway status.

---

## 🔧 Self-Hosting Configuration

### Environment Variables

Create a `.env` file:

```bash
# Your provider API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
GROQ_API_KEY=gsk_...

# Gateway authentication (for your apps/customers)
GATEWAY_API_KEYS=key1,key2,key3

# Optional features
CACHE_ENABLED=true
RATE_LIMIT_ENABLED=true
```

### Model Configuration

Edit `config/models.yaml` to customize available models:

```yaml
models:
  - name: "my-gpt"
    provider: "openai"
    model_id: "gpt-4o"
    fallbacks: ["claude-sonnet", "gemini-pro"]
    timeout_seconds: 60
    cache_enabled: true

default_model: "my-gpt"
```

Hot-reload without restart:
```bash
curl -X POST http://localhost:5000/v1/models/reload
```

---

## 🆘 Support

- **Website:** [kgai.pro](https://kgai.pro)
- **Issues:** [GitHub Issues](https://github.com/kushgrandhi/llm_routing_server/issues)

---

## 📄 License

MIT License - use it however you want.

