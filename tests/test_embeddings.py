"""Tests for POST /v1/embeddings.

No real provider is called: closed-provider routing mocks litellm.embedding,
and the open-weight route talks to a local fake OpenAI-compatible server.
"""

import base64
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import litellm
import pytest
import yaml
from werkzeug.serving import make_server

from app import create_flask_application
from app.services.llm_router import LLMRouterService


# Exactly representable in float32, so base64 round trips compare equal
FAKE_VECTORS = {
    "alpha": [0.5, -0.25, 0.125],
    "beta": [1.0, 0.0, -1.0],
}


class _FakeEmbeddingsHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /v1/embeddings (like Ollama, vLLM, TEI)."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.received.append({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "body": body,
        })
        if self.path != "/v1/embeddings":
            self.send_response(404)
            self.end_headers()
            return
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = []
        for index, text in enumerate(inputs):
            vector = FAKE_VECTORS.get(text, [0.0, 0.0, 0.0])
            if body.get("encoding_format") == "base64":
                vector = base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
            data.append({"object": "embedding", "index": index, "embedding": vector})
        payload = json.dumps({
            "object": "list",
            "data": data,
            "model": body["model"],
            "usage": {"prompt_tokens": 2 * len(inputs), "total_tokens": 2 * len(inputs)},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_open_weight_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeEmbeddingsHandler)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def embeddings_config_path(tmp_path, fake_open_weight_server):
    port = fake_open_weight_server.server_address[1]
    config = {
        "models": [
            {"name": "chat-model", "provider": "openai", "model_id": "gpt-4o"},
            {
                "name": "closed-embed",
                "mode": "embedding",
                "provider": "gemini",
                "model_id": "gemini/gemini-embedding-001",
                "timeout_seconds": 12,
                "max_retries": 3,
                "cache_enabled": False,
            },
            {
                "name": "open-embed",
                "mode": "embedding",
                "provider": "custom_openai",
                "model_id": "qwen3-embedding:0.6b",
                "api_base": f"http://127.0.0.1:{port}/v1",
                "max_retries": 0,
                "cache_enabled": False,
            },
        ],
        "default_model": "chat-model",
        # An embedding model in global_fallbacks must never serve chat
        "global_fallbacks": ["closed-embed", "chat-model"],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.safe_dump(config))
    return str(config_file)


@pytest.fixture
def app(embeddings_config_path):
    flask_app = create_flask_application()
    flask_app.config["TESTING"] = True
    flask_app.llm_router = LLMRouterService(config_file_path=embeddings_config_path)
    flask_app.usage_tracker = MagicMock()
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def _embedding_response(vectors, prompt_tokens=3):
    return litellm.EmbeddingResponse(
        model="gemini-embedding-001",
        data=[
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        usage=litellm.Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


class TestEmbeddingsValidation:
    """Malformed requests are rejected with 400 before any provider call."""

    @pytest.mark.parametrize("payload", [
        {"input": "hi"},                                   # no model
        {"model": ["closed-embed"], "input": "hi"},        # non-string model
        {"model": "closed-embed"},                         # no input
        {"model": "closed-embed", "input": ""},
        {"model": "closed-embed", "input": []},
        {"model": "closed-embed", "input": ["ok", ""]},
        {"model": "closed-embed", "input": ["ok", 5]},
        {"model": "closed-embed", "input": {"text": "hi"}},
        {"model": "closed-embed", "input": [[1, 2], "mixed"]},
        {"model": "closed-embed", "input": [True, False]},
        {"model": "closed-embed", "input": ["x"] * 2049},
        {"model": "closed-embed", "input": "hi", "encoding_format": "int8"},
        {"model": "closed-embed", "input": "hi", "dimensions": 0},
        {"model": "closed-embed", "input": "hi", "dimensions": "256"},
        {"model": "closed-embed", "input": "hi", "user": 7},
    ])
    def test_malformed_payload_is_400(self, client, payload):
        with patch("app.services.llm_router.litellm.embedding") as mock_embedding:
            response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 400
        assert response.get_json()["error"]["type"] == "invalid_request_error"
        mock_embedding.assert_not_called()

    @pytest.mark.parametrize("body", ["", "not json", "[1, 2]", '"hi"'])
    def test_non_object_body_is_400(self, client, body):
        response = client.post(
            "/v1/embeddings", data=body, content_type="application/json"
        )
        assert response.status_code == 400

    def test_unknown_model_is_400(self, client):
        response = client.post(
            "/v1/embeddings", json={"model": "no-such-embedder", "input": "hi"}
        )
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "invalid_model"

    def test_chat_model_is_rejected_for_embeddings(self, client):
        with patch("app.services.llm_router.litellm.embedding") as mock_embedding:
            response = client.post(
                "/v1/embeddings", json={"model": "chat-model", "input": "hi"}
            )
        mock_embedding.assert_not_called()
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "invalid_model"

    def test_embedding_model_is_rejected_for_chat(self, client):
        with patch("app.services.llm_router.litellm.completion") as mock_completion:
            response = client.post("/v1/chat/completions", json={
                "model": "closed-embed",
                "messages": [{"role": "user", "content": "hi"}],
            })
        mock_completion.assert_not_called()
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "invalid_model"

    def test_token_id_inputs_are_accepted(self, client):
        with patch("app.services.llm_router.litellm.embedding") as mock_embedding:
            mock_embedding.return_value = _embedding_response([[0.1], [0.2]])
            response = client.post(
                "/v1/embeddings",
                json={"model": "closed-embed", "input": [[1, 2, 3], [4, 5]]},
            )
        assert response.status_code == 200
        assert mock_embedding.call_args.kwargs["input"] == [[1, 2, 3], [4, 5]]


class TestEmbeddingsRouting:
    """Requests go to the provider declared in config."""

    def test_routes_to_configured_provider_model(self, client, app):
        with patch("app.services.llm_router.litellm.embedding") as mock_embedding:
            mock_embedding.return_value = _embedding_response([[0.1, 0.2], [0.3, 0.4]], 7)
            response = client.post("/v1/embeddings", json={
                "model": "closed-embed",
                "input": ["first", "second"],
                "dimensions": 256,
                "user": "u-1",
            })

        assert response.status_code == 200
        call_kwargs = mock_embedding.call_args.kwargs
        assert call_kwargs["model"] == "gemini/gemini-embedding-001"
        assert call_kwargs["input"] == ["first", "second"]
        assert call_kwargs["timeout"] == 12
        assert call_kwargs["max_retries"] == 3
        assert call_kwargs["dimensions"] == 256
        assert call_kwargs["user"] == "u-1"
        assert "api_base" not in call_kwargs
        assert "cache" not in call_kwargs

        data = response.get_json()
        assert data["object"] == "list"
        assert data["model"] == "closed-embed"
        assert [item["index"] for item in data["data"]] == [0, 1]
        assert data["data"][0] == {
            "object": "embedding", "index": 0, "embedding": [0.1, 0.2]
        }
        assert data["usage"]["prompt_tokens"] == 7
        assert data["usage"]["total_tokens"] == 7
        assert data["gateway_metadata"]["provider_model"] == "gemini/gemini-embedding-001"

    def test_usage_is_logged_for_embeddings(self, client, app):
        with patch("app.services.llm_router.litellm.embedding") as mock_embedding:
            mock_embedding.return_value = _embedding_response([[0.1]], 5)
            client.post("/v1/embeddings", json={"model": "closed-embed", "input": "x"})

        log_kwargs = app.usage_tracker.log_request.call_args.kwargs
        assert log_kwargs["model_name"] == "closed-embed"
        assert log_kwargs["prompt_tokens"] == 5
        assert log_kwargs["completion_tokens"] == 0
        assert log_kwargs["status_code"] == 200
        assert log_kwargs["provider_model"] == "gemini/gemini-embedding-001"

    def test_provider_bad_request_is_400(self, client):
        with patch("app.services.llm_router.litellm.embedding") as mock_embedding:
            mock_embedding.side_effect = litellm.BadRequestError(
                message="dimensions not supported",
                model="gemini-embedding-001",
                llm_provider="gemini",
            )
            response = client.post(
                "/v1/embeddings", json={"model": "closed-embed", "input": "x"}
            )
        assert response.status_code == 400

    def test_provider_failure_is_500(self, client, app):
        with patch("app.services.llm_router.litellm.embedding") as mock_embedding:
            mock_embedding.side_effect = RuntimeError("upstream down")
            response = client.post(
                "/v1/embeddings", json={"model": "closed-embed", "input": "x"}
            )
        assert response.status_code == 500
        assert app.usage_tracker.log_request.call_args.kwargs["status_code"] == 500

    def test_chat_fallbacks_skip_embedding_models(self, app):
        router = app.llm_router
        assert router._fallback_model_names({}, None) == ["chat-model"]

    def test_models_list_reports_mode(self, client):
        models = {m["id"]: m for m in client.get("/v1/models").get_json()["data"]}
        assert models["chat-model"]["mode"] == "chat"
        assert models["open-embed"]["mode"] == "embedding"


class TestOpenWeightRoute:
    """custom_openai + api_base to a local OpenAI-compatible server, no mocks."""

    def test_open_weight_model_end_to_end(self, client, fake_open_weight_server):
        response = client.post(
            "/v1/embeddings", json={"model": "open-embed", "input": ["alpha", "beta"]}
        )

        assert response.status_code == 200, response.get_json()
        data = response.get_json()
        assert data["model"] == "open-embed"
        assert [item["embedding"] for item in data["data"]] == [
            FAKE_VECTORS["alpha"], FAKE_VECTORS["beta"]
        ]
        assert data["usage"]["prompt_tokens"] == 4

        upstream = fake_open_weight_server.received[-1]
        assert upstream["path"] == "/v1/embeddings"
        assert upstream["body"]["model"] == "qwen3-embedding:0.6b"
        assert upstream["body"]["input"] == ["alpha", "beta"]
        # Placeholder key, never a real provider key from the environment
        assert upstream["authorization"] == "Bearer not-needed"

    def test_base64_encoding_format(self, client):
        response = client.post("/v1/embeddings", json={
            "model": "open-embed", "input": "alpha", "encoding_format": "base64"
        })
        assert response.status_code == 200
        encoded = response.get_json()["data"][0]["embedding"]
        assert list(struct.unpack("<3f", base64.b64decode(encoded))) == FAKE_VECTORS["alpha"]

    def test_openai_python_client_against_gateway(self, app):
        openai = pytest.importorskip("openai")
        server = make_server("127.0.0.1", 0, app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = openai.OpenAI(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="any-key-in-dev-mode",
                max_retries=0,
            )
            # Default: the SDK asks for base64 and decodes it
            result = client.embeddings.create(model="open-embed", input=["alpha", "beta"])
            assert [list(item.embedding) for item in result.data] == [
                FAKE_VECTORS["alpha"], FAKE_VECTORS["beta"]
            ]
            assert result.model == "open-embed"

            floats = client.embeddings.create(
                model="open-embed", input="beta", encoding_format="float"
            )
            assert floats.data[0].embedding == FAKE_VECTORS["beta"]
        finally:
            server.shutdown()


class TestEmbeddingsAuth:
    """Embeddings use the same API-key auth as chat."""

    def test_missing_key_is_401(self, client, monkeypatch):
        monkeypatch.setenv("GATEWAY_API_KEYS", "secret-test-key")
        response = client.post(
            "/v1/embeddings", json={"model": "open-embed", "input": "alpha"}
        )
        assert response.status_code == 401

    def test_wrong_key_is_401(self, client, monkeypatch):
        monkeypatch.setenv("GATEWAY_API_KEYS", "secret-test-key")
        response = client.post(
            "/v1/embeddings",
            json={"model": "open-embed", "input": "alpha"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401

    def test_valid_key_is_200(self, client, monkeypatch):
        monkeypatch.setenv("GATEWAY_API_KEYS", "secret-test-key")
        response = client.post(
            "/v1/embeddings",
            json={"model": "open-embed", "input": "alpha"},
            headers={"Authorization": "Bearer secret-test-key"},
        )
        assert response.status_code == 200


def test_shipped_config_has_an_open_weight_embedding_route():
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    models = yaml.safe_load(config_path.read_text())["models"]
    open_routes = [
        m for m in models
        if m.get("mode") == "embedding" and m.get("api_base")
    ]
    assert open_routes, "config/models.yaml should declare a self-hosted embedding model"
