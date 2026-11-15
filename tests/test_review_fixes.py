import hashlib
import os
import subprocess
import sys

import pytest
from flask import request

from app import create_flask_application
from app.routes.chat_routes import _get_api_key_hash


@pytest.fixture()
def client():
    application = create_flask_application()
    application.config["TESTING"] = True
    return application.test_client()


@pytest.mark.parametrize("body", ["[1, 2]", '"hi"', "5"])
def test_valid_json_that_is_not_an_object_is_400(client, body):
    response = client.post(
        "/v1/chat/completions", data=body, content_type="application/json"
    )
    assert response.status_code == 400


def test_non_string_model_is_400(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": ["x"], "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400


def test_unknown_model_with_stream_is_400_not_200(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "no-such-model-xyz",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 400


def test_client_address_comes_from_proxy_header(client):
    application = create_flask_application()
    application.config["TESTING"] = True

    @application.route("/_test_ip")
    def _report_remote_address():
        return request.remote_addr

    response = application.test_client().get(
        "/_test_ip", headers={"X-Forwarded-For": "203.0.113.7"}
    )
    assert response.data == b"203.0.113.7"


def test_api_key_hash_is_sha256_and_stable_across_processes():
    code = (
        "from flask import Flask\n"
        "from app.routes.chat_routes import _get_api_key_hash\n"
        "app = Flask('x')\n"
        "with app.test_request_context('/', headers={'Authorization': 'Bearer abc'}):\n"
        "    print(_get_api_key_hash())\n"
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs = set()
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=project_root)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outputs.add(result.stdout.strip())
    assert outputs == {hashlib.sha256(b"abc").hexdigest()[:32]}
