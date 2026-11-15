import hashlib
import subprocess
import sys
import os

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
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


def test_different_api_keys_get_different_keys():
    assert _key_for_header("Bearer key-one") != _key_for_header("Bearer key-two")


def test_empty_bearer_token_falls_back_to_ip():
    assert _key_for_header("Bearer ").startswith("ip:")


def test_missing_header_uses_ip():
    assert _key_for_header(None).startswith("ip:")
