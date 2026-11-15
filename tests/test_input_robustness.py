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
