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
