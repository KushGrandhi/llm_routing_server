"""Tests for LLM Router Service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.services.llm_router import LLMRouterService


@pytest.fixture
def sample_config_path(tmp_path):
    """Create a temporary config file for testing."""
    config_content = """
models:
  - name: "test-model"
    provider: "openai"
    model_id: "gpt-4o"
    fallbacks:
      - "fallback-model"
    timeout_seconds: 30
    max_retries: 2
    cache_enabled: true
    
  - name: "fallback-model"
    provider: "gemini"
    model_id: "gemini/gemini-2.0-flash"
    timeout_seconds: 30

default_model: "test-model"
global_fallbacks:
  - "fallback-model"
"""
    config_file = tmp_path / "models.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def llm_router(sample_config_path):
    """Create LLM Router with test config."""
    with patch("app.services.llm_router.get_gateway_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            cache_enabled=False,
            default_timeout_seconds=60,
            default_max_retries=2,
            cache_type="local",
            cache_default_ttl_seconds=3600
        )
        return LLMRouterService(config_file_path=sample_config_path)


class TestLLMRouterInitialization:
    """Tests for LLM Router initialization."""
    
    def test_loads_models_from_config(self, llm_router):
        """Should load models from YAML config."""
        assert len(llm_router.available_models) == 2
        assert "test-model" in llm_router.available_models
        assert "fallback-model" in llm_router.available_models
    
    def test_sets_default_model(self, llm_router):
        """Should set the default model from config."""
        assert llm_router.default_model_name == "test-model"
    
    def test_loads_global_fallbacks(self, llm_router):
        """Should load global fallbacks from config."""
        assert "fallback-model" in llm_router.global_fallback_models
    
    def test_raises_on_missing_config(self):
        """Should raise FileNotFoundError for missing config."""
        with patch("app.services.llm_router.get_gateway_settings") as mock_settings:
            mock_settings.return_value = MagicMock(cache_enabled=False)
            with pytest.raises(FileNotFoundError):
                LLMRouterService(config_file_path="/nonexistent/path.yaml")


class TestModelResolution:
    """Tests for model resolution logic."""
    
    def test_resolves_valid_model(self, llm_router):
        """Should resolve a valid model name."""
        model_id, config = llm_router._resolve_model_identifier("test-model")
        assert model_id == "gpt-4o"
        assert config["provider"] == "openai"
    
    def test_uses_default_when_none_specified(self, llm_router):
        """Should use default model when none specified."""
        model_id, config = llm_router._resolve_model_identifier(None)
        assert model_id == "gpt-4o"
    
    def test_raises_on_invalid_model(self, llm_router):
        """Should raise ValueError for invalid model name."""
        with pytest.raises(ValueError) as exc_info:
            llm_router._resolve_model_identifier("nonexistent-model")
        assert "not found" in str(exc_info.value)


class TestFallbackChain:
    """Tests for fallback chain building."""
    
    def test_builds_fallback_list(self, llm_router):
        """Should build fallback model list."""
        model_config = llm_router.available_models["test-model"]
        fallbacks = llm_router._build_fallback_model_list(model_config)
        
        assert len(fallbacks) == 1
        assert "gemini/gemini-2.0-flash" in fallbacks
    
    def test_includes_global_fallbacks(self, llm_router):
        """Should include global fallbacks."""
        model_config = {"fallbacks": []}
        fallbacks = llm_router._build_fallback_model_list(model_config)
        
        assert "gemini/gemini-2.0-flash" in fallbacks


class TestAvailableModels:
    """Tests for available models listing."""
    
    def test_returns_model_list(self, llm_router):
        """Should return list of available models."""
        models_list = llm_router.get_available_models_list()
        
        assert len(models_list) == 2
        
        test_model = next(m for m in models_list if m["id"] == "test-model")
        assert test_model["provider"] == "openai"
        assert test_model["is_default"] is True
        assert test_model["has_fallbacks"] is True
    
    def test_marks_default_model(self, llm_router):
        """Should mark the default model."""
        models_list = llm_router.get_available_models_list()
        
        default_models = [m for m in models_list if m["is_default"]]
        assert len(default_models) == 1
        assert default_models[0]["id"] == "test-model"


class TestConfigReload:
    """Tests for configuration hot-reload."""
    
    def test_reload_returns_status(self, llm_router):
        """Should return reload status."""
        result = llm_router.reload_configuration()
        
        assert result["status"] == "reloaded"
        assert result["models_count"] == 2
        assert result["default_model"] == "test-model"


class TestChatCompletion:
    """Tests for chat completion generation."""
    
    @patch("app.services.llm_router.litellm")
    def test_calls_litellm_completion(self, mock_litellm, llm_router):
        """Should call litellm.completion with correct params."""
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello!"),
                finish_reason="stop"
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        mock_litellm.completion.return_value = mock_response
        
        with patch("app.services.llm_router.completion_cost", return_value=0.001):
            result = llm_router.generate_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model"
            )
        
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args.kwargs
        
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
        assert "fallbacks" in call_kwargs
    
    @patch("app.services.llm_router.litellm")
    def test_returns_formatted_response(self, mock_litellm, llm_router):
        """Should return properly formatted response."""
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello!"),
                finish_reason="stop"
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        mock_litellm.completion.return_value = mock_response
        
        with patch("app.services.llm_router.completion_cost", return_value=0.001):
            result = llm_router.generate_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model"
            )
        
        assert result["id"] == "test-id"
        assert result["model"] == "test-model"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["usage"]["total_tokens"] == 15
        assert result["usage"]["cost_usd"] == 0.001
        assert "gateway_metadata" in result

