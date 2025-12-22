"""LLM Router Service - Unified interface to multiple LLM providers."""

import os
from pathlib import Path
from typing import Any, Generator, Optional, Union

import yaml
import litellm


class LLMRouterService:
    """Service for routing requests to different LLM providers via LiteLLM."""
    
    def __init__(self, config_file_path: Optional[str] = None):
        """Initialize the LLM Router with model configuration."""
        if config_file_path is None:
            config_file_path = str(Path(__file__).parent.parent.parent / "config" / "models.yaml")
        
        self.config_file_path = Path(config_file_path)
        self.available_models: dict[str, dict] = {}
        self.default_model_name: Optional[str] = None
        
        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported params instead of erroring
        
        self._load_model_configuration()
    
    def _load_model_configuration(self):
        """Load model configuration from YAML file."""
        if not self.config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file_path}")
        
        with open(self.config_file_path, "r") as config_file:
            config_data = yaml.safe_load(config_file)
        
        self.available_models = {}
        for model_config in config_data.get("models", []):
            model_name = model_config.get("name")
            if model_name:
                self.available_models[model_name] = model_config
        
        self.default_model_name = config_data.get("default_model")
    
    def reload_configuration(self):
        """Hot-reload model configuration without server restart."""
        self._load_model_configuration()
        return {"status": "reloaded", "models_count": len(self.available_models)}
    
    def get_available_models_list(self) -> list[dict]:
        """Return list of available models with their metadata."""
        models_list = []
        for model_name, model_config in self.available_models.items():
            models_list.append({
                "id": model_name,
                "provider": model_config.get("provider"),
                "model_id": model_config.get("model_id"),
                "is_default": model_name == self.default_model_name
            })
        return models_list
    
    def _resolve_model_identifier(self, requested_model: str) -> tuple[str, dict]:
        """
        Resolve the requested model name to LiteLLM model identifier.
        
        Returns:
            Tuple of (litellm_model_id, model_config)
        """
        # Use default if no model specified
        if not requested_model:
            requested_model = self.default_model_name
        
        if requested_model not in self.available_models:
            available_names = list(self.available_models.keys())
            raise ValueError(f"Model '{requested_model}' not found. Available: {available_names}")
        
        model_config = self.available_models[requested_model]
        provider_name = model_config.get("provider")
        model_identifier = model_config.get("model_id")
        
        # Build LiteLLM model string based on provider
        if provider_name == "openai":
            litellm_model_id = model_identifier
        elif provider_name == "anthropic":
            litellm_model_id = model_identifier
        elif provider_name == "gemini":
            litellm_model_id = model_identifier
        elif provider_name == "groq":
            litellm_model_id = model_identifier
        elif provider_name == "huggingface":
            litellm_model_id = model_identifier
        elif provider_name == "custom_openai":
            litellm_model_id = f"openai/{model_identifier}"
        else:
            litellm_model_id = model_identifier
        
        return litellm_model_id, model_config
    
    def generate_chat_completion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **additional_params
    ) -> Union[dict, Generator]:
        """
        Generate chat completion using the specified model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name from config (uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **additional_params: Additional parameters passed to LiteLLM
        
        Returns:
            Chat completion response or generator for streaming
        """
        litellm_model_id, model_config = self._resolve_model_identifier(model)
        
        # Build completion kwargs
        completion_kwargs = {
            "model": litellm_model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        
        # Handle custom OpenAI-compatible endpoints
        if model_config.get("provider") == "custom_openai":
            api_base_url = model_config.get("api_base")
            if api_base_url:
                completion_kwargs["api_base"] = api_base_url
                # Custom endpoints might need a dummy key
                completion_kwargs["api_key"] = model_config.get("api_key", "not-needed")
        
        # Add any extra params
        completion_kwargs.update(additional_params)
        
        # Call LiteLLM
        response = litellm.completion(**completion_kwargs)
        
        if stream:
            return self._stream_response_generator(response, model)
        
        return self._format_completion_response(response, model)
    
    def _stream_response_generator(
        self, 
        stream_response, 
        model_name: str
    ) -> Generator[dict, None, None]:
        """Generate streaming response chunks."""
        for chunk in stream_response:
            yield {
                "id": chunk.id if hasattr(chunk, "id") else "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": chunk.choices[0].delta.content or ""
                    } if chunk.choices[0].delta else {},
                    "finish_reason": chunk.choices[0].finish_reason
                }]
            }
    
    def _format_completion_response(self, response, model_name: str) -> dict:
        """Format the completion response."""
        return {
            "id": response.id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                },
                "finish_reason": response.choices[0].finish_reason
            }],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
        }

