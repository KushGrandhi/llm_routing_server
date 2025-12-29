"""LLM Router Service - Unified interface to multiple LLM providers."""

import logging
import time
from pathlib import Path
from typing import Any, Generator, Optional, Union

import yaml
import litellm
from litellm import completion_cost

from app.config.settings import get_gateway_settings


logger = logging.getLogger(__name__)


class LLMRouterService:
    """Service for routing requests to different LLM providers via LiteLLM."""
    
    def __init__(self, config_file_path: Optional[str] = None):
        """Initialize the LLM Router with model configuration."""
        if config_file_path is None:
            config_file_path = str(
                Path(__file__).parent.parent.parent / "config" / "models.yaml"
            )
        
        self.config_file_path = Path(config_file_path)
        self.available_models: dict[str, dict] = {}
        self.default_model_name: Optional[str] = None
        self.global_fallback_models: list[str] = []
        self.gateway_settings = get_gateway_settings()
        
        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported params instead of erroring
        litellm.set_verbose = False  # Reduce noise in logs
        
        self._load_model_configuration()
        self._setup_caching()
    
    def _setup_caching(self):
        """Configure LiteLLM caching based on settings."""
        if not self.gateway_settings.cache_enabled:
            logger.info("Caching disabled via settings")
            return
        
        try:
            if self.gateway_settings.cache_type == "redis":
                litellm.cache = litellm.Cache(
                    type="redis",
                    host=self.gateway_settings.redis_host,
                    port=self.gateway_settings.redis_port,
                    password=self.gateway_settings.redis_password,
                )
                logger.info(
                    f"Redis cache enabled at "
                    f"{self.gateway_settings.redis_host}:{self.gateway_settings.redis_port}"
                )
            else:
                litellm.cache = litellm.Cache(type="local")
                logger.info("Local in-memory cache enabled")
        except Exception as cache_error:
            logger.warning(f"Failed to setup cache, continuing without: {cache_error}")
    
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
        self.global_fallback_models = config_data.get("global_fallbacks", [])
        
        logger.info(
            f"Loaded {len(self.available_models)} models, "
            f"default: {self.default_model_name}"
        )
    
    def reload_configuration(self) -> dict:
        """Hot-reload model configuration without server restart."""
        self._load_model_configuration()
        return {
            "status": "reloaded",
            "models_count": len(self.available_models),
            "default_model": self.default_model_name
        }
    
    def get_available_models_list(self) -> list[dict]:
        """Return list of available models with their metadata."""
        models_list = []
        for model_name, model_config in self.available_models.items():
            models_list.append({
                "id": model_name,
                "provider": model_config.get("provider"),
                "model_id": model_config.get("model_id"),
                "is_default": model_name == self.default_model_name,
                "has_fallbacks": bool(model_config.get("fallbacks")),
                "cache_enabled": model_config.get("cache_enabled", True),
                "timeout_seconds": model_config.get(
                    "timeout_seconds",
                    self.gateway_settings.default_timeout_seconds
                ),
                "description": model_config.get("description", "")
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
            raise ValueError(
                f"Model '{requested_model}' not found. Available: {available_names}"
            )
        
        model_config = self.available_models[requested_model]
        provider_name = model_config.get("provider")
        model_identifier = model_config.get("model_id")
        
        # Build LiteLLM model string based on provider
        if provider_name == "custom_openai":
            litellm_model_id = f"openai/{model_identifier}"
        else:
            litellm_model_id = model_identifier
        
        return litellm_model_id, model_config
    
    def _build_fallback_model_list(self, model_config: dict) -> list[str]:
        """Build list of fallback model IDs for LiteLLM."""
        fallback_names = model_config.get("fallbacks", [])
        fallback_model_ids = []
        
        for fallback_name in fallback_names:
            if fallback_name in self.available_models:
                fallback_config = self.available_models[fallback_name]
                fallback_id = fallback_config.get("model_id")
                if fallback_config.get("provider") == "custom_openai":
                    fallback_id = f"openai/{fallback_id}"
                fallback_model_ids.append(fallback_id)
        
        # Add global fallbacks if configured
        for global_fallback in self.global_fallback_models:
            if global_fallback in self.available_models:
                global_config = self.available_models[global_fallback]
                global_id = global_config.get("model_id")
                if global_config.get("provider") == "custom_openai":
                    global_id = f"openai/{global_id}"
                if global_id not in fallback_model_ids:
                    fallback_model_ids.append(global_id)
        
        return fallback_model_ids
    
    def _calculate_request_cost(
        self,
        response,
        model_identifier: str
    ) -> Optional[float]:
        """Calculate cost for the request using LiteLLM's cost tracking."""
        try:
            cost_value = completion_cost(completion_response=response)
            return round(cost_value, 6)
        except Exception as cost_error:
            logger.debug(f"Could not calculate cost for {model_identifier}: {cost_error}")
            return None
    
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
        request_start_time = time.time()
        
        litellm_model_id, model_config = self._resolve_model_identifier(model)
        
        # Get timeout and retry settings
        timeout_seconds = model_config.get(
            "timeout_seconds",
            self.gateway_settings.default_timeout_seconds
        )
        max_retries_count = model_config.get(
            "max_retries",
            self.gateway_settings.default_max_retries
        )
        
        # Build fallback list
        fallback_model_ids = self._build_fallback_model_list(model_config)
        
        # Build completion kwargs
        completion_kwargs = {
            "model": litellm_model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "timeout": timeout_seconds,
            "num_retries": max_retries_count,
        }
        
        # Add fallbacks if available
        if fallback_model_ids:
            completion_kwargs["fallbacks"] = fallback_model_ids
        
        # Add caching control
        if model_config.get("cache_enabled", True) and self.gateway_settings.cache_enabled:
            cache_ttl = model_config.get(
                "cache_ttl_seconds",
                self.gateway_settings.cache_default_ttl_seconds
            )
            completion_kwargs["cache"] = {"ttl": cache_ttl}
        
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        
        # Handle custom OpenAI-compatible endpoints
        if model_config.get("provider") == "custom_openai":
            api_base_url = model_config.get("api_base")
            if api_base_url:
                completion_kwargs["api_base"] = api_base_url
                completion_kwargs["api_key"] = model_config.get("api_key", "not-needed")
        
        # Add any extra params
        completion_kwargs.update(additional_params)
        
        # Call LiteLLM
        response = litellm.completion(**completion_kwargs)
        
        request_duration_ms = int((time.time() - request_start_time) * 1000)
        
        if stream:
            return self._stream_response_generator(
                response,
                model or self.default_model_name,
                request_start_time
            )
        
        return self._format_completion_response(
            response,
            model or self.default_model_name,
            litellm_model_id,
            request_duration_ms
        )
    
    def _stream_response_generator(
        self,
        stream_response,
        model_name: str,
        request_start_time: float
    ) -> Generator[dict, None, None]:
        """Generate streaming response chunks."""
        for chunk in stream_response:
            delta_data = {}
            
            if chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                # Content
                if hasattr(delta, "content") and delta.content:
                    delta_data["content"] = delta.content
                
                # Role (usually only in first chunk)
                if hasattr(delta, "role") and delta.role:
                    delta_data["role"] = delta.role
                
                # Tool calls (function calling in streaming)
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    delta_data["tool_calls"] = [
                        {
                            "index": tc.index if hasattr(tc, "index") else 0,
                            "id": tc.id if hasattr(tc, "id") else None,
                            "type": tc.type if hasattr(tc, "type") else "function",
                            "function": {
                                "name": tc.function.name if hasattr(tc.function, "name") else None,
                                "arguments": tc.function.arguments if hasattr(tc.function, "arguments") else ""
                            }
                        }
                        for tc in delta.tool_calls
                    ]
            
            yield {
                "id": chunk.id if hasattr(chunk, "id") else "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": delta_data,
                    "finish_reason": chunk.choices[0].finish_reason
                }]
            }
    
    def _format_completion_response(
        self,
        response,
        model_name: str,
        litellm_model_id: str,
        latency_milliseconds: int
    ) -> dict:
        """Format the completion response with cost and metadata."""
        # Calculate cost
        cost_usd = self._calculate_request_cost(response, litellm_model_id)
        
        # Build usage dict with cost
        usage_data = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        
        if cost_usd is not None:
            usage_data["cost_usd"] = cost_usd
        
        # Build message object with all possible fields
        response_message = response.choices[0].message
        message_data = {
            "role": "assistant",
            "content": response_message.content
        }
        
        # Include tool_calls if present (function calling)
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            message_data["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response_message.tool_calls
            ]
        
        # Include refusal if present (safety)
        if hasattr(response_message, "refusal") and response_message.refusal:
            message_data["refusal"] = response_message.refusal
        
        return {
            "id": response.id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": message_data,
                "finish_reason": response.choices[0].finish_reason
            }],
            "usage": usage_data,
            "gateway_metadata": {
                "latency_ms": latency_milliseconds,
                "provider_model": litellm_model_id,
                "cached": getattr(response, "_hidden_params", {}).get("cache_hit", False)
            }
        }
