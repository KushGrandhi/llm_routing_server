"""Pydantic settings for LLM Gateway configuration."""

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class GatewaySettings(BaseSettings):
    """Central configuration for the LLM Gateway."""
    
    # Server Settings
    server_port: int = Field(default=5000, description="Port to run the server on")
    debug_mode_enabled: bool = Field(default=False, description="Enable Flask debug mode")
    
    # Authentication
    gateway_api_keys: str = Field(
        default="",
        description="Comma-separated list of valid API keys"
    )
    
    # LLM Provider API Keys
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    gemini_api_key: Optional[str] = Field(default=None)
    groq_api_key: Optional[str] = Field(default=None)
    huggingface_api_key: Optional[str] = Field(default=None)
    
    # Cache Settings
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_type: Literal["local", "redis"] = Field(
        default="local",
        description="Cache backend type"
    )
    redis_host: str = Field(default="localhost", description="Redis host for caching")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    cache_default_ttl_seconds: int = Field(
        default=3600,
        description="Default cache TTL in seconds"
    )
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_default: str = Field(
        default="100 per minute",
        description="Default rate limit"
    )
    rate_limit_storage_uri: str = Field(
        default="memory://",
        description="Rate limit storage backend URI"
    )
    
    # Logging & Monitoring
    request_logging_enabled: bool = Field(
        default=True,
        description="Enable request/response logging"
    )
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Request Defaults
    default_timeout_seconds: int = Field(
        default=60,
        description="Default request timeout"
    )
    default_max_retries: int = Field(
        default=2,
        description="Default retry count on failure"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_gateway_settings() -> GatewaySettings:
    """Get cached gateway settings instance."""
    return GatewaySettings()

