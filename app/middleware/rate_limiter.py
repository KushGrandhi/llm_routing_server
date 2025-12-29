"""Rate limiting middleware for LLM Gateway."""

import logging
from typing import Optional

from flask import Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from app.config.settings import get_gateway_settings


logger = logging.getLogger(__name__)


def _get_rate_limit_key() -> str:
    """
    Get the key for rate limiting.
    
    Uses API key if present, otherwise falls back to IP address.
    This allows per-client rate limiting.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        # Use hash of API key for privacy
        api_key_value = auth_header[7:]
        return f"apikey:{hash(api_key_value)}"
    return f"ip:{get_remote_address()}"


def create_rate_limiter() -> Limiter:
    """Create and configure the rate limiter instance."""
    gateway_settings = get_gateway_settings()
    
    limiter_instance = Limiter(
        key_func=_get_rate_limit_key,
        default_limits=[gateway_settings.rate_limit_default],
        storage_uri=gateway_settings.rate_limit_storage_uri,
        strategy="fixed-window",
    )
    
    return limiter_instance


def init_rate_limiter(
    flask_app: Flask,
    limiter_instance: Limiter
) -> Optional[Limiter]:
    """
    Initialize rate limiter with Flask app.
    
    Returns None if rate limiting is disabled.
    """
    gateway_settings = get_gateway_settings()
    
    if not gateway_settings.rate_limit_enabled:
        logger.info("Rate limiting disabled via settings")
        return None
    
    limiter_instance.init_app(flask_app)
    logger.info(
        f"Rate limiting enabled: {gateway_settings.rate_limit_default}"
    )
    
    return limiter_instance


# Global limiter instance
rate_limiter = create_rate_limiter()

