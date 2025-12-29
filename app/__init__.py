"""Flask application factory for LLM Gateway."""

import logging

from flask import Flask
from flask_cors import CORS

from app.config.settings import get_gateway_settings
from app.routes.chat_routes import chat_blueprint
from app.routes.models_routes import models_blueprint
from app.routes.admin_routes import admin_blueprint
from app.services.llm_router import LLMRouterService
from app.services.usage_tracker import UsageTrackerService
from app.middleware.rate_limiter import rate_limiter, init_rate_limiter


def setup_logging():
    """Configure application logging."""
    gateway_settings = get_gateway_settings()
    
    logging.basicConfig(
        level=getattr(logging, gateway_settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def create_flask_application() -> Flask:
    """Create and configure the Flask application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    flask_application = Flask(__name__)
    gateway_settings = get_gateway_settings()
    
    # Store settings in app config
    flask_application.config["CACHE_TYPE"] = gateway_settings.cache_type
    
    # Enable CORS for all routes
    CORS(flask_application)
    
    # Initialize rate limiter
    init_rate_limiter(flask_application, rate_limiter)
    
    # Initialize services
    llm_router_service = LLMRouterService()
    usage_tracker_service = UsageTrackerService()
    
    # Attach services to app context
    flask_application.llm_router = llm_router_service
    flask_application.usage_tracker = usage_tracker_service
    
    # Register blueprints
    flask_application.register_blueprint(chat_blueprint, url_prefix="/v1")
    flask_application.register_blueprint(models_blueprint, url_prefix="/v1")
    flask_application.register_blueprint(admin_blueprint, url_prefix="/v1")
    
    # Simple health check endpoint (detailed one in admin_routes)
    @flask_application.route("/health")
    def health_check_endpoint():
        return {"status": "healthy", "service": "llm-gateway"}
    
    # Root endpoint
    @flask_application.route("/")
    def root_endpoint():
        return {
            "service": "llm-gateway",
            "version": "1.0.0",
            "endpoints": {
                "chat": "/v1/chat/completions",
                "models": "/v1/models",
                "usage": "/v1/usage",
                "metrics": "/v1/metrics",
                "health": "/health"
            }
        }
    
    logger.info(
        f"LLM Gateway initialized with {len(llm_router_service.available_models)} models"
    )
    
    return flask_application
