"""Flask application factory for LLM Gateway."""

from flask import Flask
from flask_cors import CORS

from app.routes.chat_routes import chat_blueprint
from app.routes.models_routes import models_blueprint
from app.services.llm_router import LLMRouterService


def create_flask_application():
    """Create and configure the Flask application."""
    flask_application = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(flask_application)
    
    # Initialize LLM Router service
    llm_router_service = LLMRouterService()
    flask_application.llm_router = llm_router_service
    
    # Register blueprints
    flask_application.register_blueprint(chat_blueprint, url_prefix="/v1")
    flask_application.register_blueprint(models_blueprint, url_prefix="/v1")
    
    # Health check endpoint
    @flask_application.route("/health")
    def health_check_endpoint():
        return {"status": "healthy", "service": "llm-gateway"}
    
    return flask_application

