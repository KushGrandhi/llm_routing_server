"""Models management routes for LLM Gateway."""

from flask import Blueprint, jsonify, current_app

from app.middleware.auth_middleware import require_api_key_authentication


models_blueprint = Blueprint("models", __name__)


@models_blueprint.route("/models", methods=["GET"])
@require_api_key_authentication
def list_available_models():
    """
    List all available models in the gateway.
    
    Returns:
        List of model objects with id, provider, and metadata.
    """
    llm_router_service = current_app.llm_router
    available_models_list = llm_router_service.get_available_models_list()
    
    return jsonify({
        "object": "list",
        "data": available_models_list
    })


@models_blueprint.route("/models/reload", methods=["POST"])
@require_api_key_authentication
def reload_model_configuration():
    """
    Hot-reload the model configuration from disk.
    
    Use this endpoint after editing config/models.yaml to apply changes
    without restarting the server.
    
    Returns:
        Status message with count of loaded models.
    """
    try:
        llm_router_service = current_app.llm_router
        reload_result = llm_router_service.reload_configuration()
        
        return jsonify({
            "status": "success",
            "message": "Model configuration reloaded",
            "models_count": reload_result["models_count"]
        })
    
    except FileNotFoundError as file_error:
        return jsonify({
            "error": {
                "message": f"Config file not found: {str(file_error)}",
                "type": "config_error",
                "code": "file_not_found"
            }
        }), 500
    
    except Exception as unexpected_error:
        return jsonify({
            "error": {
                "message": f"Failed to reload config: {str(unexpected_error)}",
                "type": "config_error",
                "code": "reload_failed"
            }
        }), 500


@models_blueprint.route("/models/<model_name>", methods=["GET"])
@require_api_key_authentication
def get_model_details(model_name: str):
    """
    Get details for a specific model.
    
    Args:
        model_name: The name/id of the model to retrieve.
    
    Returns:
        Model details including provider and configuration.
    """
    llm_router_service = current_app.llm_router
    available_models_list = llm_router_service.get_available_models_list()
    
    for model_info in available_models_list:
        if model_info["id"] == model_name:
            return jsonify(model_info)
    
    return jsonify({
        "error": {
            "message": f"Model '{model_name}' not found",
            "type": "not_found_error",
            "code": "model_not_found"
        }
    }), 404

