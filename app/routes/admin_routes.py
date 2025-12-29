"""Admin routes for usage stats and monitoring."""

from flask import Blueprint, jsonify, request, current_app

from app.middleware.auth_middleware import require_api_key_authentication
from app.services.metrics_service import metrics_service


admin_blueprint = Blueprint("admin", __name__)


@admin_blueprint.route("/usage", methods=["GET"])
@require_api_key_authentication
def get_usage_statistics():
    """
    Get usage statistics for the gateway.
    
    Query Parameters:
        days: int - Number of days to look back (default: 30)
        model: str - Filter by model name (optional)
    
    Returns:
        Usage summary with totals and per-model breakdown.
    """
    usage_tracker_service = current_app.usage_tracker
    
    days_param = request.args.get("days", 30, type=int)
    model_filter = request.args.get("model")
    
    # Get API key hash for per-key stats (optional)
    api_key_hash_value = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key_hash_value = str(hash(auth_header[7:]))
    
    usage_summary = usage_tracker_service.get_usage_summary(
        api_key_hash=None,  # Admin sees all, use api_key_hash_value for per-key
        days=days_param,
        model_name=model_filter
    )
    
    return jsonify(usage_summary)


@admin_blueprint.route("/usage/recent", methods=["GET"])
@require_api_key_authentication
def get_recent_requests():
    """
    Get recent request logs.
    
    Query Parameters:
        limit: int - Number of requests to return (default: 50, max: 200)
    
    Returns:
        List of recent request logs.
    """
    usage_tracker_service = current_app.usage_tracker
    
    limit_param = min(request.args.get("limit", 50, type=int), 200)
    
    recent_requests_list = usage_tracker_service.get_recent_requests(
        limit=limit_param
    )
    
    return jsonify({
        "count": len(recent_requests_list),
        "requests": recent_requests_list
    })


@admin_blueprint.route("/metrics", methods=["GET"])
def get_prometheus_metrics():
    """
    Get Prometheus-format metrics.
    
    Returns:
        Prometheus metrics in text format.
    """
    return metrics_service.get_metrics_response()


@admin_blueprint.route("/health", methods=["GET"])
def detailed_health_check():
    """
    Detailed health check with dependency status.
    
    Returns:
        Health status with component checks.
    """
    health_status = {
        "status": "healthy",
        "service": "llm-gateway",
        "checks": {}
    }
    
    # Check LLM Router
    try:
        llm_router_service = current_app.llm_router
        models_count = len(llm_router_service.available_models)
        health_status["checks"]["llm_router"] = {
            "status": "healthy",
            "models_count": models_count
        }
    except Exception as router_error:
        health_status["checks"]["llm_router"] = {
            "status": "unhealthy",
            "error": str(router_error)
        }
        health_status["status"] = "degraded"
    
    # Check Usage Tracker
    try:
        usage_tracker_service = current_app.usage_tracker
        _ = usage_tracker_service.get_usage_summary(days=1)
        health_status["checks"]["usage_tracker"] = {"status": "healthy"}
    except Exception as tracker_error:
        health_status["checks"]["usage_tracker"] = {
            "status": "unhealthy",
            "error": str(tracker_error)
        }
        health_status["status"] = "degraded"
    
    # Check cache if enabled
    try:
        import litellm
        if litellm.cache:
            health_status["checks"]["cache"] = {
                "status": "healthy",
                "type": current_app.config.get("CACHE_TYPE", "local")
            }
    except Exception as cache_error:
        health_status["checks"]["cache"] = {
            "status": "unhealthy",
            "error": str(cache_error)
        }
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code

