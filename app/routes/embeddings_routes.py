"""Embeddings routes for LLM Gateway (OpenAI-compatible /v1/embeddings)."""

import base64
import struct
import time

import litellm
from flask import Blueprint, request, jsonify, current_app

from app.middleware.auth_middleware import require_api_key_authentication
from app.routes.chat_routes import _log_error_request, _log_successful_request
from app.services.metrics_service import metrics_service


embeddings_blueprint = Blueprint("embeddings", __name__)

# OpenAI's documented per-request limit on input items
MAX_INPUT_ITEMS = 2048
ENCODING_FORMATS = {"float", "base64"}
# Request fields this route handles itself; never taken from passthrough_params
STANDARD_FIELDS = {"model", "input", "dimensions", "encoding_format", "user"}


def _passthrough_params(request_payload: dict, model_config: dict) -> dict:
    """
    Provider-specific request keys the model's config allowlists via
    `passthrough_params` (e.g. [task_type]). Any other unknown key is
    dropped, so clients cannot set api_base, api_key or similar.
    """
    allowed = set(model_config.get("passthrough_params") or []) - STANDARD_FIELDS
    return {key: request_payload[key] for key in allowed if key in request_payload}


def _invalid_request(message: str, code: str = "invalid_request"):
    return jsonify({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": code
        }
    }), 400


def _is_token_list(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(token, int) and not isinstance(token, bool) for token in value)
    )


def _validate_input(input_value):
    """Return an error message for an invalid `input`, else None."""
    if isinstance(input_value, str):
        return None if input_value else "input must not be empty"
    if not isinstance(input_value, list) or not input_value:
        return "input must be a non-empty string or a non-empty array"
    if len(input_value) > MAX_INPUT_ITEMS:
        return f"input must have at most {MAX_INPUT_ITEMS} items"
    if all(isinstance(item, str) for item in input_value):
        if any(not item for item in input_value):
            return "input strings must not be empty"
        return None
    if _is_token_list(input_value) or all(_is_token_list(item) for item in input_value):
        return None
    return "input must be a string, an array of strings, or token id array(s)"


def _to_base64(vector: list) -> str:
    """Encode floats the way OpenAI does for encoding_format=base64."""
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


@embeddings_blueprint.route("/embeddings", methods=["POST"])
@require_api_key_authentication
def create_embeddings():
    """
    Create embeddings with a configured `mode: embedding` model.

    Request Body:
        model: str - Embedding model name from config (required)
        input: str | list[str] | list[int] | list[list[int]] - Text or token ids
        encoding_format: str - "float" (default) or "base64" (optional)
        dimensions: int - Output size, if the model supports it (optional)
        user: str - End-user id passed to the provider (optional)
        Any key listed in the model's `passthrough_params` config (optional);
        other keys are ignored.
    """
    request_start_time = time.time()
    request_payload = request.get_json(silent=True)

    if not isinstance(request_payload, dict) or not request_payload:
        return _invalid_request("Request body must be a JSON object", "missing_body")

    model_name = request_payload.get("model")
    usage_tracker_service = current_app.usage_tracker

    if not isinstance(model_name, str) or not model_name:
        return _invalid_request("model is required and must be a string", "invalid_model")

    input_error = _validate_input(request_payload.get("input"))
    if input_error:
        return _invalid_request(input_error, "invalid_input")

    encoding_format = request_payload.get("encoding_format", "float")
    if encoding_format not in ENCODING_FORMATS:
        return _invalid_request(
            f"encoding_format must be one of {sorted(ENCODING_FORMATS)}"
        )

    additional_params = {}
    dimensions = request_payload.get("dimensions")
    if dimensions is not None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            return _invalid_request("dimensions must be a positive integer")
        additional_params["dimensions"] = dimensions
    user_id = request_payload.get("user")
    if user_id is not None:
        if not isinstance(user_id, str):
            return _invalid_request("user must be a string")
        additional_params["user"] = user_id

    llm_router_service = current_app.llm_router
    try:
        _, model_config = llm_router_service._resolve_model_identifier(
            model_name, mode="embedding"
        )
    except ValueError as model_error:
        _log_error_request(
            usage_tracker_service, model_name, 400, str(model_error), request_start_time
        )
        return _invalid_request(str(model_error), "invalid_model")
    additional_params.update(_passthrough_params(request_payload, model_config))

    metrics_service.increment_active_requests(model_name)
    try:
        embedding_response = llm_router_service.generate_embeddings(
            request_payload["input"],
            model=model_name,
            **additional_params
        )
    except litellm.BadRequestError as provider_error:
        _log_error_request(
            usage_tracker_service, model_name, 400, str(provider_error), request_start_time
        )
        return _invalid_request(str(provider_error), "provider_rejected_request")
    except Exception as unexpected_error:
        _log_error_request(
            usage_tracker_service, model_name, 500, str(unexpected_error), request_start_time
        )
        return jsonify({
            "error": {
                "message": f"Internal server error: {str(unexpected_error)}",
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500
    finally:
        metrics_service.decrement_active_requests(model_name)

    _log_successful_request(
        usage_tracker_service, embedding_response, model_name, request_start_time
    )

    if encoding_format == "base64":
        for item in embedding_response["data"]:
            item["embedding"] = _to_base64(item["embedding"])

    return jsonify(embedding_response)
