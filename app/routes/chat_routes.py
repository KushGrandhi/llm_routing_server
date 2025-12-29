"""Chat completion routes for LLM Gateway."""

import json
import time

from flask import Blueprint, request, jsonify, Response, current_app, stream_with_context

from app.middleware.auth_middleware import require_api_key_authentication
from app.services.metrics_service import metrics_service


chat_blueprint = Blueprint("chat", __name__)

# Parameters we handle explicitly (not passed through as additional_params)
HANDLED_PARAMETERS = {
    "model", "messages", "temperature", "max_tokens", "stream"
}

# OpenAI-compatible parameters to pass through
PASSTHROUGH_PARAMETERS = {
    "response_format",      # JSON mode, structured outputs
    "tools",                # Function calling
    "tool_choice",          # Function calling control
    "top_p",                # Nucleus sampling
    "presence_penalty",     # Repetition control
    "frequency_penalty",    # Repetition control
    "stop",                 # Stop sequences
    "logprobs",             # Log probabilities
    "top_logprobs",         # Top log probabilities
    "n",                    # Number of completions
    "seed",                 # Reproducibility
    "user",                 # End-user tracking
    "logit_bias",           # Token biasing
    "parallel_tool_calls",  # Parallel function calls
}


def _get_api_key_hash() -> str:
    """Get hash of API key for logging."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return str(hash(auth_header[7:]))
    return "anonymous"


def _extract_additional_parameters(request_payload: dict) -> dict:
    """Extract additional OpenAI-compatible parameters from request."""
    additional_params = {}
    
    for param_name in PASSTHROUGH_PARAMETERS:
        if param_name in request_payload:
            additional_params[param_name] = request_payload[param_name]
    
    return additional_params


@chat_blueprint.route("/chat/completions", methods=["POST"])
@require_api_key_authentication
def create_chat_completion():
    """
    Create a chat completion using the specified model.
    
    OpenAI-compatible endpoint that routes to various LLM providers.
    
    Request Body:
        model: str - Model name from config (optional, uses default)
        messages: list - List of message objects with 'role' and 'content'
        temperature: float - Sampling temperature (default: 0.7)
        max_tokens: int - Maximum tokens to generate (optional)
        stream: bool - Whether to stream the response (default: false)
        
        Additional supported parameters:
        response_format: dict - Output format (e.g., {"type": "json_object"})
        tools: list - Function definitions for function calling
        tool_choice: str/dict - Control function calling behavior
        top_p: float - Nucleus sampling parameter
        presence_penalty: float - Presence penalty (-2.0 to 2.0)
        frequency_penalty: float - Frequency penalty (-2.0 to 2.0)
        stop: str/list - Stop sequences
        seed: int - Random seed for reproducibility
    """
    request_start_time = time.time()
    model_name = None
    
    try:
        request_payload = request.get_json()
        
        if not request_payload:
            return jsonify({
                "error": {
                    "message": "Request body is required",
                    "type": "invalid_request_error",
                    "code": "missing_body"
                }
            }), 400
        
        messages_list = request_payload.get("messages")
        if not messages_list:
            return jsonify({
                "error": {
                    "message": "messages field is required",
                    "type": "invalid_request_error",
                    "code": "missing_messages"
                }
            }), 400
        
        # Core parameters
        model_name = request_payload.get("model")
        temperature_value = request_payload.get("temperature", 0.7)
        max_tokens_value = request_payload.get("max_tokens")
        stream_enabled = request_payload.get("stream", False)
        
        # Extract additional parameters (response_format, tools, etc.)
        additional_params = _extract_additional_parameters(request_payload)
        
        # Get services from app context
        llm_router_service = current_app.llm_router
        usage_tracker_service = current_app.usage_tracker
        
        # Track active requests
        effective_model = model_name or llm_router_service.default_model_name
        metrics_service.increment_active_requests(effective_model)
        
        try:
            if stream_enabled:
                return _handle_streaming_response(
                    llm_router_service,
                    usage_tracker_service,
                    messages_list,
                    model_name,
                    temperature_value,
                    max_tokens_value,
                    request_start_time,
                    additional_params
                )
            
            # Non-streaming response
            completion_response = llm_router_service.generate_chat_completion(
                messages=messages_list,
                model=model_name,
                temperature=temperature_value,
                max_tokens=max_tokens_value,
                stream=False,
                **additional_params
            )
            
            # Log the request
            _log_successful_request(
                usage_tracker_service,
                completion_response,
                effective_model,
                request_start_time
            )
            
            return jsonify(completion_response)
        
        finally:
            metrics_service.decrement_active_requests(effective_model)
    
    except ValueError as validation_error:
        _log_error_request(
            current_app.usage_tracker,
            model_name,
            400,
            str(validation_error),
            request_start_time
        )
        return jsonify({
            "error": {
                "message": str(validation_error),
                "type": "invalid_request_error",
                "code": "invalid_model"
            }
        }), 400
    
    except Exception as unexpected_error:
        _log_error_request(
            current_app.usage_tracker,
            model_name,
            500,
            str(unexpected_error),
            request_start_time
        )
        return jsonify({
            "error": {
                "message": f"Internal server error: {str(unexpected_error)}",
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500


def _log_successful_request(
    usage_tracker_service,
    completion_response: dict,
    model_name: str,
    request_start_time: float
):
    """Log a successful request to usage tracker and metrics."""
    usage_data = completion_response.get("usage", {})
    gateway_metadata = completion_response.get("gateway_metadata", {})
    
    latency_seconds = time.time() - request_start_time
    latency_ms = int(latency_seconds * 1000)
    
    # Log to usage tracker
    usage_tracker_service.log_request(
        model_name=model_name,
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
        cost_usd=usage_data.get("cost_usd"),
        latency_ms=latency_ms,
        status_code=200,
        cached=gateway_metadata.get("cached", False),
        api_key_hash=_get_api_key_hash(),
        provider_model=gateway_metadata.get("provider_model")
    )
    
    # Record metrics
    metrics_service.record_request(
        model_name=model_name,
        status="success",
        cached=gateway_metadata.get("cached", False),
        latency_seconds=latency_seconds,
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        cost_usd=usage_data.get("cost_usd")
    )


def _log_error_request(
    usage_tracker_service,
    model_name: str,
    status_code: int,
    error_message: str,
    request_start_time: float
):
    """Log a failed request."""
    latency_seconds = time.time() - request_start_time
    latency_ms = int(latency_seconds * 1000)
    
    usage_tracker_service.log_request(
        model_name=model_name or "unknown",
        latency_ms=latency_ms,
        status_code=status_code,
        api_key_hash=_get_api_key_hash(),
        error_message=error_message
    )
    
    metrics_service.record_request(
        model_name=model_name or "unknown",
        status="error",
        cached=False,
        latency_seconds=latency_seconds,
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=None
    )


def _handle_streaming_response(
    llm_router_service,
    usage_tracker_service,
    messages_list: list,
    model_name: str,
    temperature_value: float,
    max_tokens_value: int,
    request_start_time: float,
    additional_params=None
) -> Response:
    """Handle streaming response using Server-Sent Events."""
    
    effective_model = model_name or llm_router_service.default_model_name
    additional_params = additional_params or {}
    
    def generate_sse_stream():
        """Generator for SSE streaming."""
        total_completion_tokens = 0
        
        try:
            stream_generator = llm_router_service.generate_chat_completion(
                messages=messages_list,
                model=model_name,
                temperature=temperature_value,
                max_tokens=max_tokens_value,
                stream=True,
                **additional_params
            )
            
            for chunk_data in stream_generator:
                # Count tokens (approximate for streaming)
                content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    total_completion_tokens += len(content.split()) // 4 + 1
                
                sse_formatted_data = f"data: {json.dumps(chunk_data)}\n\n"
                yield sse_formatted_data
            
            # Send done signal
            yield "data: [DONE]\n\n"
            
            # Log streaming request (approximate tokens)
            latency_seconds = time.time() - request_start_time
            usage_tracker_service.log_request(
                model_name=effective_model,
                completion_tokens=total_completion_tokens,
                latency_ms=int(latency_seconds * 1000),
                status_code=200,
                api_key_hash=_get_api_key_hash()
            )
            
            metrics_service.record_request(
                model_name=effective_model,
                status="success",
                cached=False,
                latency_seconds=latency_seconds,
                prompt_tokens=0,
                completion_tokens=total_completion_tokens,
                cost_usd=None
            )
        
        except Exception as stream_error:
            error_payload = {
                "error": {
                    "message": str(stream_error),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            
            # Log error
            _log_error_request(
                usage_tracker_service,
                effective_model,
                500,
                str(stream_error),
                request_start_time
            )
        
        finally:
            metrics_service.decrement_active_requests(effective_model)
    
    return Response(
        stream_with_context(generate_sse_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
