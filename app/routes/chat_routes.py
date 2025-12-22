"""Chat completion routes for LLM Gateway."""

import json

from flask import Blueprint, request, jsonify, Response, current_app, stream_with_context

from app.middleware.auth_middleware import require_api_key_authentication


chat_blueprint = Blueprint("chat", __name__)


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
    """
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
        
        model_name = request_payload.get("model")
        temperature_value = request_payload.get("temperature", 0.7)
        max_tokens_value = request_payload.get("max_tokens")
        stream_enabled = request_payload.get("stream", False)
        
        # Get LLM router from app context
        llm_router_service = current_app.llm_router
        
        if stream_enabled:
            return _handle_streaming_response(
                llm_router_service,
                messages_list,
                model_name,
                temperature_value,
                max_tokens_value
            )
        
        # Non-streaming response
        completion_response = llm_router_service.generate_chat_completion(
            messages=messages_list,
            model=model_name,
            temperature=temperature_value,
            max_tokens=max_tokens_value,
            stream=False
        )
        
        return jsonify(completion_response)
    
    except ValueError as validation_error:
        return jsonify({
            "error": {
                "message": str(validation_error),
                "type": "invalid_request_error",
                "code": "invalid_model"
            }
        }), 400
    
    except Exception as unexpected_error:
        return jsonify({
            "error": {
                "message": f"Internal server error: {str(unexpected_error)}",
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500


def _handle_streaming_response(
    llm_router_service,
    messages_list: list,
    model_name: str,
    temperature_value: float,
    max_tokens_value: int
) -> Response:
    """Handle streaming response using Server-Sent Events."""
    
    def generate_sse_stream():
        """Generator for SSE streaming."""
        try:
            stream_generator = llm_router_service.generate_chat_completion(
                messages=messages_list,
                model=model_name,
                temperature=temperature_value,
                max_tokens=max_tokens_value,
                stream=True
            )
            
            for chunk_data in stream_generator:
                sse_formatted_data = f"data: {json.dumps(chunk_data)}\n\n"
                yield sse_formatted_data
            
            # Send done signal
            yield "data: [DONE]\n\n"
        
        except Exception as stream_error:
            error_payload = {
                "error": {
                    "message": str(stream_error),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
    
    return Response(
        stream_with_context(generate_sse_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

