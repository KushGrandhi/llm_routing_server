"""Authentication middleware for API key validation."""

import os
from functools import wraps

from flask import request, jsonify


def require_api_key_authentication(handler_function):
    """
    Decorator to require API key authentication for routes.
    
    Checks the Authorization header for a valid Bearer token
    against the GATEWAY_API_KEYS environment variable.
    """
    @wraps(handler_function)
    def decorated_authentication_handler(*args, **kwargs):
        # Get allowed API keys from environment
        allowed_api_keys_string = os.getenv("GATEWAY_API_KEYS", "")
        allowed_api_keys_list = [
            key.strip() 
            for key in allowed_api_keys_string.split(",") 
            if key.strip()
        ]
        
        # If no keys configured, allow all requests (development mode)
        if not allowed_api_keys_list:
            return handler_function(*args, **kwargs)
        
        # Check Authorization header
        authorization_header = request.headers.get("Authorization", "")
        
        if not authorization_header:
            return jsonify({
                "error": {
                    "message": "Missing Authorization header",
                    "type": "authentication_error",
                    "code": "missing_api_key"
                }
            }), 401
        
        # Extract Bearer token
        if not authorization_header.startswith("Bearer "):
            return jsonify({
                "error": {
                    "message": "Invalid Authorization header format. Use: Bearer <api_key>",
                    "type": "authentication_error",
                    "code": "invalid_format"
                }
            }), 401
        
        provided_api_key = authorization_header[7:]  # Remove "Bearer " prefix
        
        if provided_api_key not in allowed_api_keys_list:
            return jsonify({
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                    "code": "invalid_api_key"
                }
            }), 401
        
        return handler_function(*args, **kwargs)
    
    return decorated_authentication_handler

