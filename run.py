"""Entry point for LLM Gateway server."""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app import create_flask_application


flask_application = create_flask_application()


if __name__ == "__main__":
    server_port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    
    print(f"🚀 LLM Gateway starting on port {server_port}")
    print(f"📋 Available models: {list(flask_application.llm_router.available_models.keys())}")
    print(f"🔧 Debug mode: {debug_mode}")
    
    flask_application.run(
        host="0.0.0.0",
        port=server_port,
        debug=debug_mode
    )

