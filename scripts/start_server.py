#!/usr/bin/env python3
"""
Start the Voice Chat AI Assistant server.
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from config import Config

def main():
    """Start the server with proper configuration."""
    try:
        # Validate configuration before starting
        Config.validate()
        
        print("🎙️ Starting Voice Chat AI Assistant Server...")
        print(f"📍 Server will run on: http://{Config.HOST}:{Config.PORT}")
        print(f"📁 Knowledge base: {Config.KNOWLEDGE_BASE_PATH}")
        print(f"🔊 Voice ID: {Config.DEFAULT_VOICE_ID}")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the server
        uvicorn.run(
            "main:app",
            host=Config.HOST,
            port=Config.PORT,
            reload=True,
            reload_dirs=[str(backend_dir)],
            app_dir=str(backend_dir)
        )
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
