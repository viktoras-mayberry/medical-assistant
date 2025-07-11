"""Configuration settings for the Voice Chat AI Assistant."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    # Voice Settings
    DEFAULT_VOICE_ID = os.getenv("VOICE_ID", "XMBhZBHTRP7bO9ojOPI0")
    
    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # File Paths
    KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base.txt")
    
    # Audio Settings
    AUDIO_FORMAT = "mp3_22050_32"
    TTS_MODEL = "eleven_multilingual_v2"
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY is required. Please set it in your .env file.")
        
        if not os.path.exists(cls.KNOWLEDGE_BASE_PATH):
            raise FileNotFoundError(f"Knowledge base file not found at: {cls.KNOWLEDGE_BASE_PATH}")
    
    @classmethod
    def get_cors_origins(cls):
        """Get allowed CORS origins."""
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "*"  # Allow all origins for development
        ]
