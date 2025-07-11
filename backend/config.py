"""Configuration settings for the Medical AI Assistant."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Voice Settings
    DEFAULT_VOICE_ID = os.getenv("VOICE_ID", "XMBhZBHTRP7bO9ojOPI0")
    
    # Medical LLM Settings
    MODEL = os.getenv("MODEL", "gpt-4")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    
    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
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
        
        # OpenAI API key is optional for testing (will use mock responses)
        if not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set. Using mock responses for medical queries.")
    
    @classmethod
    def get_cors_origins(cls):
        """Get allowed CORS origins."""
        origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        return origins + [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "*"  # Allow all origins for development
        ]
