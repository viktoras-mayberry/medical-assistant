"""Advanced Configuration Settings for the Medical AI Assistant."""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class MedicalAIConfig:
    """Advanced Medical AI Application Configuration."""
    
    # Application Metadata
    APP_NAME = "Advanced Medical AI Assistant"
    APP_VERSION = "2.0.0"
    APP_DESCRIPTION = "State-of-the-art medical AI assistant with advanced capabilities"
    
    # API Keys (Optional for open-source models)
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # Advanced Medical AI Models
    PRIMARY_MEDICAL_MODEL = os.getenv("PRIMARY_MEDICAL_MODEL", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "microsoft/DialoGPT-medium")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    
    # Voice Processing Settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    TTS_ENGINE = os.getenv("TTS_ENGINE", "gtts")  # gtts, espeak, festival
    DEFAULT_VOICE_ID = os.getenv("VOICE_ID", "XMBhZBHTRP7bO9ojOPI0")
    VOICE_LANGUAGE = os.getenv("VOICE_LANGUAGE", "en")
    
    # Medical AI Parameters
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
    EMERGENCY_THRESHOLD = float(os.getenv("EMERGENCY_THRESHOLD", 0.9))
    
    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    WORKERS = int(os.getenv("WORKERS", 4))
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///medical_ai.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Vector Database Settings
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chromadb")  # chromadb, pinecone, weaviate
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./medical_vector_db")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 768))
    
    # File Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    MEDICAL_KNOWLEDGE_PATH = DATA_DIR / "medical_training_data.json"
    DRUG_INTERACTIONS_PATH = DATA_DIR / "drug_interactions.json"
    ICD_CODES_PATH = DATA_DIR / "icd_codes.json"
    
    # Audio Settings
    AUDIO_FORMAT = "mp3_22050_32"
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
    AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", 1))
    MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", 300))  # seconds
    
    # Security Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", 24))
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 3600))  # seconds
    
    # Medical Safety Settings
    ENABLE_EMERGENCY_DETECTION = os.getenv("ENABLE_EMERGENCY_DETECTION", "true").lower() == "true"
    ENABLE_DRUG_INTERACTION_CHECK = os.getenv("ENABLE_DRUG_INTERACTION_CHECK", "true").lower() == "true"
    ENABLE_LITERATURE_SEARCH = os.getenv("ENABLE_LITERATURE_SEARCH", "true").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Performance Settings
    ENABLE_GPU = os.getenv("ENABLE_GPU", "true").lower() == "true"
    MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", 2))  # GB
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
    
    # Medical Specialties Configuration
    MEDICAL_SPECIALTIES = {
        "cardiology": {
            "keywords": ["heart", "cardiac", "chest pain", "blood pressure", "arrhythmia"],
            "urgency_multiplier": 1.5,
            "required_tests": ["ECG", "echocardiogram", "cardiac enzymes"]
        },
        "neurology": {
            "keywords": ["brain", "neurological", "seizure", "stroke", "headache"],
            "urgency_multiplier": 1.4,
            "required_tests": ["CT scan", "MRI", "EEG"]
        },
        "pulmonology": {
            "keywords": ["lung", "respiratory", "breathing", "cough", "asthma"],
            "urgency_multiplier": 1.3,
            "required_tests": ["chest X-ray", "pulmonary function test"]
        },
        "emergency": {
            "keywords": ["emergency", "urgent", "severe", "critical", "call 911"],
            "urgency_multiplier": 2.0,
            "required_tests": ["immediate assessment", "vital signs"]
        }
    }
    
    # Risk Assessment Configuration
    RISK_LEVELS = {
        "low": {
            "threshold": 0.3,
            "color": "green",
            "action": "routine_care"
        },
        "moderate": {
            "threshold": 0.6,
            "color": "yellow",
            "action": "schedule_appointment"
        },
        "high": {
            "threshold": 0.8,
            "color": "orange",
            "action": "urgent_care"
        },
        "critical": {
            "threshold": 0.9,
            "color": "red",
            "action": "emergency_care"
        }
    }
    
    # Model Performance Monitoring
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", 300))  # seconds
    
    # Feature Flags
    FEATURES = {
        "voice_chat": os.getenv("FEATURE_VOICE_CHAT", "true").lower() == "true",
        "drug_interactions": os.getenv("FEATURE_DRUG_INTERACTIONS", "true").lower() == "true",
        "literature_search": os.getenv("FEATURE_LITERATURE_SEARCH", "true").lower() == "true",
        "symptom_checker": os.getenv("FEATURE_SYMPTOM_CHECKER", "true").lower() == "true",
        "emergency_detection": os.getenv("FEATURE_EMERGENCY_DETECTION", "true").lower() == "true",
        "analytics": os.getenv("FEATURE_ANALYTICS", "true").lower() == "true",
        "multi_language": os.getenv("FEATURE_MULTI_LANGUAGE", "false").lower() == "true"
    }
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        # Create directories if they don't exist
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate critical settings
        if cls.SECRET_KEY == "your-secret-key-here":
            print("⚠️  Warning: Using default SECRET_KEY. Please set a secure secret key.")
        
        # Optional API key validation
        if not cls.ELEVENLABS_API_KEY and cls.TTS_ENGINE == "elevenlabs":
            print("⚠️  Warning: ELEVENLABS_API_KEY not set. Voice synthesis may be limited.")
        
        if not cls.OPENAI_API_KEY:
            print("ℹ️  Info: Using open-source models only. No OpenAI API key required.")
        
        # Validate model paths
        if not cls.MEDICAL_KNOWLEDGE_PATH.exists():
            print(f"⚠️  Warning: Medical knowledge file not found at {cls.MEDICAL_KNOWLEDGE_PATH}")
        
        print(f"✅ Configuration validated for {cls.APP_NAME} v{cls.APP_VERSION}")
    
    @classmethod
    def get_cors_origins(cls) -> List[str]:
        """Get allowed CORS origins."""
        origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        return origins + [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5173",  # Vite dev server
            "http://localhost:4000",  # Next.js dev server
        ]
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": cls.DATABASE_URL,
            "echo": cls.DEBUG,
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
            "pool_recycle": 3600
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "primary_model": cls.PRIMARY_MEDICAL_MODEL,
            "fallback_model": cls.FALLBACK_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "enable_gpu": cls.ENABLE_GPU,
            "batch_size": cls.BATCH_SIZE
        }
    
    @classmethod
    def get_security_config(cls) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "secret_key": cls.SECRET_KEY,
            "jwt_algorithm": cls.JWT_ALGORITHM,
            "jwt_expiration_hours": cls.JWT_EXPIRATION_HOURS,
            "rate_limit_requests": cls.RATE_LIMIT_REQUESTS,
            "rate_limit_window": cls.RATE_LIMIT_WINDOW
        }
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "app_name": cls.APP_NAME,
            "app_version": cls.APP_VERSION,
            "features": cls.FEATURES,
            "model_config": cls.get_model_config(),
            "security_config": cls.get_security_config(),
            "specialties": cls.MEDICAL_SPECIALTIES,
            "risk_levels": cls.RISK_LEVELS
        }

# Backward compatibility
Config = MedicalAIConfig
