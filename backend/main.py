import os
import tempfile
import re
import json
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
# Voice-related imports removed for non-voice backend
# import speech_recognition as sr
# import elevenlabs
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config import Config
from opensource_medical_llm import OpenSourceMedicalLLM, MedicalResponse
from opensource_voice import OpenSourceVoiceProcessor
from medical_knowledge import MedicalKnowledgeEngine
from medical_analytics import MedicalAnalytics
from database import init_db, get_db
from auth import authenticate_user, create_access_token, get_current_active_user, get_password_hash
from schemas import Token, UserCreate, UserResponse, LoginRequest, UserUpdate, InteractionCreate, SubscriptionCreate, SubscriptionUpdate
from models import User
from fastapi import status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration
Config.validate()

# Voice features removed

# Initialize open-source medical components
medical_llm = OpenSourceMedicalLLM(model_name="microsoft/DialoGPT-medium")
voice_processor = OpenSourceVoiceProcessor(whisper_model="base")
try:
    medical_knowledge = MedicalKnowledgeEngine()
except Exception as e:
    logger.warning(f"Medical knowledge engine not available: {e}")
    medical_knowledge = None

try:
    medical_analytics = MedicalAnalytics()
except Exception as e:
    logger.warning(f"Medical analytics not available: {e}")
    medical_analytics = None

# Initialize FastAPI app
app = FastAPI(
    title="Medical Chat AI Assistant",
    description="A voice-enabled medical assistant powered by AI",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize database
init_db()

@app.post("/auth/register", response_model=UserResponse)
def register(user_create: UserCreate, db: Session = Depends(get_db)):
    """
    Register new user.
    """
    user = db.query(User).filter(User.email == user_create.email).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered.",
        )
    
    hashed_password = get_password_hash(user_create.password)
    user = User(email=user_create.email, password_hash=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Login user and return access token.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Incorrect username or password",
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user detail.
    """
    return current_user

class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

# Voice-related classes removed

# Voice chat endpoint removed

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Text-based medical chat endpoint
    """
    start_time = datetime.now()
    
    try:
        # Get medical AI response
        medical_response = await medical_llm.process_query(
            query=request.message,
            patient_id=request.patient_id,
            context=request.context or {}
        )
        
        # Log analytics if available
        if medical_analytics:
            response_time = (datetime.now() - start_time).total_seconds()
            await medical_analytics.log_interaction(
                interaction_type="text_chat",
                query=request.message,
                response=medical_response.response,
                patient_id=request.patient_id,
                risk_level=medical_response.risk_level,
                response_time=response_time,
                metadata={
                    "context": request.context,
                    "confidence_score": medical_response.confidence_score,
                    "emergency_detected": medical_response.is_emergency,
                    "sources": medical_response.sources
                }
            )
        
        return {
            "response": medical_response.response,
            "risk_level": medical_response.risk_level,
            "is_emergency": medical_response.is_emergency,
            "confidence_score": medical_response.confidence_score,
            "sources": medical_response.sources,
            "recommendations": medical_response.recommendations
        }
        
    except Exception as e:
        # Log error if analytics available
        if medical_analytics:
            await medical_analytics.log_interaction(
                interaction_type="text_chat_error",
                query=request.message,
                response=str(e),
                patient_id=request.patient_id,
                risk_level="unknown",
                response_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )
        
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

# Text-to-speech endpoint removed

# Additional medical endpoints
@app.get("/medical/symptoms")
async def get_symptoms():
    """Get available symptoms for assessment"""
    return medical_knowledge.get_symptoms()

@app.post("/medical/assess")
async def assess_symptoms(symptoms: List[str], patient_id: Optional[str] = None):
    """Assess symptoms and provide medical insights"""
    start_time = datetime.now()
    
    try:
        assessment = medical_knowledge.assess_symptoms(symptoms)
        
        # Log assessment
        await medical_analytics.log_interaction(
            interaction_type="symptom_assessment",
            query=f"Symptoms: {', '.join(symptoms)}",
            response=str(assessment),
            patient_id=patient_id,
            risk_level=assessment.get('risk_level', 'low'),
            response_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "symptoms": symptoms,
                "assessment": assessment
            }
        )
        
        return assessment
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment error: {e}")

@app.post("/medical/drug-interactions")
async def check_drug_interactions(drugs: List[str], patient_id: Optional[str] = None):
    """Check for drug interactions"""
    start_time = datetime.now()
    
    try:
        interactions = medical_knowledge.check_drug_interactions(drugs)
        
        # Log interaction check
        await medical_analytics.log_interaction(
            interaction_type="drug_interaction_check",
            query=f"Drugs: {', '.join(drugs)}",
            response=str(interactions),
            patient_id=patient_id,
            risk_level=interactions.get('risk_level', 'low'),
            response_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "drugs": drugs,
                "interactions": interactions
            }
        )
        
        return interactions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drug interaction check error: {e}")

# Voice processing endpoints
@app.post("/voice/transcribe")
async def transcribe_voice(audio_file: UploadFile = File(...)):
    """Transcribe audio file to text"""
    try:
        # Read audio data
        audio_data = await audio_file.read()
        
        # Transcribe using voice processor
        result = await voice_processor.process_voice_query(audio_data)
        
        if result["success"]:
            return {
                "transcription": result["transcription"],
                "confidence": result["confidence"],
                "language": result["language"],
                "model": result["model"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

@app.post("/voice/synthesize")
async def synthesize_voice(text: str, language: str = "en", format: str = "mp3"):
    """Convert text to speech"""
    try:
        # Synthesize speech
        audio_data = await voice_processor.synthesize_speech(text, format)
        
        # Return audio response
        return StreamingResponse(
            BytesIO(audio_data),
            media_type=f"audio/{format}",
            headers={"Content-Disposition": f"attachment; filename=speech.{format}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {e}")

@app.post("/voice/chat")
async def voice_chat(audio_file: UploadFile = File(...)):
    """Complete voice chat pipeline: transcribe -> process -> synthesize"""
    try:
        # Transcribe audio
        audio_data = await audio_file.read()
        transcription_result = await voice_processor.process_voice_query(audio_data)
        
        if not transcription_result["success"]:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        transcription = transcription_result["transcription"]
        
        # Process with medical LLM
        medical_response = await medical_llm.process_query(transcription)
        
        # Synthesize response
        response_audio = await voice_processor.synthesize_speech(medical_response.response)
        
        # Return both text and audio response
        return {
            "transcription": transcription,
            "response": medical_response.response,
            "risk_level": medical_response.risk_level,
            "is_emergency": medical_response.is_emergency,
            "confidence_score": medical_response.confidence_score,
            "sources": medical_response.sources,
            "recommendations": medical_response.recommendations,
            "audio_available": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice chat error: {e}")

# Model management endpoints
@app.get("/models/available")
async def get_available_models():
    """Get available AI models"""
    return {
        "llm_models": medical_llm.list_available_models(),
        "voice_models": voice_processor.get_available_models(),
        "current_llm": medical_llm.get_model_info()
    }

@app.post("/models/llm/change")
async def change_llm_model(model_name: str):
    """Change the LLM model"""
    try:
        # Create new LLM instance with different model
        global medical_llm
        medical_llm = OpenSourceMedicalLLM(model_name=model_name)
        
        return {
            "success": True,
            "message": f"LLM model changed to {model_name}",
            "model_info": medical_llm.get_model_info()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model change error: {e}")

@app.post("/models/voice/whisper/change")
async def change_whisper_model(model_name: str):
    """Change the Whisper model"""
    try:
        success = voice_processor.change_whisper_model(model_name)
        
        if success:
            return {
                "success": True,
                "message": f"Whisper model changed to {model_name}",
                "model_info": voice_processor.get_system_info()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to change Whisper model")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model change error: {e}")

@app.get("/models/system-info")
async def get_system_info():
    """Get system information"""
    return {
        "llm_info": medical_llm.get_model_info(),
        "voice_info": voice_processor.get_system_info(),
        "knowledge_base_available": medical_knowledge is not None,
        "analytics_available": medical_analytics is not None
    }

@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """Get medical analytics dashboard data"""
    try:
        return await medical_analytics.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")

@app.get("/analytics/patient/{patient_id}")
async def get_patient_analytics(patient_id: str):
    """Get analytics for a specific patient"""
    try:
        return await medical_analytics.get_patient_analytics(patient_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patient analytics error: {e}")

@app.get("/")
def root():
    return {"message": "Medical AI Assistant API is running."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "services": [
            "medical-ai", 
            "knowledge-engine",
            "analytics"
        ]
    }

    