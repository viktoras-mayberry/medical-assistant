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
from pydantic import BaseModel, Field, validator
from elevenlabs.client import ElevenLabs
import speech_recognition as sr
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config import Config
from medical_llm import MedicalLLMClient, MedicalResponse
from medical_knowledge import MedicalKnowledgeEngine
from medical_analytics import MedicalAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration
Config.validate()

# Initialize ElevenLabs client
eleven = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)

# Initialize medical components
medical_llm = MedicalLLMClient()
medical_knowledge = MedicalKnowledgeEngine()
medical_analytics = MedicalAnalytics()

# Initialize FastAPI app
app = FastAPI(
    title="Medical Chat AI Assistant",
    description="A voice-enabled medical assistant powered by AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class VoiceChatRequest(BaseModel):
    patient_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AudioTranscriber:
    """Handles audio transcription using speech recognition"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio using speech_recognition library"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                # Listen for audio
                audio = self.recognizer.listen(source)
                
            # Use Google's free speech recognition
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.UnknownValueError:
            return "I couldn't understand the audio. Please speak clearly and try again."
        except sr.RequestError as e:
            return f"Could not request results from speech recognition service; {e}"
        except Exception as e:
            return f"Error processing audio: {e}"

# Initialize audio transcriber
audio_transcriber = AudioTranscriber()

@app.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    patient_id: Optional[str] = None,
    context: Optional[str] = None
):
    """
    Accepts voice note, converts to text, processes with medical AI, and returns voice response
    """
    start_time = datetime.now()
    
    try:
        # Parse context if provided
        parsed_context = json.loads(context) if context else {}
        
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        # Convert speech to text using speech_recognition
        user_text = audio_transcriber.transcribe_audio(temp_audio_path)
        
        # Get medical AI response
        medical_response = await medical_llm.process_query(
            query=user_text,
            patient_id=patient_id,
            context=parsed_context
        )
        
        # Log analytics
        response_time = (datetime.now() - start_time).total_seconds()
        await medical_analytics.log_interaction(
            interaction_type="voice_chat",
            query=user_text,
            response=medical_response.response,
            patient_id=patient_id,
            risk_level=medical_response.risk_level,
            response_time=response_time,
            metadata={
                "audio_file_size": len(content),
                "context": parsed_context,
                "confidence_score": medical_response.confidence_score,
                "emergency_detected": medical_response.is_emergency,
                "sources": medical_response.sources
            }
        )
        
        # Convert AI response to speech
        tts_response = eleven.text_to_speech.convert(
            text=medical_response.response,
            voice_id=Config.DEFAULT_VOICE_ID,
            model_id=Config.TTS_MODEL,
            output_format=Config.AUDIO_FORMAT
        )
        
        audio_bytes = b"".join(tts_response)
        
        # Clean up temporary file
        os.unlink(temp_audio_path)
        
        return StreamingResponse(
            BytesIO(audio_bytes), 
            media_type="audio/mpeg",
            headers={
                "X-Transcribed-Text": user_text,
                "X-AI-Response": medical_response.response,
                "X-Risk-Level": medical_response.risk_level,
                "X-Emergency": str(medical_response.is_emergency),
                "X-Confidence": str(medical_response.confidence_score)
            }
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_audio_path' in locals():
            try:
                os.unlink(temp_audio_path)
            except:
                pass
        
        # Log error
        await medical_analytics.log_interaction(
            interaction_type="voice_chat_error",
            query=user_text if 'user_text' in locals() else "Unknown",
            response=str(e),
            patient_id=patient_id,
            risk_level="unknown",
            response_time=(datetime.now() - start_time).total_seconds(),
            metadata={"error": str(e)}
        )
        
        raise HTTPException(status_code=500, detail=f"Voice processing error: {e}")

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
        
        # Log analytics
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
        # Log error
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

@app.post("/speak")
async def speak(request: ChatRequest):
    """
    Convert text to speech
    """
    try:
        response = eleven.text_to_speech.convert(
            text=request.message,
            voice_id=Config.DEFAULT_VOICE_ID,
            model_id=Config.TTS_MODEL,
            output_format=Config.AUDIO_FORMAT
        )
        audio_bytes = b"".join(response)
        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

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
            "speech-to-text", 
            "text-to-speech", 
            "medical-ai", 
            "knowledge-engine",
            "analytics"
        ]
    }

    