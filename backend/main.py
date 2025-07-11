import os
import tempfile
import re
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from elevenlabs.client import ElevenLabs
import speech_recognition as sr

from config import Config
from medical_llm import MedicalLLMClient, MedicalResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration
Config.validate()

# Initialize ElevenLabs client
eleven = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)

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

class LLMClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or "YOUR_API_KEY"
        # Sample initialization, replace with actual model client initialization if needed
        # self.model = SomeLLMClient(api_key=self.api_key)
        self.recognizer = sr.Recognizer()
        """Load knowledge base from text file"""
FAQ - AI Travel Assistant

Q: What services do you offer?
A: I'm an AI travel assistant that can help you with trip planning, destination recommendations, booking assistance, travel tips, and answering questions about various travel destinations.

Q: How can you help me plan my trip?
A: I can help you create itineraries, suggest destinations based on your preferences, recommend accommodations, provide information about local attractions, weather conditions, and travel requirements.

Q: What destinations do you know about?
A: I have knowledge about destinations worldwide, including popular tourist spots, hidden gems, cultural sites, adventure locations, and business travel destinations.

Q: Can you help with booking flights and hotels?
A: I can provide guidance on booking platforms, compare options, and give tips on finding the best deals, but I cannot make actual bookings for you.

Q: What about travel documents and visa requirements?
A: I can provide general information about visa requirements, passport validity, and travel documents needed for different countries, but always verify with official sources.

Q: Do you provide real-time information?
A: I provide general travel information and tips. For real-time updates like flight status, weather, or current events, please check official sources.

Q: How do I use the voice feature?
A: Simply click and hold the microphone button to record your voice message. Release to send. I'll respond with voice as well!

Q: What languages do you support?
A: Currently, I primarily support English, but I can understand and respond to basic queries in several languages.

Q: How can I find cheap flights?
A: I recommend using flight comparison sites, being flexible with dates, considering nearby airports, and booking in advance for better deals.

Q: What should I pack for my trip?
A: Pack based on your destination's weather, planned activities, and duration. Always include essentials like passport, medications, and weather-appropriate clothing.

Q: How do I stay safe while traveling?
A: Research your destination, keep copies of important documents, stay aware of your surroundings, use reputable transportation, and avoid risky areas.

Q: What about travel insurance?
A: I recommend getting travel insurance that covers medical emergencies, trip cancellations, and lost luggage. Compare different providers for the best coverage.

Q: Can you help with cultural tips?
A: Yes! I can provide information about local customs, etiquette, dining traditions, and cultural norms to help you travel respectfully.

Q: What if I have dietary restrictions?
A: I can help you find restaurants that accommodate special diets, provide translations for dietary needs, and suggest safe food options for various dietary requirements.

Q: How do I deal with jet lag?
A: I can provide strategies for minimizing jet lag including adjusting sleep schedules, staying hydrated, and managing light exposure.

Q: What are some budget travel tips?
A: Consider traveling during off-peak seasons, use public transportation, stay in hostels or budget accommodations, and look for free activities and attractions.
                """
                with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                    f.write(default_knowledge)
                return default_knowledge
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return "I'm an AI travel assistant ready to help you with your travel needs."

    def parse_qa_pairs(self):
        """Parse Q&A pairs from knowledge base"""
        qa_pairs = []
        lines = self.knowledge.split('\n')
        current_q = ""
        current_a = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q: '):
                if current_q and current_a:
                    qa_pairs.append((current_q, current_a))
                current_q = line[3:].strip()
                current_a = ""
            elif line.startswith('A: '):
                current_a = line[3:].strip()
            elif current_a and line and not line.startswith('='):
                current_a += " " + line
        
        if current_q and current_a:
            qa_pairs.append((current_q, current_a))
        
        return qa_pairs

    def query_llm(self, user_message):
        """Query the LLM model to get medical information"""
        user_message = user_message.strip()
        response = "Sorry, I'm unable to process your request right now."
        try:
            # Mock query to LLM model, replace with actual API call
            # response = self.model.query(user_message)
            print(f"Querying LLM with message: {user_message}")
            return "This is a mock response from the LLM model"
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return response
        
        # Check for keyword matches first
        for category, words in keywords.items():
            if any(word in user_message for word in words):
                for question, answer in self.qa_pairs:
                    if any(word in question.lower() for word in words):
                        score = SequenceMatcher(None, user_message, question.lower()).ratio()
                        if score > best_score:
                            best_score = score
                            best_match = (question, answer)
        
        # If no keyword match, use general similarity
        if best_score < 0.3:
            for question, answer in self.qa_pairs:
                score = SequenceMatcher(None, user_message, question.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = (question, answer)
        
        return best_match, best_score

    def get_response(self, user_message):
        """Get response using the LLM model"""
        try:
            # Clean and normalize the message
            user_message = re.sub(r'[^\w\s]', '', user_message).strip()
            
            if not user_message:
                return "I didn't catch that. Could you please repeat your question?"
            
            # Query LLM model for a response
            llm_response = self.query_llm(user_message)
            return llm_response
            
        except Exception as e:
            return "I'm having trouble processing your request right now. Please try again."

    def generate_generic_response(self, user_message):
        """Generate generic travel-related response"""
        travel_keywords = ['travel', 'trip', 'vacation', 'holiday', 'destination', 'visit', 'go']
        
        if any(keyword in user_message.lower() for keyword in travel_keywords):
            return "That's a great travel question! I'd be happy to help you with travel planning, destinations, booking tips, or any other travel-related needs. Could you be more specific about what you'd like to know?"
        
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in user_message.lower() for greeting in greetings):
            return "Hello! Welcome to your AI travel assistant. I'm here to help you with trip planning, destination recommendations, travel tips, and more. What would you like to know about travel?"
        
        # Check for gratitude
        thanks = ['thank', 'thanks', 'appreciate']
        if any(word in user_message.lower() for word in thanks):
            return "You're welcome! I'm glad I could help. Feel free to ask me anything else about travel planning or destinations."
        
        # Default response
        return "I'm your travel assistant and I'm here to help! I can assist with trip planning, destination recommendations, booking guidance, travel tips, and much more. What would you like to know about travel?"

    def transcribe_audio(self, audio_file_path):
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

# Initialize knowledge base
kb = LLMClient()

@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    """
    Accepts voice note, converts to text, processes with NLP, and returns voice response
    """
    try:
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        # Convert speech to text using speech_recognition
        user_text = kb.transcribe_audio(temp_audio_path)
        
        # Get AI response using knowledge base
        ai_response = kb.get_response(user_text)
        
        # Convert AI response to speech
        tts_response = eleven.text_to_speech.convert(
            text=ai_response,
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
                "X-AI-Response": ai_response
            }
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_audio_path' in locals():
            try:
                os.unlink(temp_audio_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Voice processing error: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Text-based chat endpoint (fallback)
    """
    try:
        ai_response = kb.get_response(request.message)
        return {"response": ai_response}
    except Exception as e:
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

@app.get("/")
def root():
    return {"message": "Voice Chat AI Assistant API is running."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "services": ["speech-to-text", "text-to-speech", "ai-chat"]}

    