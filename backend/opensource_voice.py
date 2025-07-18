"""
Open-Source Voice Processing System
===================================

This module provides voice processing capabilities using:
- OpenAI Whisper for speech-to-text (runs locally)
- Google Text-to-Speech (gTTS) for text-to-speech
- No external API dependencies for core functionality
"""

import os
import io
import logging
import tempfile
from typing import Optional, Union, BinaryIO
from pathlib import Path
import asyncio

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper not available. Install with: pip install openai-whisper")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("gTTS not available. Install with: pip install gtts")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("SpeechRecognition not available. Install with: pip install speechrecognition")

try:
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Pydub not available. Install with: pip install pydub")

logger = logging.getLogger(__name__)

class OpenSourceVoiceProcessor:
    """Complete open-source voice processing system"""
    
    def __init__(self, 
                 whisper_model: str = "base",
                 tts_language: str = "en",
                 tts_slow: bool = False):
        """
        Initialize voice processor
        
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            tts_language: Language for text-to-speech
            tts_slow: Whether to use slow speech
        """
        self.whisper_model_name = whisper_model
        self.tts_language = tts_language
        self.tts_slow = tts_slow
        
        # Models
        self.whisper_model = None
        self.speech_recognizer = None
        
        # Initialize components
        self._initialize_whisper()
        self._initialize_speech_recognition()
        
        logger.info("Voice processor initialized")
    
    def _initialize_whisper(self):
        """Initialize Whisper model for speech-to-text"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available - STT will be limited")
            return
        
        try:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
    
    def _initialize_speech_recognition(self):
        """Initialize SpeechRecognition as fallback"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.warning("SpeechRecognition not available - using Whisper only")
            return
        
        try:
            self.speech_recognizer = sr.Recognizer()
            logger.info("SpeechRecognition initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SpeechRecognition: {e}")
            self.speech_recognizer = None
    
    async def transcribe_audio(self, 
                             audio_data: Union[bytes, str, BinaryIO],
                             use_whisper: bool = True) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio data (bytes, file path, or file-like object)
            use_whisper: Whether to use Whisper (True) or SpeechRecognition (False)
            
        Returns:
            Transcribed text
        """
        if use_whisper and self.whisper_model:
            return await self._transcribe_with_whisper(audio_data)
        elif self.speech_recognizer:
            return await self._transcribe_with_speech_recognition(audio_data)
        else:
            raise RuntimeError("No speech-to-text engine available")
    
    async def _transcribe_with_whisper(self, audio_data: Union[bytes, str, BinaryIO]) -> str:
        """Transcribe using Whisper model"""
        try:
            # Handle different input types
            if isinstance(audio_data, bytes):
                # Save bytes to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    audio_path = tmp_file.name
            elif isinstance(audio_data, str):
                # File path
                audio_path = audio_data
            else:
                # File-like object
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(audio_data.read())
                    audio_path = tmp_file.name
            
            # Transcribe with Whisper
            result = await asyncio.to_thread(
                self.whisper_model.transcribe,
                audio_path,
                language=self.tts_language if self.tts_language != 'en' else None
            )
            
            # Clean up temporary file
            if isinstance(audio_data, (bytes, BinaryIO)):
                try:
                    os.unlink(audio_path)
                except:
                    pass
            
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            # Fallback to SpeechRecognition
            if self.speech_recognizer:
                return await self._transcribe_with_speech_recognition(audio_data)
            return ""
    
    async def _transcribe_with_speech_recognition(self, audio_data: Union[bytes, str, BinaryIO]) -> str:
        """Transcribe using SpeechRecognition library"""
        try:
            # Handle different input types
            if isinstance(audio_data, bytes):
                # Convert bytes to AudioFile
                audio_file = io.BytesIO(audio_data)
            elif isinstance(audio_data, str):
                # File path
                audio_file = audio_data
            else:
                # File-like object
                audio_file = audio_data
            
            # Use SpeechRecognition
            with sr.AudioFile(audio_file) as source:
                audio = self.speech_recognizer.record(source)
            
            # Try Google Speech Recognition (free)
            try:
                text = await asyncio.to_thread(
                    self.speech_recognizer.recognize_google,
                    audio,
                    language=self.tts_language
                )
                return text
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return ""
            except sr.RequestError as e:
                logger.error(f"Speech recognition request failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"SpeechRecognition transcription failed: {e}")
            return ""
    
    async def synthesize_speech(self, text: str, output_format: str = "mp3") -> bytes:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            output_format: Output audio format ('mp3', 'wav')
            
        Returns:
            Audio data as bytes
        """
        if not GTTS_AVAILABLE:
            raise RuntimeError("gTTS not available for text-to-speech")
        
        try:
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=self.tts_language,
                slow=self.tts_slow
            )
            
            # Generate audio to bytes
            audio_buffer = io.BytesIO()
            await asyncio.to_thread(tts.write_to_fp, audio_buffer)
            audio_buffer.seek(0)
            
            audio_data = audio_buffer.getvalue()
            
            # Convert format if needed
            if output_format.lower() == "wav" and PYDUB_AVAILABLE:
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                audio_data = wav_buffer.getvalue()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Text-to-speech synthesis failed: {e}")
            raise
    
    def get_available_models(self) -> dict:
        """Get available Whisper models"""
        models = {
            "whisper": {
                "tiny": {"size": "~39MB", "speed": "~32x", "accuracy": "Good"},
                "base": {"size": "~74MB", "speed": "~16x", "accuracy": "Better"},
                "small": {"size": "~244MB", "speed": "~6x", "accuracy": "Good"},
                "medium": {"size": "~769MB", "speed": "~2x", "accuracy": "Very Good"},
                "large": {"size": "~1550MB", "speed": "~1x", "accuracy": "Best"}
            },
            "current_model": self.whisper_model_name,
            "whisper_available": WHISPER_AVAILABLE,
            "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
            "gtts_available": GTTS_AVAILABLE
        }
        return models
    
    def change_whisper_model(self, model_name: str) -> bool:
        """Change Whisper model"""
        if not WHISPER_AVAILABLE:
            return False
        
        try:
            logger.info(f"Changing Whisper model to: {model_name}")
            self.whisper_model = whisper.load_model(model_name)
            self.whisper_model_name = model_name
            logger.info("Whisper model changed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to change Whisper model: {e}")
            return False
    
    def get_supported_languages(self) -> list:
        """Get supported languages for TTS"""
        # Common gTTS supported languages
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'th', 'vi', 'id', 'nl', 'pl', 'sv', 'da', 'no'
        ]
    
    def set_language(self, language: str) -> bool:
        """Set language for TTS"""
        if language in self.get_supported_languages():
            self.tts_language = language
            logger.info(f"Language set to: {language}")
            return True
        else:
            logger.warning(f"Language {language} not supported")
            return False
    
    async def process_voice_query(self, audio_data: Union[bytes, str, BinaryIO]) -> dict:
        """
        Complete voice processing pipeline
        
        Args:
            audio_data: Audio input
            
        Returns:
            Dictionary with transcription and metadata
        """
        try:
            # Transcribe audio
            transcription = await self.transcribe_audio(audio_data)
            
            if not transcription:
                return {
                    "success": False,
                    "error": "Could not transcribe audio",
                    "transcription": "",
                    "confidence": 0.0
                }
            
            return {
                "success": True,
                "transcription": transcription,
                "confidence": 0.9,  # Placeholder confidence
                "language": self.tts_language,
                "model": self.whisper_model_name
            }
            
        except Exception as e:
            logger.error(f"Voice processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcription": "",
                "confidence": 0.0
            }
    
    def get_system_info(self) -> dict:
        """Get system information"""
        return {
            "whisper_available": WHISPER_AVAILABLE,
            "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
            "gtts_available": GTTS_AVAILABLE,
            "pydub_available": PYDUB_AVAILABLE,
            "current_whisper_model": self.whisper_model_name,
            "tts_language": self.tts_language,
            "torch_available": torch.cuda.is_available() if 'torch' in globals() else False,
            "gpu_count": torch.cuda.device_count() if 'torch' in globals() and torch.cuda.is_available() else 0
        }

# Convenience functions
async def transcribe_audio_file(file_path: str, model: str = "base") -> str:
    """Quick transcription of audio file"""
    processor = OpenSourceVoiceProcessor(whisper_model=model)
    return await processor.transcribe_audio(file_path)

async def text_to_speech(text: str, language: str = "en") -> bytes:
    """Quick text-to-speech conversion"""
    processor = OpenSourceVoiceProcessor(tts_language=language)
    return await processor.synthesize_speech(text)
