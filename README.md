# 🏥 Open-Source Medical AI Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.2.0-blue.svg)](https://reactjs.org/)
[![React Native](https://img.shields.io/badge/React%20Native-0.72.6-blue.svg)](https://reactnative.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)

Welcome to the Open-Source Medical AI Assistant - a completely free, privacy-focused, and professional-grade medical AI platform. This comprehensive system combines advanced open-source language models with intuitive voice interaction, available as web and mobile applications.

## 🌟 Project Vision

A revolutionary medical AI platform designed to:
- **🔓 Complete Privacy**: All processing happens locally on your device
- **💡 Evidence-Based**: Provide medical information using cutting-edge AI models
- **🌐 Multi-Platform**: Available as web app, mobile app, and API
- **🤖 Professional Grade**: Enterprise-level features without subscription costs
- **🔄 Extensible**: Open architecture for customization and enhancement
- **🏥 Healthcare Bridge**: Connect patients with professional medical resources

## ✨ Key Features

### Core Capabilities
- **🔓 100% Open Source**: No API keys, no subscriptions, no external dependencies
- **🖥️ Local Processing**: All AI computation happens on your device
- **🔒 Privacy First**: Your medical data never leaves your computer
- **🧠 Multiple AI Models**: Support for various open-source LLMs (Llama, Mistral, GPT-2, etc.)
- **🎙️ Voice Integration**: Local speech-to-text (Whisper) and text-to-speech (gTTS)
- **⚡ GPU Acceleration**: Optimized for both CPU and GPU processing
- **🎯 Medical Focus**: Specialized medical knowledge base and emergency detection

### Advanced Features
- **🎛️ Fine-Tuning Support**: Customize models with your own medical datasets
- **🔄 Multi-Modal Processing**: Handle both voice and text inputs seamlessly
- **📊 Real-Time Analytics**: Track usage and model performance
- **🛡️ Safety Protocols**: Built-in medical disclaimers and emergency detection
- **🎨 Modern UI**: Clean, accessible design optimized for medical consultations
- **🔧 Model Management**: Switch between different AI models on the fly

## Technology Stack

### Backend
- **API Framework**: FastAPI (Python)
- **AI Models**: Transformers, PyTorch, Hugging Face
- **Voice Processing**: OpenAI Whisper (local), gTTS
- **Vector Database**: ChromaDB for medical knowledge
- **Fine-Tuning**: Custom training pipeline for medical datasets

### Frontend
- **UI**: Modern HTML5, CSS3, JavaScript
- **Audio**: Web Audio API, MediaRecorder API
- **Responsive Design**: Mobile-friendly interface
- **Real-Time Updates**: Live system status and model information

### AI Models Supported
- **LLMs**: DialoGPT, GPT-2, DistilGPT-2, GODEL
- **Speech**: Whisper (tiny, base, small, medium, large)
- **Embeddings**: Sentence Transformers
- **Custom**: Support for fine-tuned medical models

## Installation

### Prerequisites

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **4GB RAM minimum** (8GB+ recommended for larger models)
- **Optional**: NVIDIA GPU with CUDA support for faster processing

### Step 1: Clone and Setup

```bash
git clone https://github.com/yourusername/open-source-medical-ai
cd open-source-medical-ai
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**System Dependencies:**

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
```

**On macOS:**
```bash
brew install portaudio ffmpeg
```

**On Windows:**
```bash
# Most dependencies install automatically
# For FFmpeg, download from https://ffmpeg.org/download.html
```

### Step 3: Initialize Models

```bash
# The system will automatically download models on first run
# Or manually download specific models:
python -c "import whisper; whisper.load_model('base')"
```

### Step 4: Optional Fine-Tuning

```bash
# Fine-tune on your medical data
python fine_tune_opensource.py --data data/medical_training_data.json
```

## Usage

### Quick Start

1. **Start the backend server:**
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Open the frontend:**
```bash
# Serve the frontend
python -m http.server 3000 --directory frontend
# Visit: http://localhost:3000/index_new.html
```

3. **Or open directly in browser:**
```bash
# Simply open frontend/index_new.html in your browser
```

### Features

1. **Text Chat**: 
   - Type medical questions in the input field
   - Get detailed responses with risk assessment
   - View sources and recommendations

2. **Voice Chat**: 
   - Click the microphone button to record
   - Speak your medical question
   - Get both text and audio responses

3. **System Management**:
   - View current AI model status
   - Switch between different models
   - Monitor system resources

4. **Model Information**:
   - Click "Models" to see available AI models
   - View memory requirements and capabilities
   - Check GPU acceleration status

## API Endpoints

### POST /voice-chat
- **Description**: Main voice chat endpoint
- **Input**: Audio file (multipart/form-data)
- **Output**: Audio response with transcription headers
- **Process**: Speech-to-text → AI processing → Text-to-speech

### POST /chat
- **Description**: Text-based chat
- **Input**: JSON `{"message": "your message"}`
- **Output**: JSON `{"response": "AI response"}`

### POST /speak
- **Description**: Convert text to speech
- **Input**: JSON `{"message": "text to convert"}`
- **Output**: Audio stream (MP3)

### GET /health
- **Description**: Health check endpoint
- **Output**: Service status

## LLM and Medical Information

The medical assistant is powered by an extensive language model with specialized medical knowledge, capable of providing information on:

- Medical symptoms and conditions
- Medications and treatments
- Health tips and preventive care
- Emergency response suggestions

The application is fine-tuned to detect emergency situations and emphasize the need for professional medical consultations.

## Customization

### Medical Query Handling

The medical assistant is designed to handle queries in various categories, and it categorizes them using advanced NLP techniques. This is essential for improving response accuracy and ensuring user safety.

### Voice Customization

1. Browse ElevenLabs voices in your dashboard.
2. Copy the voice ID of your preferred voice.
3. Update the `voice_id` variable in the backend.

### Changing AI Personality

Modify the system prompt in `medical_llm.py` for specific personality traits or information emphasis.

## Troubleshooting

### Common Issues

1. **Microphone Access Denied**:
   - Ensure HTTPS or localhost for microphone access
   - Check browser permissions

2. **API Key Errors**:
   - Verify API keys are correct in `.env`
   - Check API key permissions and quotas

3. **Audio Not Playing**:
   - Check browser audio permissions
   - Verify ElevenLabs API is working

4. **CORS Issues**:
   - Backend includes CORS middleware
   - Ensure frontend URL is allowed

### Audio Issues

- Use Chrome/Firefox for best audio support
- Ensure microphone permissions are granted
- Check that audio output is not muted

## Browser Compatibility

- **Chrome**: Full support
- **Firefox**: Full support
- **Safari**: Partial support (some audio features may vary)
- **Edge**: Full support

## Production Deployment

For production deployment:

1. Set up proper environment variables
2. Use a production WSGI server (e.g., Gunicorn)
3. Configure proper CORS origins
4. Set up HTTPS for microphone access
5. Consider rate limiting for API calls
6. Examine the rate limiting for API call
7. Evaluate the performance of the model by optimizing the efficiency

## 🧪 Running Tests

To run backend tests, use:

```bash
pytest
```

You can add your tests in the `backend/tests/` directory. Example tests should cover authentication, database models, and API endpoints.

## 🔒 Security Best Practices

- **Never commit real secrets or API keys.** Use a `.env` file for all sensitive information.
- Set a strong, unique `SECRET_KEY` in your environment for JWT and cryptography.
- Regularly update dependencies to patch vulnerabilities.
- Use HTTPS in production to protect user data and enable microphone access.
- Restrict CORS origins to trusted domains.
- Consider rate limiting and monitoring for abuse.
- Deactivate users by setting their `is_active` field to `False` (see User Management below).

## 👤 User Management

- Each user has an `is_active` field in the database.
- To deactivate a user (prevent login and API access), set `is_active` to `False` in the database.
- Inactive users will be denied access to protected resources by the backend.
- You can manage users via direct database access or by building an admin interface.

## 📁 Project Structure

```
open-source-medical-ai/
├── 📁 backend/                     # FastAPI Backend
│   ├── main.py                    # Main API server
│   ├── opensource_medical_llm.py  # Open-source LLM implementation
│   ├── opensource_voice.py        # Voice processing system
│   ├── medical_knowledge.py       # Medical knowledge engine
│   ├── medical_analytics.py       # Analytics and logging
│   ├── auth.py                    # Authentication system
│   ├── database.py               # Database models
│   ├── config.py                 # Configuration
│   └── tests/                    # Backend tests
│
├── 📁 web/                        # React Web Application
│   ├── public/
│   │   └── index.html            # Web app entry point
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── pages/               # Page components
│   │   ├── contexts/            # React contexts
│   │   ├── styles/              # Styling
│   │   ├── App.js               # Main App component
│   │   └── index.js             # React entry point
│   └── package.json             # Web dependencies
│
├── 📁 mobile/                     # React Native Mobile App
│   ├── src/                     # Mobile app source
│   └── package.json             # Mobile dependencies
│
├── 📁 frontend/                   # Legacy HTML Interface
│   └── index.html               # Simple HTML interface
│
├── 📁 data/                       # Training Data
│   └── medical_training_data.json # Medical training dataset
│
├── 📁 scripts/                    # Utility Scripts
│   ├── setup.py                # Setup and initialization
│   └── start_server.py         # Server startup
│
├── fine_tune_opensource.py       # Model fine-tuning script
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
└── README.md                    # This documentation
```

### Platform Overview

| Platform | Technology | Features | Status |
|----------|------------|----------|--------|
| 🖥️ **Web App** | React 18 + FastAPI | Full-featured, Professional UI | ✅ Ready |
| 📱 **Mobile App** | React Native + Expo | Cross-platform, Native feel | 🚧 In Development |
| 🌐 **Legacy Web** | HTML5 + JavaScript | Simple, Fast loading | ✅ Ready |
| 🔧 **Backend API** | FastAPI + Python | RESTful, High performance | ✅ Ready |
| 🤖 **AI Engine** | Transformers + PyTorch | Local processing, Privacy-first | ✅ Ready |

## Medical Disclaimer

⚠️ **Important Medical Disclaimer**: This application is for educational and informational purposes only. The medical information provided by this AI assistant should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application. Please ensure that any medical information contributions are accurate and include appropriate disclaimers.

---

Enjoy your voice-powered AI medical assistant! 🩺💬🎙️
