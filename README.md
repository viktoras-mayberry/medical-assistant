# 🩺 Advanced Medical AI Assistant

Welcome to the Medical AI Assistant, a state-of-the-art application that combines advanced medical language models with intuitive voice interaction. This sophisticated system goes beyond simple FAQ responses to provide comprehensive medical knowledge, insights, and guidance across all aspects of healthcare.

## 🚀 Project Vision

This isn't just another chatbot - it's a comprehensive medical knowledge companion designed to:
- Provide evidence-based medical information across all specialties
- Offer intelligent analysis of symptoms, conditions, and treatments
- Serve as a bridge between patients and healthcare professionals
- Deliver personalized health guidance while maintaining medical safety standards

## ✨ Key Features

### Core Capabilities
- **Comprehensive Medical Knowledge**: Access to extensive medical databases covering symptoms, conditions, treatments, medications, and preventive care
- **Dual Model Architecture**: Combines OpenAI's GPT-4 with fine-tuned local transformer models for specialized medical responses
- **Voice-Enabled Interaction**: Natural speech recognition and synthesis for hands-free medical consultations
- **Emergency Detection System**: Real-time identification of critical medical situations with immediate guidance
- **Medical Query Classification**: Advanced NLP categorization for precise, context-aware responses
- **Professional Interface**: Clean, accessible design optimized for medical use cases

### Advanced Features
- **Fine-Tuning Capabilities**: Customize models with domain-specific medical datasets
- **Multi-Modal Processing**: Handle both voice and text inputs seamlessly
- **Intelligent Fallbacks**: Robust response system ensuring continuous operation
- **Medical Safety Protocols**: Built-in disclaimers and professional consultation reminders
- **Real-Time Audio Processing**: High-quality speech synthesis and recognition

## Technology Stack

- **Backend**: FastAPI (Python)
- **Medical LLM**: OpenAI GPT-4 / Custom Medical LLM
- **Speech-to-Text**: Google Speech Recognition (Free)
- **AI Processing**: Advanced medical knowledge base with emergency detection
- **Text-to-Speech**: ElevenLabs API
- **Frontend**: HTML5, CSS3, JavaScript (Medical-themed UI)
- **Audio**: Web Audio API, MediaRecorder API
- **Medical Safety**: Built-in disclaimers and emergency detection

## Installation

### Prerequisites

- Python 3.8 or higher
- ElevenLabs API key

### Step 1: Clone and Setup

```bash
git clone <your-repository>
cd voice-chat-ai-assistant
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Note: If you have issues installing PyAudio, you may need to install system dependencies:

**On Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**On macOS:**
```bash
brew install portaudio
```

**On Windows:**
PyAudio should install automatically with pip.

### Step 3: Environment Configuration

1. Copy the environment example file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API key:
```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### Step 4: Get ElevenLabs API Key

#### ElevenLabs API Key:
1. Go to [ElevenLabs](https://elevenlabs.io/)
2. Create an account or sign in
3. Go to your profile settings
4. Copy your API key to your `.env` file

### Step 5: Configure Voice

1. In the backend code (`main.py`), update the voice ID:
```python
voice_id = "XMBhZBHTRP7bO9ojOPI0"  # Replace with your preferred voice ID
```

2. To find available voices, you can use ElevenLabs dashboard or API to list voices.

## Usage

### Running the Application

1. Start the backend server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Open the frontend in your browser:
```bash
# Open the HTML file directly in browser or serve it
python -m http.server 3000
# Then visit http://localhost:3000/voice_chat_frontend.html
```

### How to Use

1. **Voice Messages**: 
   - Press and hold the microphone button to record
   - Release to send the voice message
   - The AI will respond with both text and voice

2. **Text Messages**:
   - Type your message in the input field
   - Click send or press Enter
   - The AI will respond with voice

3. **Playing Voice Messages**:
   - Click the play button on any voice message bubble
   - Audio controls show message duration

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

## File Structure

```
medical-ai-assistant/
├── backend/
│   ├── main.py                # Backend API server
│   ├── medical_llm.py         # Medical LLM client
│   └── config.py              # Configuration settings
├── frontend/
│   └── index.html             # Medical-themed frontend interface
├── data/
│   └── knowledge_base.txt     # Legacy knowledge base (deprecated)
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
└── .env                      # Your API keys (create this)
```

## Medical Disclaimer

⚠️ **Important Medical Disclaimer**: This application is for educational and informational purposes only. The medical information provided by this AI assistant should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application. Please ensure that any medical information contributions are accurate and include appropriate disclaimers.

---

Enjoy your voice-powered AI medical assistant! 🩺💬🎙️
