# 🎙️ Voice Chat AI Travel Assistant

A voice chat application that allows users to interact with an AI travel assistant through both voice notes and text messages. The AI responds with voice messages, creating a natural conversation experience.

## Features

- **Voice-to-Voice Communication**: Record voice messages and get AI responses in voice
- **Speech-to-Text**: Automatic transcription of voice messages using OpenAI Whisper
- **Text-to-Speech**: AI responses converted to natural speech using ElevenLabs
- **Knowledge Base**: Comprehensive travel FAQ and information database
- **WhatsApp-style Interface**: Familiar chat interface with voice message bubbles
- **Dual Input**: Support for both voice and text input
- **Real-time Audio**: Play voice messages with audio controls

## Technology Stack

- **Backend**: FastAPI (Python)
- **Speech-to-Text**: Google Speech Recognition (Free)
- **AI Chat**: Custom NLP with similarity matching
- **Text-to-Speech**: ElevenLabs API
- **Frontend**: HTML5, CSS3, JavaScript
- **Audio**: Web Audio API, MediaRecorder API

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

## Knowledge Base

The AI uses a comprehensive knowledge base stored in `knowledge_base.txt` containing:

- Travel planning guidance
- Destination recommendations
- Booking assistance
- Safety tips
- Cultural information
- Emergency procedures
- And much more...

You can customize the knowledge base by editing this file to add more travel information or modify existing responses.

## Customization

### Adding More Travel Information

Edit `knowledge_base.txt` and add new FAQ entries:

```text
Q: Your new question here?
A: Your detailed answer here with helpful travel information.
```

### Changing AI Personality

Modify the system prompt in `main.py` in the `get_response` method:

```python
system_prompt = f"""
You are a helpful AI travel assistant. [Add your personality traits here]
...
"""
```

### Voice Customization

1. Browse ElevenLabs voices in your dashboard
2. Copy the voice ID of your preferred voice
3. Update the `voice_id` variable in the backend

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

## File Structure

```
voice-chat-ai-assistant/
├── main.py                    # Backend API server
├── voice_chat_frontend.html   # Frontend interface
├── knowledge_base.txt         # Travel information database
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
└── .env                      # Your API keys (create this)
```

## License

This project is an open source .

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

---

Enjoy your voice-powered AI travel assistant! 🌍✈️🎙️