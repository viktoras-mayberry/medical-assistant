#!/usr/bin/env python3
"""
Interactive Medical Chat Interface
FastAPI-based web interface for the Advanced Medical AI
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our advanced medical AI
from advanced_medical_ai import AdvancedMedicalAI, ConversationResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Medical AI Chat Interface",
    description="Interactive medical AI assistant with advanced NLP capabilities",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI instance
medical_ai = AdvancedMedicalAI()

# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., description="User's medical query")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
class ChatResponse(BaseModel):
    response: str
    confidence: float
    intent: str
    risk_level: str
    category: str
    recommendations: List[str]
    context_awareness: Dict[str, Any]
    semantic_similarity: float
    timestamp: str

class ConversationSummary(BaseModel):
    total_interactions: int
    intents_discussed: List[str]
    risk_levels_encountered: List[str]
    last_interaction: str
    session_duration: float

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_ais: Dict[str, AdvancedMedicalAI] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if session_id not in self.session_ais:
            self.session_ais[session_id] = AdvancedMedicalAI()
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        self.active_connections.remove(websocket)
        if session_id in self.session_ais:
            del self.session_ais[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Medical AI Assistant</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .chat-container {
                width: 90%;
                max-width: 800px;
                height: 80vh;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #2563eb, #06b6d4);
                color: white;
                padding: 20px;
                text-align: center;
            }
            
            .chat-header h1 {
                font-size: 24px;
                margin-bottom: 5px;
            }
            
            .chat-header p {
                opacity: 0.9;
                font-size: 14px;
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background-color: #f8fafc;
            }
            
            .message {
                margin-bottom: 15px;
                display: flex;
                align-items: flex-start;
                gap: 10px;
                animation: slideIn 0.3s ease-out;
            }
            
            .message.user {
                flex-direction: row-reverse;
            }
            
            .message-avatar {
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                font-weight: bold;
                color: white;
                flex-shrink: 0;
            }
            
            .message.ai .message-avatar {
                background: linear-gradient(135deg, #2563eb, #06b6d4);
            }
            
            .message.user .message-avatar {
                background: linear-gradient(135deg, #10b981, #22c55e);
            }
            
            .message-content {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 15px;
                word-wrap: break-word;
                line-height: 1.4;
            }
            
            .message.ai .message-content {
                background: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border: 1px solid #e2e8f0;
            }
            
            .message.user .message-content {
                background: linear-gradient(135deg, #2563eb, #06b6d4);
                color: white;
            }
            
            .message-meta {
                font-size: 11px;
                color: #64748b;
                margin-top: 5px;
            }
            
            .risk-badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 10px;
                font-weight: bold;
                margin-left: 5px;
            }
            
            .risk-low { background: #dcfce7; color: #166534; }
            .risk-moderate { background: #fef3c7; color: #92400e; }
            .risk-high { background: #fee2e2; color: #dc2626; }
            .risk-critical { background: #fca5a5; color: #991b1b; }
            
            .recommendations {
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #e2e8f0;
            }
            
            .recommendations-title {
                font-size: 12px;
                font-weight: bold;
                color: #374151;
                margin-bottom: 4px;
            }
            
            .recommendation {
                font-size: 11px;
                color: #6b7280;
                margin-bottom: 2px;
            }
            
            .chat-input {
                padding: 20px;
                background: white;
                border-top: 1px solid #e2e8f0;
                display: flex;
                gap: 10px;
            }
            
            .chat-input input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #d1d5db;
                border-radius: 25px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.2s;
            }
            
            .chat-input input:focus {
                border-color: #2563eb;
            }
            
            .chat-input button {
                padding: 12px 20px;
                background: linear-gradient(135deg, #2563eb, #06b6d4);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 14px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            
            .chat-input button:hover {
                transform: translateY(-1px);
            }
            
            .chat-input button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            
            .typing-indicator {
                display: none;
                align-items: center;
                gap: 10px;
                margin-bottom: 15px;
            }
            
            .typing-dots {
                display: flex;
                gap: 4px;
            }
            
            .typing-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #64748b;
                animation: typing 1.4s infinite;
            }
            
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }
            
            .disclaimer {
                background: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 8px;
                padding: 12px;
                margin: 10px 20px;
                font-size: 12px;
                color: #92400e;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🩺 Advanced Medical AI Assistant</h1>
                <p>Intelligent medical guidance powered by advanced NLP</p>
            </div>
            
            <div class="disclaimer">
                ⚠️ Medical Disclaimer: This information is for educational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for proper diagnosis and treatment.
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message ai">
                    <div class="message-avatar">🩺</div>
                    <div class="message-content">
                        Hello! I'm your Advanced Medical AI Assistant. I'm here to help you with medical questions, symptom assessment, and health guidance. How can I assist you today?
                    </div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="message-avatar">🩺</div>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            
            <div class="chat-input">
                <input type="text" id="messageInput" placeholder="Ask me about symptoms, conditions, medications, or health tips..." />
                <button onclick="sendMessage()" id="sendButton">Send</button>
            </div>
        </div>
        
        <script>
            const chatMessages = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const typingIndicator = document.getElementById('typingIndicator');
            
            // Generate session ID
            const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
            
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, 'user');
                messageInput.value = '';
                sendButton.disabled = true;
                
                // Show typing indicator
                typingIndicator.style.display = 'flex';
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Hide typing indicator
                    typingIndicator.style.display = 'none';
                    
                    // Add AI response
                    addMessage(data.response, 'ai', data);
                    
                } catch (error) {
                    console.error('Error:', error);
                    typingIndicator.style.display = 'none';
                    addMessage('Sorry, I encountered an error. Please try again.', 'ai');
                }
                
                sendButton.disabled = false;
                messageInput.focus();
            }
            
            function addMessage(content, sender, metadata = null) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const avatar = document.createElement('div');
                avatar.className = 'message-avatar';
                avatar.textContent = sender === 'user' ? '👤' : '🩺';
                
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContent.textContent = content;
                
                if (metadata && sender === 'ai') {
                    const meta = document.createElement('div');
                    meta.className = 'message-meta';
                    
                    const riskBadge = document.createElement('span');
                    riskBadge.className = `risk-badge risk-${metadata.risk_level}`;
                    riskBadge.textContent = metadata.risk_level.toUpperCase();
                    
                    meta.innerHTML = `
                        Intent: ${metadata.intent} | Category: ${metadata.category} | 
                        Confidence: ${(metadata.confidence * 100).toFixed(1)}%
                    `;
                    meta.appendChild(riskBadge);
                    
                    messageContent.appendChild(meta);
                    
                    if (metadata.recommendations && metadata.recommendations.length > 0) {
                        const recommendations = document.createElement('div');
                        recommendations.className = 'recommendations';
                        
                        const title = document.createElement('div');
                        title.className = 'recommendations-title';
                        title.textContent = '💡 Recommendations:';
                        recommendations.appendChild(title);
                        
                        metadata.recommendations.forEach(rec => {
                            const recDiv = document.createElement('div');
                            recDiv.className = 'recommendation';
                            recDiv.textContent = `• ${rec}`;
                            recommendations.appendChild(recDiv);
                        });
                        
                        messageContent.appendChild(recommendations);
                    }
                }
                
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(messageContent);
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Focus on input when page loads
            messageInput.focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Process chat message and return AI response"""
    try:
        # Process the message with our AI
        response = await medical_ai.process_conversation(message.message)
        
        # Convert to response model
        chat_response = ChatResponse(
            response=response.response,
            confidence=response.confidence,
            intent=response.intent,
            risk_level=response.risk_level,
            category=response.category,
            recommendations=response.recommendations,
            context_awareness=response.context_awareness,
            semantic_similarity=response.semantic_similarity,
            timestamp=response.timestamp.isoformat()
        )
        
        logger.info(f"Processed chat message: {message.message[:50]}... -> {response.intent}")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/summary", response_model=ConversationSummary)
async def get_conversation_summary():
    """Get conversation summary"""
    try:
        summary = medical_ai.get_conversation_summary()
        return ConversationSummary(**summary)
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/reset")
async def reset_conversation():
    """Reset conversation history"""
    try:
        global medical_ai
        medical_ai = AdvancedMedicalAI()
        return {"message": "Conversation reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Advanced Medical AI Chat Interface",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint for real-time chat
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Get AI instance for this session
            ai_instance = manager.session_ais.get(session_id, medical_ai)
            
            # Process message
            response = await ai_instance.process_conversation(message_data["message"])
            
            # Send response back to client
            response_data = {
                "response": response.response,
                "confidence": response.confidence,
                "intent": response.intent,
                "risk_level": response.risk_level,
                "category": response.category,
                "recommendations": response.recommendations,
                "context_awareness": response.context_awareness,
                "semantic_similarity": response.semantic_similarity,
                "timestamp": response.timestamp.isoformat()
            }
            
            await manager.send_personal_message(json.dumps(response_data), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_personal_message(
            json.dumps({"error": "An error occurred processing your message"}),
            websocket
        )

if __name__ == "__main__":
    print("🩺 Starting Advanced Medical AI Chat Interface...")
    print("=" * 60)
    print("🌐 Web Interface: http://localhost:8001")
    print("📊 Health Check: http://localhost:8001/health")
    print("💬 Chat API: http://localhost:8001/chat")
    print("🔄 Reset: http://localhost:8001/conversation/reset")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True
    )
