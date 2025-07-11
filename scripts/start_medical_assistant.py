#!/usr/bin/env python3
"""
Medical AI Assistant Startup Script
This script helps you run the medical AI assistant with proper configuration.
"""

import sys
import os
import subprocess
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'python-dotenv', 'elevenlabs', 
        'pydantic', 'SpeechRecognition', 'pyaudio', 'openai'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_environment():
    """Check if environment variables are set."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please copy .env.example to .env and configure your API keys.")
        return False
    
    # Load environment variables
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    if "your_elevenlabs_api_key_here" in env_content:
        print("⚠️  Please configure your ElevenLabs API key in the .env file")
        return False
    
    if "your_openai_api_key_here" in env_content:
        print("⚠️  Warning: OpenAI API key not configured. Using mock responses for medical queries.")
    
    print("✅ Environment configuration looks good!")
    return True

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting Medical AI Assistant backend...")
    
    # Change to backend directory
    os.chdir("backend")
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start backend: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped by user.")
        return True
    
    return True

def open_frontend():
    """Open the frontend in the default web browser."""
    frontend_path = Path("frontend/index.html").resolve()
    if frontend_path.exists():
        print("🌐 Opening Medical AI Assistant in your browser...")
        webbrowser.open(f"file://{frontend_path}")
    else:
        print("❌ Frontend file not found!")

def main():
    """Main startup function."""
    print("🩺 Medical AI Assistant Startup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend").exists() or not Path("frontend").exists():
        print("❌ Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Open frontend
    open_frontend()
    
    # Start backend (this will block until interrupted)
    start_backend()

if __name__ == "__main__":
    main()
