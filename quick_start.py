#!/usr/bin/env python3
"""
Quick Start Script for Advanced Medical AI Assistant
===================================================

This script helps you quickly set up and test the medical AI assistant system.
It will:
1. Check system dependencies
2. Initialize the database
3. Start the server
4. Test basic functionality
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🏥 {text}")
    print("="*60)

def print_step(step, text):
    """Print formatted step"""
    print(f"\n{step}. {text}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print_step("1", "Checking Dependencies")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "psycopg2-binary",
        "python-jose",
        "passlib",
        "python-multipart",
        "pydantic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def setup_environment():
    """Set up environment variables"""
    print_step("2", "Setting up Environment")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        env_content = """
# Advanced Medical AI Assistant Configuration
APP_NAME="Advanced Medical AI Assistant"
APP_VERSION="2.0.0"
DEBUG=true
SECRET_KEY="your-super-secret-key-change-this-in-production"

# Database
DATABASE_URL="sqlite:///./medical_ai.db"
REDIS_URL="redis://localhost:6379"

# AI Models
PRIMARY_MEDICAL_MODEL="microsoft/DialoGPT-medium"
WHISPER_MODEL="base"
ENABLE_GPU=false

# Security
JWT_EXPIRATION_HOURS=24
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Features
FEATURE_VOICE_CHAT=true
FEATURE_DRUG_INTERACTIONS=true
FEATURE_LITERATURE_SEARCH=true
FEATURE_EMERGENCY_DETECTION=true
FEATURE_ANALYTICS=true

# CORS
ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000"
"""
        env_file.write_text(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")

def initialize_database():
    """Initialize the database"""
    print_step("3", "Initializing Database")
    
    try:
        # Import and run database initialization
        from backend.database import init_db
        init_db()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print_step("4", "Starting Server")
    
    try:
        print("🚀 Starting Advanced Medical AI Assistant server...")
        print("📋 Server will be available at: http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🔐 Authentication endpoints: http://localhost:8000/auth")
        print("\nPress Ctrl+C to stop the server")
        
        # Start the server
        os.chdir("backend")
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        return False

def test_system():
    """Test the system functionality"""
    print_step("5", "Testing System")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test authentication endpoints
    try:
        response = requests.get(f"{base_url}/auth/health", timeout=5)
        if response.status_code == 200:
            print("✅ Authentication system working")
        else:
            print(f"❌ Authentication system failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication system failed: {e}")
    
    # Test registration (example)
    try:
        test_user = {
            "email": "test@example.com",
            "password": "TestPass123!",
            "confirm_password": "TestPass123!",
            "first_name": "Test",
            "last_name": "User"
        }
        
        response = requests.post(f"{base_url}/auth/register", json=test_user, timeout=5)
        if response.status_code == 200:
            print("✅ User registration working")
        else:
            print(f"⚠️  User registration test: {response.status_code} (may be normal if user exists)")
    except requests.exceptions.RequestException as e:
        print(f"❌ User registration test failed: {e}")
    
    return True

def main():
    """Main function"""
    print_header("Advanced Medical AI Assistant - Quick Start")
    
    print("🩺 Welcome to the Advanced Medical AI Assistant!")
    print("This script will help you set up and test the system.")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return
    
    # Setup environment
    setup_environment()
    
    # Initialize database
    if not initialize_database():
        print("\n❌ Database initialization failed. Please check the error above.")
        return
    
    # Ask user if they want to start the server
    print("\n" + "="*60)
    choice = input("Do you want to start the server now? (y/n): ").lower().strip()
    
    if choice == 'y' or choice == 'yes':
        start_server()
    else:
        print("\n📋 To start the server manually, run:")
        print("   cd backend")
        print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        print("\n📖 Then visit: http://localhost:8000/docs")
        print("\n🎉 Setup complete! Your Advanced Medical AI Assistant is ready.")

if __name__ == "__main__":
    main()
