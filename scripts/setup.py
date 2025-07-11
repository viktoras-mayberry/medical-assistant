#!/usr/bin/env python3
"""
Setup script for Voice Chat AI Assistant.
"""

import os
import sys
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            sys.exit(1)
        print("✅ Dependencies installed successfully")
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def setup_environment():
    """Set up environment file."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from .env.example")
            print("⚠️  Please edit .env file and add your ElevenLabs API key!")
        else:
            print("❌ .env.example file not found!")
            sys.exit(1)
    else:
        print("✅ .env file already exists")

def verify_directories():
    """Verify all required directories exist."""
    project_root = Path(__file__).parent.parent
    required_dirs = ["backend", "frontend", "data", "scripts"]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            print(f"❌ Required directory missing: {dir_name}")
            sys.exit(1)
        print(f"✅ Directory exists: {dir_name}")

def main():
    """Main setup function."""
    print("🎙️ Voice Chat AI Assistant Setup")
    print("=" * 40)
    
    check_python_version()
    verify_directories()
    install_dependencies()
    setup_environment()
    
    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Edit .env file with your ElevenLabs API key")
    print("2. Run: python scripts/start_server.py")
    print("3. Open frontend/index.html in your browser")
    print("\n🎯 Happy chatting!")

if __name__ == "__main__":
    main()
