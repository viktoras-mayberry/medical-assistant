#!/usr/bin/env python3
"""
Medical AI Assistant Runner
This script starts the backend server and optionally runs tests
"""
import os
import sys
import subprocess
import time
import threading
from pathlib import Path
import webbrowser

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting Medical AI Assistant Backend...")
    backend_dir = Path(__file__).parent / "backend"
    
    # Change to backend directory
    os.chdir(backend_dir)
    
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

def run_tests():
    """Run API tests"""
    print("🧪 Running API tests...")
    time.sleep(5)  # Wait for server to start
    
    # Run the test script
    test_script = Path(__file__).parent / "test_api.py"
    try:
        subprocess.run([sys.executable, str(test_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
    except FileNotFoundError:
        print("❌ Test script not found")

def open_frontend():
    """Open the frontend in browser"""
    frontend_file = Path(__file__).parent / "frontend" / "index.html"
    if frontend_file.exists():
        print("🌐 Opening frontend in browser...")
        webbrowser.open(f"file://{frontend_file.resolve()}")
    else:
        print("❌ Frontend file not found")

def main():
    """Main function"""
    print("🩺 Medical AI Assistant")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not (Path("backend").exists() and Path("frontend").exists()):
        print("❌ Please run this script from the project root directory.")
        sys.exit(1)
    
    # Start backend in a separate thread for testing
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 Running in test mode...")
        backend_thread = threading.Thread(target=start_backend)
        backend_thread.daemon = True
        backend_thread.start()
        
        # Wait a moment then run tests
        time.sleep(5)
        run_tests()
        return
    
    # Open frontend
    open_frontend()
    
    # Start backend (this will block)
    start_backend()

if __name__ == "__main__":
    main()
