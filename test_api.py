#!/usr/bin/env python3
"""
Test script for Medical AI Assistant API
"""
import requests
import json
import time
import sys

def test_api():
    """Test the Medical AI Assistant API"""
    base_url = "http://localhost:8000"
    
    print("🩺 Testing Medical AI Assistant API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test chat endpoint
    print("\n3. Testing chat endpoint...")
    test_messages = [
        "Hello, what is a headache?",
        "What should I do if I have a fever?",
        "I have chest pain and difficulty breathing",
        "How can I prevent the flu?",
        "What are the symptoms of diabetes?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n3.{i}. Testing message: '{message}'")
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={"message": message},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                data = response.json()
                print("✅ Chat response received")
                print(f"Response: {data['response'][:100]}...")
                print(f"Risk Level: {data['risk_level']}")
                print(f"Emergency: {data['is_emergency']}")
                print(f"Confidence: {data['confidence_score']}")
            else:
                print(f"❌ Chat failed: {response.status_code}")
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"❌ Chat error: {e}")
    
    # Test symptoms endpoint
    print("\n4. Testing symptoms endpoint...")
    try:
        response = requests.get(f"{base_url}/medical/symptoms")
        if response.status_code == 200:
            print("✅ Symptoms endpoint passed")
            print(f"Available symptoms: {response.json()}")
        else:
            print(f"❌ Symptoms endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Symptoms endpoint error: {e}")
    
    # Test symptom assessment
    print("\n5. Testing symptom assessment...")
    try:
        response = requests.post(
            f"{base_url}/medical/assess",
            json=["headache", "fever", "nausea"],
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print("✅ Symptom assessment passed")
            print(f"Assessment: {response.json()}")
        else:
            print(f"❌ Symptom assessment failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Symptom assessment error: {e}")
    
    # Test drug interaction check
    print("\n6. Testing drug interaction check...")
    try:
        response = requests.post(
            f"{base_url}/medical/drug-interactions",
            json=["aspirin", "warfarin"],
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print("✅ Drug interaction check passed")
            print(f"Interactions: {response.json()}")
        else:
            print(f"❌ Drug interaction check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Drug interaction check error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API testing completed!")
    return True

if __name__ == "__main__":
    # Wait a moment for server to start
    time.sleep(2)
    test_api()
