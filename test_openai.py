#!/usr/bin/env python3
"""
Test OpenAI API functionality directly
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

async def test_openai_api():
    """Test OpenAI API directly"""
    print("🧪 Testing OpenAI API Connection")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        print("❌ No valid OpenAI API key found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        print("🔄 Testing API call...")
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": "What is a headache?"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print("✅ OpenAI API call successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return False

async def test_medical_llm():
    """Test medical LLM client"""
    print("\n🤖 Testing Medical LLM Client")
    print("=" * 50)
    
    try:
        from medical_llm import MedicalLLMClient
        
        client = MedicalLLMClient()
        print(f"✅ Medical LLM client initialized")
        print(f"API Key available: {client.api_key != 'YOUR_OPENAI_API_KEY'}")
        print(f"Client object: {client.client is not None}")
        
        # Test query
        print("\n🔄 Testing medical query...")
        response = await client.query_medical_llm("What is a headache?")
        
        print("✅ Medical query successful!")
        print(f"Category: {response.category}")
        print(f"Risk Level: {response.risk_level}")
        print(f"Response: {response.response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Medical LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🩺 Medical AI Assistant - API Testing")
    print("=" * 60)
    
    # Test OpenAI API
    openai_works = await test_openai_api()
    
    # Test Medical LLM
    medical_works = await test_medical_llm()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"OpenAI API: {'✅ Working' if openai_works else '❌ Failed'}")
    print(f"Medical LLM: {'✅ Working' if medical_works else '❌ Failed'}")
    
    if not openai_works:
        print("\n💡 Troubleshooting:")
        print("1. Check if OPENAI_API_KEY is correctly set in .env file")
        print("2. Verify API key is valid and has credit/quota")
        print("3. Check internet connection")
        print("4. System will use enhanced mock responses instead")

if __name__ == "__main__":
    asyncio.run(main())
