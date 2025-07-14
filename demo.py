#!/usr/bin/env python3
"""
Medical AI Assistant Demo
This script demonstrates the core functionality without requiring the server
"""
import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from medical_llm import MedicalLLMClient, MedicalResponse
from medical_knowledge import MedicalKnowledgeEngine
from medical_analytics import MedicalAnalytics
import asyncio

def demo_medical_llm():
    """Demo the medical LLM functionality"""
    print("🤖 Medical LLM Demo")
    print("=" * 50)
    
    # Initialize the medical LLM client
    client = MedicalLLMClient(use_local_model=False)
    
    # Test queries
    test_queries = [
        "What is a headache?",
        "I have a fever and cough",
        "I'm experiencing chest pain and difficulty breathing",
        "How can I prevent diabetes?",
        "What medications can help with high blood pressure?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 30)
        
        # Process query synchronously for demo
        response = asyncio.run(client.query_medical_llm(query))
        
        print(f"Category: {response.category}")
        print(f"Risk Level: {response.risk_level}")
        print(f"Emergency: {response.is_emergency}")
        print(f"Confidence: {response.confidence_score}")
        print(f"Response: {response.response}")
        print(f"Recommendations: {', '.join(response.recommendations)}")

def demo_medical_knowledge():
    """Demo the medical knowledge engine"""
    print("\n🧠 Medical Knowledge Engine Demo")
    print("=" * 50)
    
    engine = MedicalKnowledgeEngine()
    
    # Test knowledge search
    print("\n1. Knowledge Search for 'diabetes':")
    matches = engine.search_knowledge("diabetes")
    for match in matches[:2]:  # Show first 2 matches
        print(f"- {match.title}: {match.content[:100]}...")
    
    # Test symptom assessment
    print("\n2. Symptom Assessment:")
    symptoms = ["headache", "fever", "nausea"]
    assessment = engine.assess_symptoms(symptoms)
    print(f"Symptoms: {symptoms}")
    print(f"Assessment: {assessment}")
    
    # Test drug interaction check
    print("\n3. Drug Interaction Check:")
    drugs = ["aspirin", "warfarin"]
    interactions = engine.check_drug_interactions(drugs)
    print(f"Drugs: {drugs}")
    print(f"Interactions: {interactions}")
    
    # Test emergency detection
    print("\n4. Emergency Detection:")
    emergency_queries = [
        "I'm having chest pain",
        "I feel dizzy",
        "I have a severe headache and can't breathe"
    ]
    
    for query in emergency_queries:
        result = engine.detect_emergency(query)
        print(f"Query: '{query}' -> Emergency: {result['is_emergency']}")

def demo_medical_analytics():
    """Demo the medical analytics"""
    print("\n📊 Medical Analytics Demo")
    print("=" * 50)
    
    analytics = MedicalAnalytics()
    
    # Simulate some interactions
    print("Simulating medical interactions...")
    
    # This is a simplified demo since the full analytics requires async
    print("✅ Analytics system initialized")
    print("- Tracks user interactions")
    print("- Monitors risk levels")
    print("- Provides usage insights")
    print("- Detects emergency patterns")

def demo_health_tips():
    """Demo health tips functionality"""
    print("\n💡 Health Tips Demo")
    print("=" * 50)
    
    client = MedicalLLMClient()
    tips = client.get_health_tips()
    
    print("General Health Tips:")
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")
    
    print("\n🚨 Emergency Contacts:")
    contacts = client.get_emergency_contacts()
    for service, number in contacts.items():
        print(f"- {service}: {number}")

def main():
    """Main demo function"""
    print("🩺 Medical AI Assistant - Core Functionality Demo")
    print("=" * 60)
    
    try:
        demo_medical_llm()
        demo_medical_knowledge()
        demo_medical_analytics()
        demo_health_tips()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Medical query processing with AI")
        print("- Knowledge base search and matching")
        print("- Symptom assessment and risk evaluation")
        print("- Drug interaction checking")
        print("- Emergency detection")
        print("- Health tips and emergency contacts")
        print("- Analytics and monitoring capabilities")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
