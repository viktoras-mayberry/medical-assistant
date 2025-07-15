#!/usr/bin/env python3
"""
Advanced Medical AI Conversational Engine
This module creates an interactive medical AI assistant using advanced NLP techniques
without external dependencies like OpenAI API.
"""

import json
import re
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import nltk
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationResponse:
    """Structure for AI conversation responses"""
    response: str
    confidence: float
    intent: str
    risk_level: str
    category: str
    recommendations: List[str]
    context_awareness: Dict[str, Any]
    semantic_similarity: float
    timestamp: datetime

class AdvancedMedicalAI:
    """Advanced Medical AI Conversational Engine"""
    
    def __init__(self):
        self.conversation_history = []
        self.user_context = {}
        self.medical_knowledge_base = {}
        self.sentence_transformer = None
        self.nlp_model = None
        self.classifier = None
        self.tokenizer = None
        self.tfidf_vectorizer = None
        self.knowledge_embeddings = None
        
        # Initialize components
        self._initialize_nlp_components()
        self._load_medical_knowledge()
        self._setup_semantic_search()
        
    def _initialize_nlp_components(self):
        """Initialize NLP components"""
        try:
            # Load spaCy model for NLP processing
            self.nlp_model = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp_model = None
        
        try:
            # Initialize sentence transformer for semantic understanding
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer: {e}")
            
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
    def _load_medical_knowledge(self):
        """Load and expand medical knowledge base"""
        # Load from training data
        try:
            with open('data/medical_training_data.json', 'r') as f:
                training_data = json.load(f)
            
            for entry in training_data:
                key = entry['input'].lower()
                self.medical_knowledge_base[key] = {
                    'response': entry['output'],
                    'intent': entry['intent'],
                    'category': entry['category'],
                    'risk_level': entry['risk_level']
                }
        except FileNotFoundError:
            logger.warning("Training data not found. Using default knowledge base.")
            
        # Expand with additional medical knowledge
        self._expand_medical_knowledge()
        
    def _expand_medical_knowledge(self):
        """Expand medical knowledge with additional entries"""
        additional_knowledge = {
            "fever treatment": {
                "response": "For fever treatment, rest and stay hydrated. Use acetaminophen or ibuprofen as directed for adults. For children, use age-appropriate medications. Seek medical attention if fever exceeds 103°F (39.4°C), persists more than 3 days, or is accompanied by severe symptoms like difficulty breathing, chest pain, or severe headache.",
                "intent": "treatment_inquiry",
                "category": "treatments",
                "risk_level": "moderate"
            },
            "high cholesterol": {
                "response": "High cholesterol is managed through diet, exercise, and sometimes medication. Eat foods rich in omega-3 fatty acids, reduce saturated fats, increase fiber intake, and exercise regularly. Statins may be prescribed if lifestyle changes aren't enough. Regular monitoring is essential as high cholesterol increases heart disease risk.",
                "intent": "condition_inquiry",
                "category": "conditions",
                "risk_level": "moderate"
            },
            "covid symptoms": {
                "response": "COVID-19 symptoms include fever, cough, fatigue, body aches, sore throat, loss of taste/smell, shortness of breath, congestion, nausea, or diarrhea. Symptoms can range from mild to severe. If experiencing difficulty breathing, persistent chest pain, confusion, or inability to stay awake, seek immediate medical attention.",
                "intent": "symptom_inquiry",
                "category": "symptoms",
                "risk_level": "moderate"
            },
            "anxiety attack": {
                "response": "During an anxiety attack, try deep breathing exercises, ground yourself using the 5-4-3-2-1 technique (5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste), and remind yourself it will pass. If attacks are frequent or severe, consider therapy or medication. Seek immediate help if experiencing chest pain or thoughts of self-harm.",
                "intent": "symptom_report",
                "category": "mental_health",
                "risk_level": "moderate"
            }
        }
        
        self.medical_knowledge_base.update(additional_knowledge)
        
    def _setup_semantic_search(self):
        """Set up semantic search capabilities"""
        if self.sentence_transformer:
            # Create embeddings for knowledge base
            knowledge_texts = [entry['response'] for entry in self.medical_knowledge_base.values()]
            if knowledge_texts:
                self.knowledge_embeddings = self.sentence_transformer.encode(knowledge_texts)
                logger.info("Knowledge embeddings created")
                
    def _extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        entities = {
            'symptoms': [],
            'conditions': [],
            'medications': [],
            'body_parts': [],
            'time_expressions': []
        }
        
        if not self.nlp_model:
            return entities
            
        doc = self.nlp_model(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                continue
            elif ent.label_ == 'TIME':
                entities['time_expressions'].append(ent.text)
                
        # Pattern matching for medical terms
        symptom_patterns = [
            r'\b(pain|ache|hurt|sore|burning|itching|swelling|rash|fever|nausea|dizzy|tired|weak)\b',
            r'\b(headache|stomachache|backache|toothache)\b',
            r'\b(cough|sneeze|runny nose|congestion)\b'
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text.lower())
            entities['symptoms'].extend(matches)
            
        return entities
        
    def _classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify user intent using pattern matching and keywords"""
        text_lower = text.lower()
        
        # Intent patterns
        intent_patterns = {
            'symptom_inquiry': [
                r'\b(what is|what are|tell me about|explain|describe)\b.*\b(symptom|sign|indication)\b',
                r'\b(symptom|sign).*\b(of|for|with)\b',
                r'\b(how do I know|how can I tell)\b'
            ],
            'symptom_report': [
                r'\b(i have|i feel|i am experiencing|i\'m having)\b',
                r'\b(my|i|me).*\b(hurt|pain|ache|feel|sick|ill)\b',
                r'\b(it hurts|i\'m in pain|i feel sick)\b'
            ],
            'condition_inquiry': [
                r'\b(what is|what are|tell me about|explain|describe)\b.*\b(disease|condition|disorder|illness)\b',
                r'\b(how serious|how dangerous|how common)\b',
                r'\b(causes of|risk factors)\b'
            ],
            'medication_inquiry': [
                r'\b(what medication|what drug|what medicine)\b',
                r'\b(side effects|interactions|dosage)\b',
                r'\b(how to take|when to take|how much)\b'
            ],
            'treatment_inquiry': [
                r'\b(how to treat|treatment for|cure for)\b',
                r'\b(what can I do|how can I help|what helps)\b',
                r'\b(therapy|treatment|medication)\b'
            ],
            'prevention_inquiry': [
                r'\b(how to prevent|prevention|avoid|protect)\b',
                r'\b(what can I do to prevent|how can I avoid)\b',
                r'\b(preventive|prevention)\b'
            ],
            'emergency_inquiry': [
                r'\b(emergency|urgent|serious|critical|severe)\b',
                r'\b(call 911|go to hospital|immediate help)\b',
                r'\b(chest pain|heart attack|stroke|can\'t breathe)\b'
            ]
        }
        
        best_intent = 'general_inquiry'
        best_confidence = 0.0
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = 0.8 + (len(re.findall(pattern, text_lower)) * 0.1)
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = min(confidence, 1.0)
                        
        return best_intent, best_confidence
        
    def _assess_risk_level(self, text: str, intent: str) -> str:
        """Assess risk level based on text content and intent"""
        text_lower = text.lower()
        
        # Critical risk indicators
        critical_keywords = [
            'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 'difficulty breathing',
            'severe bleeding', 'unconscious', 'suicide', 'overdose', 'poisoning',
            'severe pain', 'emergency', 'call 911'
        ]
        
        # High risk indicators
        high_risk_keywords = [
            'severe', 'intense', 'unbearable', 'getting worse', 'persistent',
            'blood', 'fever over 103', 'confusion', 'fainting'
        ]
        
        # Moderate risk indicators
        moderate_risk_keywords = [
            'fever', 'nausea', 'vomiting', 'dizziness', 'fatigue',
            'headache', 'pain', 'cough', 'shortness of breath'
        ]
        
        for keyword in critical_keywords:
            if keyword in text_lower:
                return 'critical'
                
        for keyword in high_risk_keywords:
            if keyword in text_lower:
                return 'high'
                
        for keyword in moderate_risk_keywords:
            if keyword in text_lower:
                return 'moderate'
                
        return 'low'
        
    def _find_best_match(self, query: str) -> Tuple[str, float]:
        """Find best matching response using semantic similarity"""
        if not self.sentence_transformer or not self.knowledge_embeddings:
            return self._fallback_keyword_search(query)
            
        # Create embedding for query
        query_embedding = self.sentence_transformer.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.knowledge_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        knowledge_items = list(self.medical_knowledge_base.items())
        
        if best_score > 0.3:  # Threshold for semantic similarity
            best_key, best_value = knowledge_items[best_idx]
            return best_value['response'], best_score
            
        return self._fallback_keyword_search(query)
        
    def _fallback_keyword_search(self, query: str) -> Tuple[str, float]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        best_match = None
        best_score = 0.0
        
        for key, value in self.medical_knowledge_base.items():
            # Simple keyword matching
            common_words = set(query_lower.split()) & set(key.split())
            score = len(common_words) / max(len(query_lower.split()), len(key.split()))
            
            if score > best_score:
                best_score = score
                best_match = value['response']
                
        if best_match and best_score > 0.2:
            return best_match, best_score
            
        return self._generate_default_response(query), 0.1
        
    def _generate_default_response(self, query: str) -> str:
        """Generate default response for unknown queries"""
        return (
            "I understand you're asking about a medical topic. While I can provide general "
            "health information, I recommend consulting with a healthcare professional "
            "for personalized advice about your specific situation. They can provide "
            "proper evaluation and guidance based on your individual health needs."
        )
        
    def _get_recommendations(self, intent: str, risk_level: str) -> List[str]:
        """Get contextual recommendations based on intent and risk level"""
        recommendations = []
        
        if risk_level == 'critical':
            recommendations = [
                "Seek immediate medical attention",
                "Call 911 or go to emergency room",
                "Do not delay medical care"
            ]
        elif risk_level == 'high':
            recommendations = [
                "Contact healthcare provider immediately",
                "Consider urgent care if needed",
                "Monitor symptoms closely"
            ]
        elif risk_level == 'moderate':
            recommendations = [
                "Schedule appointment with healthcare provider",
                "Monitor symptoms for changes",
                "Follow general health guidelines"
            ]
        else:  # low risk
            recommendations = [
                "Consider consulting healthcare provider if concerned",
                "Maintain healthy lifestyle habits",
                "Stay informed about health topics"
            ]
            
        # Add intent-specific recommendations
        if intent == 'medication_inquiry':
            recommendations.extend([
                "Consult pharmacist about drug interactions",
                "Follow prescribed dosage instructions",
                "Report side effects to healthcare provider"
            ])
        elif intent == 'prevention_inquiry':
            recommendations.extend([
                "Maintain regular exercise routine",
                "Follow balanced diet",
                "Get recommended health screenings"
            ])
            
        return recommendations[:4]  # Limit to 4 recommendations
        
    async def process_conversation(self, user_input: str) -> ConversationResponse:
        """Process user input and generate intelligent response"""
        # Extract entities and classify intent
        entities = self._extract_medical_entities(user_input)
        intent, intent_confidence = self._classify_intent(user_input)
        risk_level = self._assess_risk_level(user_input, intent)
        
        # Find best matching response
        response, semantic_similarity = self._find_best_match(user_input)
        
        # Determine category based on intent
        category_map = {
            'symptom_inquiry': 'symptoms',
            'symptom_report': 'symptoms',
            'condition_inquiry': 'conditions',
            'medication_inquiry': 'medications',
            'treatment_inquiry': 'treatments',
            'prevention_inquiry': 'prevention',
            'emergency_inquiry': 'emergency'
        }
        category = category_map.get(intent, 'general')
        
        # Get recommendations
        recommendations = self._get_recommendations(intent, risk_level)
        
        # Update conversation history
        self.conversation_history.append({
            'user_input': user_input,
            'response': response,
            'intent': intent,
            'risk_level': risk_level,
            'timestamp': datetime.now()
        })
        
        # Create response object
        conversation_response = ConversationResponse(
            response=response,
            confidence=intent_confidence,
            intent=intent,
            risk_level=risk_level,
            category=category,
            recommendations=recommendations,
            context_awareness={
                'entities': entities,
                'conversation_length': len(self.conversation_history),
                'previous_topics': [h['intent'] for h in self.conversation_history[-5:]]
            },
            semantic_similarity=semantic_similarity,
            timestamp=datetime.now()
        )
        
        return conversation_response
        
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history"""
        if not self.conversation_history:
            return {"message": "No conversation history available"}
            
        return {
            "total_interactions": len(self.conversation_history),
            "intents_discussed": list(set(h['intent'] for h in self.conversation_history)),
            "risk_levels_encountered": list(set(h['risk_level'] for h in self.conversation_history)),
            "last_interaction": self.conversation_history[-1]['timestamp'].isoformat(),
            "session_duration": (
                self.conversation_history[-1]['timestamp'] - 
                self.conversation_history[0]['timestamp']
            ).total_seconds() if len(self.conversation_history) > 1 else 0
        }

# Example usage and testing
async def main():
    """Main function for testing the Advanced Medical AI"""
    print("🩺 Advanced Medical AI Conversational Engine")
    print("=" * 60)
    
    # Initialize AI
    ai = AdvancedMedicalAI()
    
    # Test queries
    test_queries = [
        "What is a headache?",
        "I have chest pain and difficulty breathing",
        "How do I prevent diabetes?",
        "What are the side effects of aspirin?",
        "I feel dizzy and nauseous"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        response = await ai.process_conversation(query)
        
        print(f"📝 Response: {response.response}")
        print(f"🎯 Intent: {response.intent}")
        print(f"⚠️  Risk Level: {response.risk_level}")
        print(f"📊 Confidence: {response.confidence:.2f}")
        print(f"💡 Recommendations: {', '.join(response.recommendations)}")
        print(f"🔗 Semantic Similarity: {response.semantic_similarity:.2f}")
        
    print("\n" + "=" * 60)
    print("📊 Conversation Summary:")
    summary = ai.get_conversation_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
