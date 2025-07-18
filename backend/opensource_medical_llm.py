"""
Open-Source Medical LLM Implementation
=====================================

This module provides a complete open-source medical AI system using:
- Local LLM models (Llama, Mistral, etc.)
- Vector database for medical knowledge
- Fine-tuning capabilities
- No external API dependencies
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import re

try:
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        pipeline,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        TextGenerationPipeline
    )
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers not available: {e}")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

logger = logging.getLogger(__name__)

@dataclass
class MedicalResponse:
    """Enhanced medical response structure"""
    response: str
    confidence_score: float
    risk_level: str  # low, moderate, high, critical
    is_emergency: bool
    sources: List[str]
    recommendations: List[str]
    medical_disclaimer: str
    requires_professional_consultation: bool
    category: str
    timestamp: datetime
    related_conditions: List[str] = None
    drug_interactions: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class MedicalKnowledgeBase:
    """Vector database for medical knowledge"""
    
    def __init__(self, persist_directory: str = "./medical_knowledge_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.medical_data = []
        
        if CHROMADB_AVAILABLE:
            self._initialize_vector_db()
        
        # Load medical training data
        self._load_medical_data()
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="medical_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            self.client = None
            self.collection = None
    
    def _load_medical_data(self):
        """Load medical training data"""
        try:
            data_file = Path("../data/medical_training_data.json")
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    self.medical_data = json.load(f)
                logger.info(f"Loaded {len(self.medical_data)} medical data entries")
            else:
                logger.warning("Medical training data file not found")
                self.medical_data = []
        except Exception as e:
            logger.error(f"Error loading medical data: {e}")
            self.medical_data = []
    
    def populate_knowledge_base(self):
        """Populate vector database with medical knowledge"""
        if not self.collection or not self.medical_data:
            logger.warning("Cannot populate knowledge base - missing collection or data")
            return False
        
        try:
            # Prepare documents for vectorization
            documents = []
            metadatas = []
            ids = []
            
            for i, item in enumerate(self.medical_data):
                # Combine input and output for better context
                document = f"Query: {item['input']}\nAnswer: {item['output']}"
                documents.append(document)
                
                metadata = {
                    "category": item.get('category', 'general'),
                    "risk_level": item.get('risk_level', 'low'),
                    "intent": item.get('intent', 'unknown')
                }
                metadatas.append(metadata)
                ids.append(f"medical_{i}")
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error populating knowledge base: {e}")
            return False
    
    def search_knowledge(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search medical knowledge base"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            knowledge_items = []
            for i, doc in enumerate(results['documents'][0]):
                knowledge_items.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

class OpenSourceMedicalLLM:
    """Open-source medical language model implementation"""
    
    # Available open-source models
    AVAILABLE_MODELS = {
        "microsoft/DialoGPT-medium": {
            "name": "DialoGPT Medium",
            "description": "Conversational AI model",
            "memory_gb": 2
        },
        "microsoft/DialoGPT-large": {
            "name": "DialoGPT Large", 
            "description": "Larger conversational model",
            "memory_gb": 4
        },
        "distilgpt2": {
            "name": "DistilGPT2",
            "description": "Lightweight GPT-2 model",
            "memory_gb": 1
        },
        "gpt2": {
            "name": "GPT-2",
            "description": "OpenAI's GPT-2 model",
            "memory_gb": 2
        },
        "microsoft/GODEL-v1_1-base-seq2seq": {
            "name": "GODEL",
            "description": "Goal-oriented dialog model",
            "memory_gb": 3
        }
    }
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 use_quantization: bool = True,
                 device: str = "auto"):
        
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.knowledge_base = MedicalKnowledgeBase()
        
        # Medical safety patterns
        self.emergency_patterns = [
            r"chest pain|heart attack|stroke|can't breathe|difficulty breathing",
            r"unconscious|passed out|severe bleeding|choking",
            r"emergency|urgent|serious|critical|severe pain"
        ]
        
        self.medical_disclaimer = (
            "⚠️ Medical Disclaimer: This information is for educational purposes only "
            "and should not replace professional medical advice. Always consult with "
            "a qualified healthcare provider for proper diagnosis and treatment."
        )
        
        # Initialize model
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
            
        # Populate knowledge base if empty
        if self.knowledge_base.collection and self.knowledge_base.collection.count() == 0:
            self.knowledge_base.populate_knowledge_base()
    
    def _initialize_model(self):
        """Initialize the local language model"""
        try:
            logger.info(f"Initializing model: {self.model_name}")
            
            # Configure quantization for memory efficiency
            if self.use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device != "auto" else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def _classify_query(self, query: str) -> str:
        """Classify medical query"""
        query_lower = query.lower()
        
        categories = {
            "symptoms": ["pain", "ache", "hurt", "feel", "sick", "nausea", "fever", "cough", "headache"],
            "medications": ["medication", "drug", "medicine", "pill", "tablet", "prescription", "dosage"],
            "conditions": ["disease", "condition", "illness", "disorder", "syndrome", "infection", "diabetes"],
            "treatments": ["treatment", "therapy", "procedure", "surgery", "operation", "cure"],
            "prevention": ["prevent", "prevention", "avoid", "protect", "vaccine", "immunization"],
            "emergency": ["emergency", "urgent", "severe", "serious", "critical", "chest pain", "stroke"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _is_emergency(self, query: str) -> bool:
        """Check if query indicates medical emergency"""
        query_lower = query.lower()
        
        for pattern in self.emergency_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _generate_response(self, query: str, context: str = "") -> str:
        """Generate response using local model"""
        if not self.pipeline:
            return self._get_fallback_response(query)
        
        try:
            # Create medical prompt
            system_prompt = """You are a helpful medical AI assistant. Provide accurate, evidence-based medical information while emphasizing the importance of professional medical consultation. Never provide specific diagnoses or prescriptions."""
            
            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
            
            # Generate response
            response = self.pipeline(
                full_prompt,
                max_length=len(full_prompt) + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            answer = generated_text[len(full_prompt):].strip()
            
            # Clean up response
            answer = self._clean_response(answer)
            
            return answer if answer else self._get_fallback_response(query)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(query)
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate generated response"""
        # Remove unwanted tokens
        response = response.replace("<|endoftext|>", "").strip()
        
        # Split by common end markers
        for marker in ["\n\n", "Question:", "Answer:", "Context:"]:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # Ensure reasonable length
        if len(response) > 500:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3]) + '.'
        
        return response
    
    def _get_fallback_response(self, query: str) -> str:
        """Generate fallback response when model fails"""
        query_lower = query.lower()
        
        # Specific condition responses
        if "headache" in query_lower:
            return "Headaches can be caused by various factors including stress, dehydration, lack of sleep, or underlying conditions. For persistent or severe headaches, please consult a healthcare provider."
        
        elif "fever" in query_lower:
            return "Fever is your body's natural response to infection. Monitor your temperature and seek medical attention if it's high (>103°F) or persistent."
        
        elif "diabetes" in query_lower:
            return "Diabetes is a condition affecting blood sugar levels. Common symptoms include excessive thirst, frequent urination, and fatigue. Please consult a healthcare provider for proper diagnosis and management."
        
        elif "blood pressure" in query_lower:
            return "Blood pressure is the force of blood against artery walls. Normal blood pressure is less than 120/80 mmHg. High blood pressure can be managed through lifestyle changes and medication."
        
        # General response
        return "I can provide general health information, but for specific medical concerns, please consult with a qualified healthcare professional who can properly assess your situation."
    
    def _get_recommendations(self, category: str) -> List[str]:
        """Get category-specific recommendations"""
        recommendations = {
            "symptoms": [
                "Monitor symptoms closely",
                "Keep a symptom diary",
                "Stay hydrated and rest",
                "Consult healthcare provider if symptoms worsen"
            ],
            "medications": [
                "Take as prescribed by healthcare provider",
                "Check for drug interactions",
                "Store medications properly",
                "Consult pharmacist for questions"
            ],
            "conditions": [
                "Follow treatment plan",
                "Regular medical check-ups",
                "Maintain healthy lifestyle",
                "Stay informed about your condition"
            ],
            "emergency": [
                "Call 911 immediately",
                "Go to nearest emergency room",
                "Do not delay medical attention",
                "Stay calm and follow emergency protocols"
            ],
            "general": [
                "Maintain regular healthcare visits",
                "Stay informed about health topics",
                "Practice preventive care",
                "Ask questions during medical visits"
            ]
        }
        
        return recommendations.get(category, recommendations["general"])
    
    async def process_query(self, query: str, patient_id: Optional[str] = None) -> MedicalResponse:
        """Process medical query and return comprehensive response"""
        try:
            # Classify query
            category = self._classify_query(query)
            is_emergency = self._is_emergency(query)
            
            # Handle emergency
            if is_emergency:
                return MedicalResponse(
                    response="🚨 This appears to be a medical emergency. Please call emergency services immediately (911 in the US) or go to the nearest emergency room. Do not delay seeking immediate medical attention.",
                    confidence_score=1.0,
                    risk_level="critical",
                    is_emergency=True,
                    sources=["Emergency medical protocols"],
                    recommendations=["Call 911 immediately", "Go to nearest emergency room"],
                    medical_disclaimer=self.medical_disclaimer,
                    requires_professional_consultation=True,
                    category="emergency",
                    timestamp=datetime.now()
                )
            
            # Search knowledge base
            knowledge_items = self.knowledge_base.search_knowledge(query)
            context = ""
            sources = ["Medical knowledge base"]
            
            if knowledge_items:
                # Use best matching knowledge
                best_match = knowledge_items[0]
                context = best_match['document']
                sources.append(f"Medical database (confidence: {1-best_match['distance']:.2f})")
            
            # Generate response
            response = self._generate_response(query, context)
            
            # Add medical disclaimer
            response += f"\n\n{self.medical_disclaimer}"
            
            # Determine risk level
            risk_level = "low"
            if category in ["symptoms", "conditions"]:
                risk_level = "moderate"
            elif category == "emergency":
                risk_level = "critical"
            
            # Get recommendations
            recommendations = self._get_recommendations(category)
            
            return MedicalResponse(
                response=response,
                confidence_score=0.8 if knowledge_items else 0.6,
                risk_level=risk_level,
                is_emergency=False,
                sources=sources,
                recommendations=recommendations,
                medical_disclaimer=self.medical_disclaimer,
                requires_professional_consultation=category in ["symptoms", "conditions"],
                category=category,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return MedicalResponse(
                response="I apologize, but I'm having trouble processing your medical query right now. Please consult with a healthcare professional for medical concerns.",
                confidence_score=0.0,
                risk_level="unknown",
                is_emergency=False,
                sources=["System error"],
                recommendations=["Consult healthcare professional", "Try again later"],
                medical_disclaimer=self.medical_disclaimer,
                requires_professional_consultation=True,
                category="error",
                timestamp=datetime.now()
            )
    
    def fine_tune_model(self, training_data: List[Dict], output_dir: str = "./fine_tuned_model"):
        """Fine-tune model on medical data"""
        if not TRANSFORMERS_AVAILABLE or not self.model:
            logger.error("Model not available for fine-tuning")
            return False
        
        try:
            # Prepare training data
            train_texts = []
            for item in training_data:
                # Format as conversation
                text = f"Human: {item['input']}\nAssistant: {item['output']}<|endoftext|>"
                train_texts.append(text)
            
            # Create dataset
            dataset = Dataset.from_dict({"text": train_texts})
            
            # Tokenize
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                warmup_steps=100,
                logging_steps=10,
                save_steps=100,
                evaluation_strategy="steps",
                eval_steps=100,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,
                push_to_hub=False,
                learning_rate=5e-5,
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=False,
                gradient_accumulation_steps=2,
                max_grad_norm=1.0,
                lr_scheduler_type="cosine"
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Split dataset
            train_size = int(0.9 * len(tokenized_dataset))
            eval_size = len(tokenized_dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                tokenized_dataset, [train_size, eval_size]
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Train model
            logger.info("Starting fine-tuning...")
            trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "model_info": self.AVAILABLE_MODELS.get(self.model_name, {}),
            "is_loaded": self.model is not None,
            "device": str(self.model.device) if self.model else "N/A",
            "quantization": self.use_quantization,
            "knowledge_base_size": self.knowledge_base.collection.count() if self.knowledge_base.collection else 0
        }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available open-source models"""
        return cls.AVAILABLE_MODELS
