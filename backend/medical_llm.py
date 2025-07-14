import openai
import os
import json
import logging
from typing import Dict, List, Optional, Any  # Added Any to the imports
from datetime import datetime
import asyncio
from dataclasses import dataclass
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")
    # Create placeholder classes to avoid import errors
    class AutoModelForCausalLM:
        pass
    class AutoTokenizer:
        pass

logger = logging.getLogger(__name__)

@dataclass
class MedicalResponse:
    """Enhanced structure for medical AI responses"""
    response: str
    confidence_score: float
    risk_level: str
    is_emergency: bool
    sources: List[str]
    recommendations: List[str]
    medical_disclaimer: str
    requires_professional_consultation: bool
    category: str
    timestamp: datetime
    
    # Legacy compatibility
    @property
    def confidence(self) -> float:
        return self.confidence_score

class MedicalLLMClient:
    """Advanced Medical LLM Client with OpenAI and local model integration"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", use_local_model: bool = False, local_model_name: str = "microsoft/DialoGPT-medium"):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        self.model = model
        self.use_local_model = use_local_model
        self.local_model_name = local_model_name
        self.client = None
        self.local_model = None
        self.tokenizer = None
        self.medical_disclaimer = (
            "⚠️ Medical Disclaimer: This information is for educational purposes only "
            "and should not replace professional medical advice. Always consult with "
            "a qualified healthcare provider for proper diagnosis and treatment."
        )
        
        # Medical categories for classification
        self.medical_categories = {
            "symptoms": ["symptom", "pain", "ache", "hurt", "feel", "sick", "nausea", "fever", "cough"],
            "medications": ["medication", "drug", "medicine", "pill", "tablet", "prescription", "dosage"],
            "conditions": ["disease", "condition", "illness", "disorder", "syndrome", "infection"],
            "treatments": ["treatment", "therapy", "procedure", "surgery", "operation", "cure"],
            "prevention": ["prevent", "prevention", "avoid", "protect", "vaccine", "immunization"],
            "emergency": ["emergency", "urgent", "severe", "serious", "critical", "chest pain", "stroke"]
        }
        
        # Initialize clients
        # Automatic model selection
        self._select_and_initialize_model()
    
    def _select_and_initialize_model(self):
        """Select and initialize the best available model"""
        # Priority: OpenAI API > Local Model > Mock responses
        if self.api_key and self.api_key != "YOUR_OPENAI_API_KEY":
            self._initialize_client()
        elif TRANSFORMERS_AVAILABLE:
            self._initialize_local_model()
        else:
            logger.warning("No valid model or API key provided. Reverting to mock responses.")

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if self.api_key and self.api_key != "YOUR_OPENAI_API_KEY":
                openai.api_key = self.api_key
                self.client = openai
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not provided. Using mock responses.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def _initialize_local_model(self):
        """Initialize local transformer model"""
        try:
            logger.info(f"Loading local model: {self.local_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(self.local_model_name)
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Local model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            self.local_model = None
            self.tokenizer = None
    
    def _classify_medical_query(self, query: str) -> str:
        """Classify the medical query into categories"""
        query_lower = query.lower()
        
        for category, keywords in self.medical_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _is_emergency_query(self, query: str) -> bool:
        """Check if query indicates a medical emergency"""
        emergency_keywords = [
            "chest pain", "heart attack", "stroke", "difficulty breathing",
            "severe bleeding", "unconscious", "emergency", "urgent",
            "severe pain", "can't breathe", "choking"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in emergency_keywords)
    
    def _get_medical_system_prompt(self) -> str:
        """Get the system prompt for medical AI"""
        return """You are a knowledgeable medical AI assistant. Your role is to:

1. Provide accurate, evidence-based medical information
2. Help users understand symptoms, conditions, and treatments
3. Always emphasize the importance of professional medical consultation
4. Never provide specific diagnoses or prescriptions
5. Be empathetic and supportive
6. Clearly state when emergency medical care is needed

Guidelines:
- Always include medical disclaimers
- Encourage professional medical consultation for serious concerns
- Provide general health information, not specific medical advice
- Be clear about limitations of AI medical assistance
- Focus on education and awareness

Remember: You are providing information, not medical diagnosis or treatment."""
    
    async def process_query(self, 
                           query: str, 
                           patient_id: Optional[str] = None,
                           context: Dict[str, Any] = None) -> 'MedicalResponse':
        """Enhanced query processing with context awareness"""
        return await self.query_medical_llm(query, patient_id, context)
    
    async def query_medical_llm(self, 
                               user_query: str, 
                               patient_id: Optional[str] = None,
                               context: Dict[str, Any] = None) -> MedicalResponse:
        """Query the medical LLM with user input"""
        try:
            # Classify the query
            category = self._classify_medical_query(user_query)
            is_emergency = self._is_emergency_query(user_query)
            
            # Handle emergency queries
            if is_emergency:
                return MedicalResponse(
                    response="🚨 This appears to be a medical emergency. Please call emergency services immediately (911 in the US) or go to the nearest emergency room. Do not delay seeking immediate medical attention.",
                    confidence_score=1.0,
                    risk_level="critical",
                    is_emergency=True,
                    sources=["Emergency medical protocols"],
                    recommendations=["Call 911 immediately", "Go to nearest emergency room", "Do not delay medical attention"],
                    medical_disclaimer=self.medical_disclaimer,
                    requires_professional_consultation=True,
                    category="emergency",
                    timestamp=datetime.now()
                )
            
            # Query OpenAI if available
            if self.client and self.api_key != "YOUR_OPENAI_API_KEY":
                response = await self._query_openai(user_query, category)
            elif self.use_local_model and self.local_model and self.tokenizer:
                response = await self._query_local_model(user_query, category)
            else:
                response = self._get_mock_response(user_query, category)
            
            # Determine risk level and recommendations
            risk_level = "moderate" if category in ["symptoms", "conditions"] else "low"
            sources = ["Medical knowledge base", "AI medical assistant"]
            recommendations = self._get_recommendations(category)
            
            return MedicalResponse(
                response=response,
                confidence_score=0.8,
                risk_level=risk_level,
                is_emergency=False,
                sources=sources,
                recommendations=recommendations,
                medical_disclaimer=self.medical_disclaimer,
                requires_professional_consultation=category in ["symptoms", "conditions", "emergency"],
                category=category,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in medical LLM query: {e}")
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
    
    async def _query_openai(self, query: str, category: str) -> str:
        """Query OpenAI API for medical information"""
        try:
            system_prompt = self._get_medical_system_prompt()
            
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Category: {category}\nQuery: {query}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_mock_response(query, category)
    
    async def _query_local_model(self, query: str, category: str) -> str:
        """Query local transformer model for medical information"""
        try:
            # Prepare the input with medical context
            system_prompt = self._get_medical_system_prompt()
            full_prompt = f"{system_prompt}\n\nCategory: {category}\nQuery: {query}\nResponse:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(full_prompt):].strip()
            
            # Fallback to mock response if generation fails
            if not response or len(response) < 10:
                return self._get_mock_response(query, category)
            
            return response
            
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return self._get_mock_response(query, category)
    
    def fine_tune_model(self, training_data: List[Dict], save_path: str = "./fine_tuned_medical_model"):
        """Fine-tune the local model on medical data"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available for fine-tuning")
            return False
        
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from datasets import Dataset
            
            # Prepare dataset
            dataset = Dataset.from_list(training_data)
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=save_path,
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                save_steps=500,
                save_total_limit=2,
                prediction_loss_only=True,
                logging_dir=f"{save_path}/logs",
                logging_steps=100,
                warmup_steps=500,
                learning_rate=5e-5
            )
            
            # Trainer
            trainer = Trainer(
                model=self.local_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
            )
            
            # Fine-tune
            trainer.train()
            
            # Save model
            trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"Fine-tuned model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning error: {e}")
            return False
    
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
            "treatments": [
                "Discuss options with healthcare provider",
                "Understand risks and benefits",
                "Follow post-treatment instructions",
                "Monitor for side effects"
            ],
            "prevention": [
                "Maintain regular exercise",
                "Follow healthy diet",
                "Get recommended screenings",
                "Practice good hygiene"
            ],
            "general": [
                "Maintain regular healthcare visits",
                "Stay informed about health topics",
                "Practice preventive care",
                "Ask questions during medical visits"
            ]
        }
        
        return recommendations.get(category, recommendations["general"])
    
    def _get_mock_response(self, query: str, category: str) -> str:
        """Generate enhanced medical responses for testing"""
        query_lower = query.lower()
        
        # Enhanced responses based on specific queries
        if "headache" in query_lower:
            return "Headaches can be caused by various factors including stress, dehydration, lack of sleep, eye strain, or underlying medical conditions. Common types include tension headaches, migraines, and cluster headaches. For mild headaches, try rest, hydration, and over-the-counter pain relievers as directed. Seek medical attention if headaches are severe, sudden, persistent, or accompanied by fever, vision changes, or neurological symptoms."
        
        elif "fever" in query_lower:
            return "Fever is a common symptom indicating that your body is fighting an infection. Normal body temperature is around 98.6°F (37°C). A fever is generally considered 100.4°F (38°C) or higher. For mild fevers, rest, fluids, and fever reducers like acetaminophen or ibuprofen can help. Seek medical care if fever exceeds 103°F (39.4°C), persists for more than 3 days, or is accompanied by severe symptoms like difficulty breathing, chest pain, or severe headache."
        
        elif "diabetes" in query_lower:
            return "Diabetes is a condition where blood sugar levels are too high. Type 1 diabetes occurs when the body doesn't produce insulin, while Type 2 diabetes occurs when the body doesn't use insulin properly. Common symptoms include excessive thirst, frequent urination, fatigue, and blurred vision. Management includes blood sugar monitoring, proper diet, regular exercise, and medication as prescribed. Regular check-ups with healthcare providers are essential for proper management."
        
        elif "blood pressure" in query_lower or "hypertension" in query_lower:
            return "High blood pressure (hypertension) is often called the 'silent killer' because it typically has no symptoms. Normal blood pressure is less than 120/80 mmHg. High blood pressure can lead to heart disease, stroke, and kidney problems. Lifestyle changes that can help include regular exercise, healthy diet (low sodium, high potassium), maintaining healthy weight, limiting alcohol, and managing stress. Medications may also be prescribed by your healthcare provider."
        
        elif "chest pain" in query_lower:
            return "🚨 IMPORTANT: Chest pain can be a sign of a heart attack or other serious condition. If you're experiencing severe, crushing, or persistent chest pain, especially with shortness of breath, sweating, nausea, or pain radiating to arm/jaw, call 911 immediately. Other causes of chest pain can include muscle strain, acid reflux, or anxiety. However, any chest pain should be evaluated by a healthcare professional promptly."
        
        # Category-based responses
        mock_responses = {
            "symptoms": f"Based on your symptoms, I can provide general information. Common symptoms like those you're describing can have various causes. It's important to monitor your symptoms and consult with a healthcare professional for proper evaluation, especially if symptoms persist, worsen, or are accompanied by concerning signs.",
            
            "medications": f"Regarding medications, I can share general information about how different classes of medications work and their common uses. However, medication decisions should always be made with your healthcare provider or pharmacist, who can consider your specific health conditions, other medications, and individual factors.",
            
            "conditions": f"I can provide educational information about various medical conditions, including their common symptoms, risk factors, and general management approaches. However, for accurate diagnosis and personalized treatment plans, it's essential to work with qualified healthcare professionals.",
            
            "treatments": f"There are many treatment approaches available for different medical conditions, ranging from lifestyle modifications to medications and procedures. The best treatment plan depends on your specific situation, medical history, and individual factors that only a healthcare provider can properly assess.",
            
            "prevention": f"Prevention is indeed crucial for maintaining good health. This includes regular exercise, balanced nutrition, adequate sleep, stress management, avoiding harmful substances, and staying up-to-date with recommended screenings and vaccinations. Your healthcare provider can help develop a personalized prevention plan.",
            
            "general": f"I'm here to provide general health education and information. While I can share knowledge about medical topics, I cannot replace professional medical advice. For specific health concerns, symptoms, or medical questions, please consult with qualified healthcare professionals."
        }
        
        return mock_responses.get(category, mock_responses["general"])
    
    def get_health_tips(self) -> List[str]:
        """Get general health tips"""
        return [
            "Stay hydrated by drinking adequate water daily",
            "Maintain regular physical activity suitable for your fitness level",
            "Follow a balanced diet rich in fruits, vegetables, and whole grains",
            "Get adequate sleep (7-9 hours for most adults)",
            "Manage stress through relaxation techniques",
            "Schedule regular check-ups with your healthcare provider",
            "Avoid smoking and limit alcohol consumption",
            "Practice good hygiene to prevent infections"
        ]
    
    def get_emergency_contacts(self) -> Dict[str, str]:
        """Get emergency contact information"""
        return {
            "Emergency Services (US)": "911",
            "Poison Control": "1-800-222-1222",
            "Crisis Text Line": "Text HOME to 741741",
            "National Suicide Prevention Lifeline": "988"
        }