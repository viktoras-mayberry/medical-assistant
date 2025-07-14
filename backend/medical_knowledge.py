import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class MedicalCategory(Enum):
    SYMPTOMS = "symptoms"
    CONDITIONS = "conditions"
    MEDICATIONS = "medications"
    TREATMENTS = "treatments"
    PREVENTION = "prevention"
    EMERGENCY = "emergency"
    DIAGNOSTICS = "diagnostics"
    ANATOMY = "anatomy"
    PROCEDURES = "procedures"
    NUTRITION = "nutrition"
    MENTAL_HEALTH = "mental_health"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    GENERAL = "general"

@dataclass
class MedicalKnowledge:
    """Structure for medical knowledge entries"""
    id: str
    title: str
    category: MedicalCategory
    content: str
    keywords: List[str]
    severity_level: str  # low, medium, high, critical
    requires_professional: bool
    related_conditions: List[str]
    common_symptoms: List[str]
    treatment_options: List[str]
    prevention_measures: List[str]
    when_to_seek_help: List[str]
    created_at: datetime
    updated_at: datetime

class MedicalKnowledgeEngine:
    """Advanced medical knowledge processing engine"""
    
    def __init__(self):
        self.knowledge_base: Dict[str, MedicalKnowledge] = {}
        self.symptom_checker = SymptomChecker()
        self.drug_interaction_checker = DrugInteractionChecker()
        self.emergency_detector = EmergencyDetector()
        self.medical_calculator = MedicalCalculator()
        
        # Load medical knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the comprehensive medical knowledge base"""
        # Common medical conditions
        conditions = [
            {
                "id": "hypertension",
                "title": "High Blood Pressure (Hypertension)",
                "category": MedicalCategory.CONDITIONS,
                "content": "Hypertension is a common condition where blood pressure is consistently elevated. It's often called the 'silent killer' because it typically has no symptoms until serious complications develop.",
                "keywords": ["high blood pressure", "hypertension", "BP", "blood pressure"],
                "severity_level": "medium",
                "requires_professional": True,
                "related_conditions": ["heart disease", "stroke", "kidney disease"],
                "common_symptoms": ["headaches", "dizziness", "chest pain", "difficulty breathing"],
                "treatment_options": ["lifestyle changes", "ACE inhibitors", "beta blockers", "diuretics"],
                "prevention_measures": ["regular exercise", "healthy diet", "limit sodium", "maintain healthy weight"],
                "when_to_seek_help": ["BP > 180/120", "severe headache", "chest pain", "vision problems"]
            },
            {
                "id": "diabetes",
                "title": "Diabetes Mellitus",
                "category": MedicalCategory.CONDITIONS,
                "content": "Diabetes is a group of metabolic disorders characterized by high blood sugar levels. Type 1 is autoimmune, Type 2 is insulin resistance.",
                "keywords": ["diabetes", "blood sugar", "glucose", "insulin"],
                "severity_level": "high",
                "requires_professional": True,
                "related_conditions": ["heart disease", "kidney disease", "neuropathy", "retinopathy"],
                "common_symptoms": ["excessive thirst", "frequent urination", "fatigue", "blurred vision"],
                "treatment_options": ["insulin therapy", "metformin", "lifestyle modifications", "glucose monitoring"],
                "prevention_measures": ["healthy diet", "regular exercise", "weight management", "avoid smoking"],
                "when_to_seek_help": ["blood sugar > 250", "ketoacidosis symptoms", "severe hypoglycemia"]
            },
            {
                "id": "covid19",
                "title": "COVID-19 (Coronavirus Disease)",
                "category": MedicalCategory.CONDITIONS,
                "content": "COVID-19 is an infectious disease caused by SARS-CoV-2 virus. Symptoms range from mild to severe, with potential for serious complications.",
                "keywords": ["covid", "coronavirus", "covid-19", "sars-cov-2"],
                "severity_level": "medium",
                "requires_professional": True,
                "related_conditions": ["pneumonia", "ARDS", "blood clots", "long covid"],
                "common_symptoms": ["fever", "cough", "fatigue", "loss of taste/smell", "body aches"],
                "treatment_options": ["rest", "hydration", "antivirals", "oxygen therapy", "steroids"],
                "prevention_measures": ["vaccination", "mask wearing", "social distancing", "hand hygiene"],
                "when_to_seek_help": ["difficulty breathing", "chest pain", "confusion", "persistent fever"]
            }
        ]
        
        # Common medications
        medications = [
            {
                "id": "aspirin",
                "title": "Aspirin (Acetylsalicylic Acid)",
                "category": MedicalCategory.MEDICATIONS,
                "content": "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used for pain relief, fever reduction, and cardiovascular protection.",
                "keywords": ["aspirin", "acetylsalicylic acid", "ASA", "pain relief"],
                "severity_level": "low",
                "requires_professional": False,
                "related_conditions": ["heart attack", "stroke", "arthritis", "fever"],
                "common_symptoms": ["pain", "fever", "inflammation", "headache"],
                "treatment_options": ["81mg daily", "325mg as needed", "follow physician guidance"],
                "prevention_measures": ["take with food", "avoid if allergic", "monitor for bleeding"],
                "when_to_seek_help": ["severe bleeding", "allergic reaction", "stomach pain", "tinnitus"]
            },
            {
                "id": "metformin",
                "title": "Metformin",
                "category": MedicalCategory.MEDICATIONS,
                "content": "Metformin is the first-line medication for type 2 diabetes, working by decreasing glucose production and improving insulin sensitivity.",
                "keywords": ["metformin", "diabetes medication", "blood sugar", "antidiabetic"],
                "severity_level": "medium",
                "requires_professional": True,
                "related_conditions": ["type 2 diabetes", "prediabetes", "PCOS"],
                "common_symptoms": ["high blood sugar", "insulin resistance"],
                "treatment_options": ["500mg twice daily", "extended release", "titrate dose"],
                "prevention_measures": ["take with meals", "monitor kidney function", "avoid alcohol"],
                "when_to_seek_help": ["lactic acidosis", "severe GI upset", "kidney problems"]
            }
        ]
        
        # Emergency conditions
        emergencies = [
            {
                "id": "heart_attack",
                "title": "Heart Attack (Myocardial Infarction)",
                "category": MedicalCategory.EMERGENCY,
                "content": "A heart attack occurs when blood flow to part of the heart muscle is blocked, usually by a blood clot.",
                "keywords": ["heart attack", "myocardial infarction", "MI", "chest pain"],
                "severity_level": "critical",
                "requires_professional": True,
                "related_conditions": ["coronary artery disease", "atherosclerosis", "hypertension"],
                "common_symptoms": ["severe chest pain", "shortness of breath", "nausea", "sweating"],
                "treatment_options": ["call 911 immediately", "aspirin", "nitroglycerin", "emergency surgery"],
                "prevention_measures": ["healthy lifestyle", "regular exercise", "manage risk factors"],
                "when_to_seek_help": ["ANY chest pain", "shortness of breath", "arm/jaw pain", "nausea"]
            },
            {
                "id": "stroke",
                "title": "Stroke (Cerebrovascular Accident)",
                "category": MedicalCategory.EMERGENCY,
                "content": "A stroke occurs when blood supply to part of the brain is interrupted or reduced, preventing brain tissue from getting oxygen.",
                "keywords": ["stroke", "CVA", "brain attack", "FAST"],
                "severity_level": "critical",
                "requires_professional": True,
                "related_conditions": ["atrial fibrillation", "hypertension", "diabetes"],
                "common_symptoms": ["face drooping", "arm weakness", "speech difficulty", "sudden confusion"],
                "treatment_options": ["call 911 immediately", "tPA", "mechanical thrombectomy", "rehabilitation"],
                "prevention_measures": ["control blood pressure", "manage diabetes", "quit smoking"],
                "when_to_seek_help": ["FAST symptoms", "sudden severe headache", "vision loss", "confusion"]
            }
        ]
        
        # Combine all knowledge entries
        all_entries = conditions + medications + emergencies
        
        # Convert to MedicalKnowledge objects
        for entry in all_entries:
            knowledge = MedicalKnowledge(
                id=entry["id"],
                title=entry["title"],
                category=entry["category"],
                content=entry["content"],
                keywords=entry["keywords"],
                severity_level=entry["severity_level"],
                requires_professional=entry["requires_professional"],
                related_conditions=entry["related_conditions"],
                common_symptoms=entry["common_symptoms"],
                treatment_options=entry["treatment_options"],
                prevention_measures=entry["prevention_measures"],
                when_to_seek_help=entry["when_to_seek_help"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.knowledge_base[entry["id"]] = knowledge
    
    def search_knowledge(self, query: str) -> List[MedicalKnowledge]:
        """Search medical knowledge base for relevant information"""
        query_lower = query.lower()
        matches = []
        
        for knowledge in self.knowledge_base.values():
            # Check keywords
            if any(keyword.lower() in query_lower for keyword in knowledge.keywords):
                matches.append(knowledge)
                continue
            
            # Check title
            if knowledge.title.lower() in query_lower:
                matches.append(knowledge)
                continue
            
            # Check content
            if any(word in knowledge.content.lower() for word in query_lower.split()):
                matches.append(knowledge)
        
        # Sort by relevance (emergency conditions first)
        matches.sort(key=lambda x: (
            x.category == MedicalCategory.EMERGENCY,
            x.severity_level == "critical",
            x.severity_level == "high"
        ), reverse=True)
        
        return matches
    
    def get_comprehensive_info(self, condition_id: str) -> Optional[Dict]:
        """Get comprehensive information about a specific condition"""
        if condition_id not in self.knowledge_base:
            return None
        
        knowledge = self.knowledge_base[condition_id]
        
        return {
            "basic_info": {
                "title": knowledge.title,
                "category": knowledge.category.value,
                "description": knowledge.content,
                "severity": knowledge.severity_level
            },
            "clinical_info": {
                "symptoms": knowledge.common_symptoms,
                "related_conditions": knowledge.related_conditions,
                "treatment_options": knowledge.treatment_options,
                "prevention": knowledge.prevention_measures
            },
            "guidance": {
                "when_to_seek_help": knowledge.when_to_seek_help,
                "requires_professional": knowledge.requires_professional,
                "emergency_level": knowledge.severity_level == "critical"
            }
        }
    
    def check_drug_interactions(self, medications: List[str]) -> Dict:
        """Check for potential drug interactions"""
        return self.drug_interaction_checker.check_interactions(medications)
    
    def assess_symptoms(self, symptoms: List[str]) -> Dict:
        """Assess symptoms and provide potential conditions"""
        return self.symptom_checker.assess_symptoms(symptoms)
    
    def detect_emergency(self, query: str) -> Dict:
        """Detect if query indicates medical emergency"""
        return self.emergency_detector.analyze_query(query)
    
    def calculate_medical_values(self, calculation_type: str, values: Dict) -> Dict:
        """Perform medical calculations (BMI, dosage, etc.)"""
        return self.medical_calculator.calculate(calculation_type, values)

class SymptomChecker:
    """Symptom assessment and condition matching"""
    
    def __init__(self):
        self.symptom_patterns = {
            "chest_pain": ["chest pain", "chest discomfort", "chest pressure", "chest tightness"],
            "shortness_of_breath": ["shortness of breath", "difficulty breathing", "breathless", "SOB"],
            "headache": ["headache", "head pain", "migraine", "head ache"],
            "fever": ["fever", "high temperature", "hot", "feverish"],
            "fatigue": ["tired", "fatigue", "exhausted", "weakness", "weak"],
            "nausea": ["nausea", "nauseous", "sick to stomach", "queasy"],
            "dizziness": ["dizzy", "lightheaded", "vertigo", "spinning"],
            "cough": ["cough", "coughing", "hacking"]
        }
    
    def assess_symptoms(self, symptoms: List[str]) -> Dict:
        """Assess symptoms and return potential conditions"""
        matched_symptoms = []
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for pattern_name, patterns in self.symptom_patterns.items():
                if any(pattern in symptom_lower for pattern in patterns):
                    matched_symptoms.append(pattern_name)
        
        # Simple rule-based assessment
        potential_conditions = []
        urgency_level = "low"
        
        if "chest_pain" in matched_symptoms and "shortness_of_breath" in matched_symptoms:
            potential_conditions.append("heart attack")
            urgency_level = "critical"
        elif "chest_pain" in matched_symptoms:
            potential_conditions.extend(["heart attack", "angina", "acid reflux"])
            urgency_level = "high"
        elif "fever" in matched_symptoms and "cough" in matched_symptoms:
            potential_conditions.extend(["respiratory infection", "flu", "pneumonia"])
            urgency_level = "medium"
        elif "headache" in matched_symptoms and "fever" in matched_symptoms:
            potential_conditions.extend(["viral infection", "meningitis"])
            urgency_level = "medium"
        
        return {
            "matched_symptoms": matched_symptoms,
            "potential_conditions": potential_conditions,
            "urgency_level": urgency_level,
            "recommendation": self._get_recommendation(urgency_level)
        }
    
    def _get_recommendation(self, urgency_level: str) -> str:
        """Get recommendation based on urgency level"""
        recommendations = {
            "critical": "Seek emergency medical attention immediately. Call 911.",
            "high": "Contact your healthcare provider immediately or go to urgent care.",
            "medium": "Schedule an appointment with your healthcare provider within 24-48 hours.",
            "low": "Monitor symptoms and contact healthcare provider if they worsen."
        }
        return recommendations.get(urgency_level, recommendations["low"])

class DrugInteractionChecker:
    """Drug interaction and medication safety checker"""
    
    def __init__(self):
        self.known_interactions = {
            ("aspirin", "warfarin"): {
                "severity": "high",
                "description": "Increased risk of bleeding",
                "recommendation": "Monitor INR closely, consider dose adjustment"
            },
            ("metformin", "alcohol"): {
                "severity": "medium",
                "description": "Increased risk of lactic acidosis",
                "recommendation": "Avoid excessive alcohol consumption"
            }
        }
    
    def check_interactions(self, medications: List[str]) -> Dict:
        """Check for drug interactions"""
        interactions = []
        
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                key1 = (med1.lower(), med2.lower())
                key2 = (med2.lower(), med1.lower())
                
                if key1 in self.known_interactions:
                    interactions.append({
                        "drugs": [med1, med2],
                        **self.known_interactions[key1]
                    })
                elif key2 in self.known_interactions:
                    interactions.append({
                        "drugs": [med1, med2],
                        **self.known_interactions[key2]
                    })
        
        return {
            "interactions_found": len(interactions) > 0,
            "interactions": interactions,
            "recommendation": "Always consult with your pharmacist or healthcare provider about potential drug interactions."
        }

class EmergencyDetector:
    """Emergency medical situation detector"""
    
    def __init__(self):
        self.emergency_keywords = {
            "heart_attack": ["heart attack", "chest pain", "chest pressure", "left arm pain"],
            "stroke": ["stroke", "face drooping", "arm weakness", "speech problems", "FAST"],
            "allergic_reaction": ["allergic reaction", "anaphylaxis", "swelling", "difficulty breathing"],
            "severe_bleeding": ["severe bleeding", "hemorrhage", "blood loss", "bleeding heavily"],
            "difficulty_breathing": ["can't breathe", "difficulty breathing", "shortness of breath", "choking"],
            "unconscious": ["unconscious", "passed out", "unresponsive", "not breathing"],
            "severe_pain": ["severe pain", "excruciating pain", "unbearable pain", "pain 10/10"]
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query for emergency indicators"""
        query_lower = query.lower()
        detected_emergencies = []
        
        for emergency_type, keywords in self.emergency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_emergencies.append(emergency_type)
        
        is_emergency = len(detected_emergencies) > 0
        
        return {
            "is_emergency": is_emergency,
            "detected_emergencies": detected_emergencies,
            "emergency_message": "🚨 This appears to be a medical emergency. Call 911 immediately or go to the nearest emergency room." if is_emergency else None,
            "urgency_level": "critical" if is_emergency else "normal"
        }

class MedicalCalculator:
    """Medical calculations and health metrics"""
    
    def calculate(self, calculation_type: str, values: Dict) -> Dict:
        """Perform medical calculations"""
        if calculation_type == "bmi":
            return self._calculate_bmi(values)
        elif calculation_type == "blood_pressure":
            return self._interpret_blood_pressure(values)
        elif calculation_type == "heart_rate":
            return self._interpret_heart_rate(values)
        else:
            return {"error": "Unknown calculation type"}
    
    def _calculate_bmi(self, values: Dict) -> Dict:
        """Calculate BMI and provide interpretation"""
        try:
            height_m = values["height_cm"] / 100
            weight_kg = values["weight_kg"]
            bmi = weight_kg / (height_m ** 2)
            
            if bmi < 18.5:
                category = "Underweight"
            elif bmi < 25:
                category = "Normal weight"
            elif bmi < 30:
                category = "Overweight"
            else:
                category = "Obese"
            
            return {
                "bmi": round(bmi, 1),
                "category": category,
                "healthy_range": "18.5 - 24.9",
                "recommendation": self._get_bmi_recommendation(category)
            }
        except (KeyError, ZeroDivisionError, TypeError):
            return {"error": "Invalid input for BMI calculation"}
    
    def _interpret_blood_pressure(self, values: Dict) -> Dict:
        """Interpret blood pressure readings"""
        try:
            systolic = values["systolic"]
            diastolic = values["diastolic"]
            
            if systolic < 120 and diastolic < 80:
                category = "Normal"
            elif systolic < 130 and diastolic < 80:
                category = "Elevated"
            elif systolic < 140 or diastolic < 90:
                category = "High Blood Pressure Stage 1"
            elif systolic < 180 or diastolic < 120:
                category = "High Blood Pressure Stage 2"
            else:
                category = "Hypertensive Crisis"
            
            return {
                "systolic": systolic,
                "diastolic": diastolic,
                "category": category,
                "recommendation": self._get_bp_recommendation(category)
            }
        except (KeyError, TypeError):
            return {"error": "Invalid input for blood pressure interpretation"}
    
    def _interpret_heart_rate(self, values: Dict) -> Dict:
        """Interpret heart rate"""
        try:
            heart_rate = values["heart_rate"]
            age = values.get("age", 30)
            
            if heart_rate < 60:
                category = "Bradycardia (slow)"
            elif heart_rate <= 100:
                category = "Normal"
            else:
                category = "Tachycardia (fast)"
            
            return {
                "heart_rate": heart_rate,
                "category": category,
                "normal_range": "60-100 bpm",
                "recommendation": self._get_hr_recommendation(category)
            }
        except (KeyError, TypeError):
            return {"error": "Invalid input for heart rate interpretation"}
    
    def _get_bmi_recommendation(self, category: str) -> str:
        """Get BMI-based recommendations"""
        recommendations = {
            "Underweight": "Consider consulting a healthcare provider about healthy weight gain strategies.",
            "Normal weight": "Maintain your current weight through balanced diet and regular exercise.",
            "Overweight": "Consider lifestyle changes including diet and exercise to achieve healthy weight.",
            "Obese": "Consult with healthcare provider about weight management strategies."
        }
        return recommendations.get(category, "Consult healthcare provider for personalized advice.")
    
    def _get_bp_recommendation(self, category: str) -> str:
        """Get blood pressure recommendations"""
        recommendations = {
            "Normal": "Maintain healthy lifestyle habits.",
            "Elevated": "Focus on lifestyle changes to prevent hypertension.",
            "High Blood Pressure Stage 1": "Lifestyle changes and possible medication. Consult healthcare provider.",
            "High Blood Pressure Stage 2": "Lifestyle changes and medication usually required. See healthcare provider.",
            "Hypertensive Crisis": "Seek immediate medical attention. Call 911 if experiencing symptoms."
        }
        return recommendations.get(category, "Consult healthcare provider.")
    
    def _get_hr_recommendation(self, category: str) -> str:
        """Get heart rate recommendations"""
        recommendations = {
            "Bradycardia (slow)": "If experiencing symptoms, consult healthcare provider.",
            "Normal": "Heart rate is within normal range.",
            "Tachycardia (fast)": "If persistent or with symptoms, consult healthcare provider."
        }
        return recommendations.get(category, "Consult healthcare provider if concerned.")
    
    def get_symptoms(self):
        """Get available symptoms for assessment"""
        return list(self.symptom_checker.symptom_patterns.keys())
