"""
Medical Analytics Module

This module provides comprehensive analytics capabilities for the medical AI assistant,
including usage tracking, patient risk assessment, interaction analysis, and reporting.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class InteractionType(Enum):
    SYMPTOM_CHECK = "symptom_check"
    MEDICATION_QUERY = "medication_query"
    EMERGENCY = "emergency"
    GENERAL_HEALTH = "general_health"
    DRUG_INTERACTION = "drug_interaction"
    HEALTH_TIP = "health_tip"

@dataclass
class InteractionRecord:
    """Record of a single user interaction"""
    timestamp: datetime
    user_id: str
    interaction_type: InteractionType
    query: str
    response: str
    risk_level: RiskLevel
    symptoms: List[str]
    medications: List[str]
    emergency_detected: bool
    response_time_ms: float
    user_satisfaction: Optional[int] = None  # 1-5 rating
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['interaction_type'] = self.interaction_type.value
        data['risk_level'] = self.risk_level.value
        return data

@dataclass
class UserProfile:
    """User profile with aggregated analytics"""
    user_id: str
    total_interactions: int
    emergency_count: int
    high_risk_count: int
    common_symptoms: List[str]
    common_medications: List[str]
    avg_response_time_ms: float
    last_interaction: datetime
    risk_score: float
    
class MedicalAnalytics:
    """
    Comprehensive medical analytics system for tracking and analyzing
    user interactions with the medical AI assistant.
    """
    
    def __init__(self):
        self.interactions: List[InteractionRecord] = []
        self.user_profiles: Dict[str, UserProfile] = {}
        self.session_data: Dict[str, List[InteractionRecord]] = defaultdict(list)
        
        # Analytics thresholds
        self.HIGH_RISK_KEYWORDS = {
            'chest pain', 'difficulty breathing', 'severe headache', 'unconscious',
            'bleeding', 'stroke', 'heart attack', 'severe pain', 'poisoning',
            'allergic reaction', 'suicide', 'overdose', 'seizure'
        }
        
        self.MODERATE_RISK_KEYWORDS = {
            'fever', 'nausea', 'vomiting', 'dizziness', 'fatigue', 'weakness',
            'shortness of breath', 'irregular heartbeat', 'persistent cough'
        }
        
    async def log_interaction(self, 
                       interaction_type: str,
                       query: str,
                       response: str,
                       patient_id: str = None,
                       risk_level: str = "low",
                       response_time: float = 0.0,
                       metadata: dict = None):
        """
        Log a new user interaction and update analytics
        """
        try:
            # Simple logging for now
            logger.info(f"Interaction logged: {interaction_type} - {risk_level}")
            return None
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
            return None
    
    def _assess_risk_level(self, query: str, symptoms: List[str], emergency_detected: bool) -> RiskLevel:
        """
        Assess the risk level of a query based on content and context
        """
        if emergency_detected:
            return RiskLevel.CRITICAL
        
        query_lower = query.lower()
        symptoms_lower = [s.lower() for s in symptoms]
        
        # Check for high-risk keywords
        for keyword in self.HIGH_RISK_KEYWORDS:
            if keyword in query_lower or any(keyword in symptom for symptom in symptoms_lower):
                return RiskLevel.HIGH
        
        # Check for moderate-risk keywords
        for keyword in self.MODERATE_RISK_KEYWORDS:
            if keyword in query_lower or any(keyword in symptom for symptom in symptoms_lower):
                return RiskLevel.MODERATE
        
        return RiskLevel.LOW
    
    def _update_user_profile(self, user_id: str, interaction: InteractionRecord):
        """
        Update user profile with new interaction data
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                total_interactions=0,
                emergency_count=0,
                high_risk_count=0,
                common_symptoms=[],
                common_medications=[],
                avg_response_time_ms=0.0,
                last_interaction=interaction.timestamp,
                risk_score=0.0
            )
        
        profile = self.user_profiles[user_id]
        profile.total_interactions += 1
        profile.last_interaction = interaction.timestamp
        
        if interaction.emergency_detected:
            profile.emergency_count += 1
        
        if interaction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            profile.high_risk_count += 1
        
        # Update common symptoms and medications
        profile.common_symptoms.extend(interaction.symptoms)
        profile.common_medications.extend(interaction.medications)
        
        # Keep only top 10 most common
        symptom_counter = Counter(profile.common_symptoms)
        profile.common_symptoms = [item for item, count in symptom_counter.most_common(10)]
        
        medication_counter = Counter(profile.common_medications)
        profile.common_medications = [item for item, count in medication_counter.most_common(10)]
        
        # Update average response time
        total_response_time = (profile.avg_response_time_ms * (profile.total_interactions - 1) + 
                              interaction.response_time_ms)
        profile.avg_response_time_ms = total_response_time / profile.total_interactions
        
        # Calculate risk score (0-100)
        profile.risk_score = self._calculate_user_risk_score(profile)
    
    def _calculate_user_risk_score(self, profile: UserProfile) -> float:
        """
        Calculate a composite risk score for a user (0-100)
        """
        if profile.total_interactions == 0:
            return 0.0
        
        # Base score from emergency and high-risk interactions
        emergency_score = (profile.emergency_count / profile.total_interactions) * 40
        high_risk_score = (profile.high_risk_count / profile.total_interactions) * 30
        
        # Frequency score (more interactions = higher engagement but potentially more issues)
        frequency_score = min(profile.total_interactions / 10, 1.0) * 20
        
        # Recency score (recent interactions indicate ongoing issues)
        days_since_last = (datetime.now() - profile.last_interaction).days
        recency_score = max(0, 10 - days_since_last) * 1.0
        
        total_score = emergency_score + high_risk_score + frequency_score + recency_score
        return min(total_score, 100.0)
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a specific user
        """
        if user_id not in self.user_profiles:
            return {"error": "User not found"}
        
        profile = self.user_profiles[user_id]
        user_interactions = [i for i in self.interactions if i.user_id == user_id]
        
        # Calculate interaction type distribution
        interaction_types = Counter([i.interaction_type.value for i in user_interactions])
        
        # Calculate risk level distribution
        risk_levels = Counter([i.risk_level.value for i in user_interactions])
        
        # Calculate trends (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_interactions = [i for i in user_interactions if i.timestamp >= thirty_days_ago]
        
        return {
            "user_id": user_id,
            "profile": asdict(profile),
            "interaction_distribution": dict(interaction_types),
            "risk_distribution": dict(risk_levels),
            "recent_activity": len(recent_interactions),
            "total_interactions": len(user_interactions),
            "avg_response_time": profile.avg_response_time_ms,
            "last_interaction": profile.last_interaction.isoformat()
        }
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """
        Get system-wide analytics and statistics
        """
        if not self.interactions:
            return {"message": "No interactions recorded yet"}
        
        # Overall statistics
        total_interactions = len(self.interactions)
        total_users = len(self.user_profiles)
        
        # Risk level distribution
        risk_levels = Counter([i.risk_level.value for i in self.interactions])
        
        # Interaction type distribution
        interaction_types = Counter([i.interaction_type.value for i in self.interactions])
        
        # Emergency statistics
        emergency_count = sum(1 for i in self.interactions if i.emergency_detected)
        emergency_rate = (emergency_count / total_interactions) * 100
        
        # Response time statistics
        response_times = [i.response_time_ms for i in self.interactions]
        avg_response_time = sum(response_times) / len(response_times)
        
        # Most common symptoms and medications
        all_symptoms = []
        all_medications = []
        for interaction in self.interactions:
            all_symptoms.extend(interaction.symptoms)
            all_medications.extend(interaction.medications)
        
        common_symptoms = Counter(all_symptoms).most_common(10)
        common_medications = Counter(all_medications).most_common(10)
        
        # Daily interaction trends (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_interactions = [i for i in self.interactions if i.timestamp >= thirty_days_ago]
        
        daily_counts = defaultdict(int)
        for interaction in recent_interactions:
            date_key = interaction.timestamp.date().isoformat()
            daily_counts[date_key] += 1
        
        # High-risk users
        high_risk_users = [
            {
                "user_id": user_id,
                "risk_score": profile.risk_score,
                "emergency_count": profile.emergency_count,
                "high_risk_count": profile.high_risk_count
            }
            for user_id, profile in self.user_profiles.items()
            if profile.risk_score > 50
        ]
        
        return {
            "total_interactions": total_interactions,
            "total_users": total_users,
            "emergency_rate": round(emergency_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "risk_distribution": dict(risk_levels),
            "interaction_types": dict(interaction_types),
            "common_symptoms": common_symptoms,
            "common_medications": common_medications,
            "daily_trends": dict(daily_counts),
            "high_risk_users": sorted(high_risk_users, key=lambda x: x["risk_score"], reverse=True)
        }
    
    def get_health_insights(self) -> Dict[str, Any]:
        """
        Generate health insights based on aggregated data
        """
        if not self.interactions:
            return {"message": "Insufficient data for insights"}
        
        # Symptom patterns
        symptom_combinations = defaultdict(int)
        for interaction in self.interactions:
            if len(interaction.symptoms) > 1:
                symptoms_sorted = sorted(interaction.symptoms)
                combination = " + ".join(symptoms_sorted)
                symptom_combinations[combination] += 1
        
        # Medication patterns
        medication_queries = [i for i in self.interactions if i.interaction_type == InteractionType.MEDICATION_QUERY]
        medication_trends = Counter([med for i in medication_queries for med in i.medications])
        
        # Emergency patterns
        emergency_interactions = [i for i in self.interactions if i.emergency_detected]
        emergency_symptoms = Counter([symptom for i in emergency_interactions for symptom in i.symptoms])
        
        # Peak usage times
        hour_distribution = Counter([i.timestamp.hour for i in self.interactions])
        peak_hours = hour_distribution.most_common(5)
        
        return {
            "common_symptom_combinations": dict(symptom_combinations),
            "medication_trends": dict(medication_trends.most_common(10)),
            "emergency_symptoms": dict(emergency_symptoms.most_common(10)),
            "peak_usage_hours": peak_hours,
            "total_emergencies": len(emergency_interactions),
            "insights": self._generate_health_insights()
        }
    
    def _generate_health_insights(self) -> List[str]:
        """
        Generate actionable health insights from the data
        """
        insights = []
        
        # Emergency rate insight
        emergency_rate = (sum(1 for i in self.interactions if i.emergency_detected) / 
                         len(self.interactions)) * 100
        
        if emergency_rate > 10:
            insights.append(f"High emergency rate detected ({emergency_rate:.1f}%). Consider implementing more proactive health monitoring.")
        
        # Common symptom insight
        all_symptoms = [symptom for i in self.interactions for symptom in i.symptoms]
        if all_symptoms:
            most_common = Counter(all_symptoms).most_common(1)[0]
            insights.append(f"Most reported symptom: '{most_common[0]}' ({most_common[1]} reports). Consider adding specialized content.")
        
        # Response time insight
        avg_response_time = sum(i.response_time_ms for i in self.interactions) / len(self.interactions)
        if avg_response_time > 2000:  # 2 seconds
            insights.append(f"Average response time is {avg_response_time:.0f}ms. Consider optimizing for faster responses.")
        
        # Risk distribution insight
        high_risk_count = sum(1 for i in self.interactions if i.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        high_risk_rate = (high_risk_count / len(self.interactions)) * 100
        
        if high_risk_rate > 20:
            insights.append(f"High percentage of high-risk interactions ({high_risk_rate:.1f}%). Enhanced triage may be needed.")
        
        return insights
    
    def export_analytics(self, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Export analytics data for a specific date range
        """
        # Filter interactions by date range
        filtered_interactions = self.interactions
        
        if start_date:
            filtered_interactions = [i for i in filtered_interactions if i.timestamp >= start_date]
        
        if end_date:
            filtered_interactions = [i for i in filtered_interactions if i.timestamp <= end_date]
        
        return {
            "export_date": datetime.now().isoformat(),
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "interactions": [interaction.to_dict() for interaction in filtered_interactions],
            "summary": {
                "total_interactions": len(filtered_interactions),
                "unique_users": len(set(i.user_id for i in filtered_interactions)),
                "emergency_count": sum(1 for i in filtered_interactions if i.emergency_detected),
                "risk_distribution": dict(Counter([i.risk_level.value for i in filtered_interactions]))
            }
        }

    async def get_dashboard_data(self):
        """Get dashboard data"""
        return {"message": "Dashboard data not yet implemented"}
    
    async def get_patient_analytics(self, patient_id: str):
        """Get patient analytics"""
        return {"message": f"Patient analytics for {patient_id} not yet implemented"}

# Global analytics instance
analytics = MedicalAnalytics()
