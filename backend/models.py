"""
Database models for the Medical AI Assistant.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum as PyEnum
from .database import Base

# Enum for subscription types
class SubscriptionType(PyEnum):
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    subscription_type = Column(Enum(SubscriptionType), default=SubscriptionType.FREE)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    interactions = relationship("Interaction", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")

class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    interaction_type = Column(String, nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="interactions")

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime)
    plan = Column(String, nullable=False)

    # Relationship
    user = relationship("User", back_populates="subscriptions")

