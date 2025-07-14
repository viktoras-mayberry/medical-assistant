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
    subscription_type = Column(Enum(SubscriptionType), default=SubscriptionType.FREE, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Integer, default=1, nullable=False)  # 1=True, 0=False for SQLite compatibility
    
    # Relationships
    interactions = relationship("Interaction", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', subscription_type='{self.subscription_type.value}', is_active={self.is_active})>"

class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    interaction_type = Column(String, nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    user = relationship("User", back_populates="interactions")

    def __repr__(self):
        return f"<Interaction(id={self.id}, user_id={self.user_id}, type='{self.interaction_type}', timestamp={self.timestamp})>"

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    start_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_date = Column(DateTime, nullable=True)
    plan = Column(String, nullable=False)

    # Relationship
    user = relationship("User", back_populates="subscriptions")

    def __repr__(self):
        return f"<Subscription(id={self.id}, user_id={self.user_id}, plan='{self.plan}', start_date={self.start_date}, end_date={self.end_date})>"

