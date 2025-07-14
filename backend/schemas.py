"""
Pydantic schemas for data validation in the Medical AI Assistant.
"""

from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class SubscriptionType(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"

# User Schemas
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    subscription_type: Optional[SubscriptionType] = None

class UserResponse(UserBase):
    id: int
    subscription_type: SubscriptionType
    created_at: datetime
    
    class Config:
        from_attributes = True

# Interaction Schemas
class InteractionBase(BaseModel):
    interaction_type: str
    content: str

class InteractionCreate(InteractionBase):
    pass

class InteractionResponse(InteractionBase):
    id: int
    user_id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

# Subscription Schemas
class SubscriptionBase(BaseModel):
    plan: str

class SubscriptionCreate(SubscriptionBase):
    pass

class SubscriptionUpdate(BaseModel):
    plan: Optional[str] = None
    end_date: Optional[datetime] = None

class SubscriptionResponse(SubscriptionBase):
    id: int
    user_id: int
    start_date: datetime
    end_date: Optional[datetime]
    
    class Config:
        from_attributes = True

# Authentication Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# Dashboard Schemas
class UserStats(BaseModel):
    total_interactions: int
    interactions_this_month: int
    subscription_type: SubscriptionType
    subscription_expires: Optional[datetime]

class DashboardResponse(BaseModel):
    user: UserResponse
    stats: UserStats
    recent_interactions: List[InteractionResponse]
    
    class Config:
        from_attributes = True
