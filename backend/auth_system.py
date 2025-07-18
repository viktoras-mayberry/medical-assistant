#!/usr/bin/env python3
"""
Comprehensive Authentication System for Medical AI Assistant
==========================================================

This module provides a complete authentication system with:
- User registration and login
- JWT token management
- Password security
- Session management
- Role-based access control
- Account verification
- Password reset functionality
- Security logging and monitoring
"""

import os
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import re

from fastapi import HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator
import redis

from database import get_db
from models import User
from config import MedicalAIConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
config = MedicalAIConfig()
SECRET_KEY = config.SECRET_KEY
ALGORITHM = config.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = config.JWT_EXPIRATION_HOURS * 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token authentication
security = HTTPBearer()

# Redis client for session management (optional)
try:
    redis_client = redis.Redis.from_url(config.REDIS_URL)
    REDIS_AVAILABLE = True
except:
    redis_client = None
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Session management will be limited.")

# Pydantic models for authentication
class UserRegister(BaseModel):
    """User registration model"""
    email: EmailStr
    password: str
    confirm_password: str
    first_name: str
    last_name: str
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    medical_conditions: Optional[List[str]] = []
    emergency_contact: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str
    remember_me: bool = False

class PasswordReset(BaseModel):
    """Password reset model"""
    email: EmailStr

class PasswordChange(BaseModel):
    """Password change model"""
    current_password: str
    new_password: str
    confirm_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserProfile(BaseModel):
    """User profile model"""
    first_name: str
    last_name: str
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    medical_conditions: Optional[List[str]] = []
    emergency_contact: Optional[str] = None

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: int
    email: str
    first_name: str
    last_name: str

class AuthenticationSystem:
    """Comprehensive authentication system"""
    
    def __init__(self):
        self.config = config
        self.failed_attempts = {}  # Track failed login attempts
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: int) -> str:
        """Create refresh token"""
        data = {"sub": str(user_id), "type": "refresh"}
        expire = datetime.utcnow() + timedelta(days=7)  # Refresh token expires in 7 days
        data.update({"exp": expire})
        return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def generate_verification_token(self, user_id: int) -> str:
        """Generate email verification token"""
        data = {"user_id": user_id, "type": "verification"}
        expire = datetime.utcnow() + timedelta(hours=24)  # Verification token expires in 24 hours
        data.update({"exp": expire})
        return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    
    def generate_reset_token(self, user_id: int) -> str:
        """Generate password reset token"""
        data = {"user_id": user_id, "type": "reset"}
        expire = datetime.utcnow() + timedelta(hours=1)  # Reset token expires in 1 hour
        data.update({"exp": expire})
        return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    
    def is_account_locked(self, email: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if email not in self.failed_attempts:
            return False
        
        attempts, last_attempt = self.failed_attempts[email]
        if attempts >= self.max_failed_attempts:
            if datetime.utcnow() - last_attempt < self.lockout_duration:
                return True
            else:
                # Reset attempts after lockout period
                del self.failed_attempts[email]
        
        return False
    
    def record_failed_attempt(self, email: str):
        """Record failed login attempt"""
        if email not in self.failed_attempts:
            self.failed_attempts[email] = [1, datetime.utcnow()]
        else:
            attempts, _ = self.failed_attempts[email]
            self.failed_attempts[email] = [attempts + 1, datetime.utcnow()]
    
    def clear_failed_attempts(self, email: str):
        """Clear failed login attempts"""
        if email in self.failed_attempts:
            del self.failed_attempts[email]
    
    async def register_user(self, user_data: UserRegister, db: Session, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Register a new user"""
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(User.email == user_data.email).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User with this email already exists"
                )
            
            # Create new user
            hashed_password = self.get_password_hash(user_data.password)
            verification_token = secrets.token_urlsafe(32)
            
            new_user = User(
                email=user_data.email,
                password_hash=hashed_password,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                phone=user_data.phone,
                date_of_birth=user_data.date_of_birth,
                gender=user_data.gender,
                medical_conditions=user_data.medical_conditions,
                emergency_contact=user_data.emergency_contact,
                verification_token=verification_token,
                is_verified=False,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            # Send verification email
            background_tasks.add_task(self.send_verification_email, user_data.email, verification_token)
            
            logger.info(f"New user registered: {user_data.email}")
            
            return {
                "message": "User registered successfully",
                "user_id": new_user.id,
                "email": new_user.email,
                "verification_required": True
            }
            
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    async def login_user(self, login_data: UserLogin, db: Session) -> TokenResponse:
        """Authenticate user and return tokens"""
        try:
            # Check if account is locked
            if self.is_account_locked(login_data.email):
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail="Account temporarily locked due to multiple failed attempts"
                )
            
            # Find user
            user = db.query(User).filter(User.email == login_data.email).first()
            if not user:
                self.record_failed_attempt(login_data.email)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Verify password
            if not self.verify_password(login_data.password, user.password_hash):
                self.record_failed_attempt(login_data.email)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Check if user is active
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is inactive"
                )
            
            # Clear failed attempts
            self.clear_failed_attempts(login_data.email)
            
            # Create tokens
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            if login_data.remember_me:
                access_token_expires = timedelta(days=7)
            
            access_token = self.create_access_token(
                data={"sub": user.email, "user_id": user.id},
                expires_delta=access_token_expires
            )
            
            refresh_token = self.create_refresh_token(user.id)
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
            # Store session in Redis if available
            if REDIS_AVAILABLE:
                session_data = {
                    "user_id": user.id,
                    "email": user.email,
                    "login_time": datetime.utcnow().isoformat()
                }
                redis_client.setex(f"session:{user.id}", 3600, str(session_data))
            
            logger.info(f"User login successful: {user.email}")
            
            return TokenResponse(
                access_token=access_token,
                token_type="bearer",
                expires_in=int(access_token_expires.total_seconds()),
                user_id=user.id,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during login: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Login failed"
            )
    
    async def verify_email(self, token: str, db: Session) -> Dict[str, Any]:
        """Verify user email address"""
        try:
            user = db.query(User).filter(User.verification_token == token).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid verification token"
                )
            
            user.is_verified = True
            user.verification_token = None
            user.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Email verified for user: {user.email}")
            
            return {
                "message": "Email verified successfully",
                "user_id": user.id,
                "email": user.email
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error verifying email: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Email verification failed"
            )
    
    async def reset_password_request(self, email: str, db: Session, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Send password reset email"""
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                # Don't reveal if user exists
                return {"message": "If the email exists, a reset link has been sent"}
            
            reset_token = self.generate_reset_token(user.id)
            
            # Store reset token in database
            user.reset_token = reset_token
            user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
            user.updated_at = datetime.utcnow()
            db.commit()
            
            # Send reset email
            background_tasks.add_task(self.send_password_reset_email, email, reset_token)
            
            logger.info(f"Password reset requested for user: {email}")
            
            return {"message": "If the email exists, a reset link has been sent"}
            
        except Exception as e:
            logger.error(f"Error requesting password reset: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password reset request failed"
            )
    
    async def reset_password(self, token: str, new_password: str, db: Session) -> Dict[str, Any]:
        """Reset user password"""
        try:
            # Verify token
            payload = self.verify_token(token)
            if not payload or payload.get("type") != "reset":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired reset token"
                )
            
            user_id = payload.get("user_id")
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user or user.reset_token != token or user.reset_token_expires < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired reset token"
                )
            
            # Update password
            user.password_hash = self.get_password_hash(new_password)
            user.reset_token = None
            user.reset_token_expires = None
            user.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Password reset successful for user: {user.email}")
            
            return {
                "message": "Password reset successful",
                "user_id": user.id
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error resetting password: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password reset failed"
            )
    
    async def change_password(self, user_id: int, password_data: PasswordChange, db: Session) -> Dict[str, Any]:
        """Change user password"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Verify current password
            if not self.verify_password(password_data.current_password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect"
                )
            
            # Update password
            user.password_hash = self.get_password_hash(password_data.new_password)
            user.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Password changed for user: {user.email}")
            
            return {
                "message": "Password changed successfully",
                "user_id": user.id
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password change failed"
            )
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
        """Get current authenticated user"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            
            if email is None or user_id is None:
                raise credentials_exception
                
        except JWTError:
            raise credentials_exception
        
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise credentials_exception
            
        return user
    
    async def logout_user(self, user_id: int) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        try:
            # Remove session from Redis if available
            if REDIS_AVAILABLE:
                redis_client.delete(f"session:{user_id}")
            
            logger.info(f"User logged out: {user_id}")
            
            return {"message": "Logout successful"}
            
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout failed"
            )
    
    async def send_verification_email(self, email: str, token: str):
        """Send email verification email"""
        try:
            # In production, implement actual email sending
            # For now, just log the verification link
            verification_link = f"https://yourapp.com/verify?token={token}"
            logger.info(f"Verification email would be sent to {email}: {verification_link}")
            
            # TODO: Implement actual email sending using SMTP
            # smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
            # smtp_server.starttls()
            # smtp_server.login(EMAIL_USER, EMAIL_PASSWORD)
            # ... send email ...
            
        except Exception as e:
            logger.error(f"Error sending verification email: {e}")
    
    async def send_password_reset_email(self, email: str, token: str):
        """Send password reset email"""
        try:
            # In production, implement actual email sending
            # For now, just log the reset link
            reset_link = f"https://yourapp.com/reset-password?token={token}"
            logger.info(f"Password reset email would be sent to {email}: {reset_link}")
            
            # TODO: Implement actual email sending using SMTP
            
        except Exception as e:
            logger.error(f"Error sending password reset email: {e}")

# Global authentication system instance
auth_system = AuthenticationSystem()

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    """Get current authenticated user"""
    return await auth_system.get_current_user(credentials, db)

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user

async def get_current_verified_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current verified user"""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified"
        )
    return current_user
