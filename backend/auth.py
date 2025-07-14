"""
Authentication utilities for the Medical AI Assistant.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .database import get_db
from .models import User
from .schemas import TokenData
import logging

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token authentication
security = HTTPBearer()

# Set up logger
logger = logging.getLogger(__name__)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db: Session, email: str, password: str):
    """Authenticate user with email and password."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.warning(f"Authentication failed: user {email} not found.")
        return False
    if not verify_password(password, user.password_hash):
        logger.warning(f"Authentication failed: invalid password for user {email}.")
        return False
    logger.info(f"Authentication successful for user {email}.")
    return user

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            logger.warning("JWT token missing subject (email)")
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        logger.warning("JWT decode error")
        raise credentials_exception
    
    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        logger.warning(f"User {token_data.email} not found for current session.")
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get current active user. Checks if user is active."""
    # Add any additional checks here (e.g., user.is_active)
    if not getattr(current_user, "is_active", True):
        logger.warning(f"Inactive user {current_user.email} tried to access protected resource.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user. Please contact support."
        )
    return current_user

def check_subscription_access(required_subscription: str):
    """Decorator to check if user has required subscription level."""
    def decorator(current_user: User = Depends(get_current_active_user)):
        subscription_hierarchy = {
            "free": 0,
            "premium": 1,
            "pro": 2
        }
        
        user_level = subscription_hierarchy.get(current_user.subscription_type.value, 0)
        required_level = subscription_hierarchy.get(required_subscription, 0)
        
        if user_level < required_level:
            logger.warning(f"User {current_user.email} with subscription {current_user.subscription_type.value} tried to access {required_subscription} feature.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {required_subscription} subscription"
            )
        return current_user
    return decorator
