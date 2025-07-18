#!/usr/bin/env python3
"""
Authentication Endpoints for Medical AI Assistant
================================================

This module provides REST API endpoints for user authentication including:
- User registration with email verification
- User login with JWT tokens
- Password reset functionality
- User profile management
- Account settings
- Security features
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime
import logging

from database import get_db
from models import User
from auth_system import (
    AuthenticationSystem,
    UserRegister,
    UserLogin,
    PasswordReset,
    PasswordChange,
    UserProfile,
    TokenResponse,
    get_current_user,
    get_current_active_user,
    get_current_verified_user
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Initialize authentication system
auth_system = AuthenticationSystem()

@router.post("/register", response_model=Dict[str, Any])
async def register(
    user_data: UserRegister,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Register a new user account
    
    - **email**: Valid email address
    - **password**: Strong password (min 8 chars, uppercase, lowercase, digit, special char)
    - **confirm_password**: Must match password
    - **first_name**: User's first name
    - **last_name**: User's last name
    - **phone**: Optional phone number
    - **date_of_birth**: Optional date of birth
    - **gender**: Optional gender
    - **medical_conditions**: Optional list of medical conditions
    - **emergency_contact**: Optional emergency contact
    """
    try:
        result = await auth_system.register_user(user_data, db, background_tasks)
        
        return {
            "success": True,
            "message": "Registration successful. Please check your email to verify your account.",
            "data": result
        }
        
    except HTTPException as e:
        logger.error(f"Registration failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed due to server error"
        )

@router.post("/login", response_model=Dict[str, Any])
async def login(
    login_data: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access token
    
    - **email**: User's email address
    - **password**: User's password
    - **remember_me**: Whether to extend token expiration
    """
    try:
        token_response = await auth_system.login_user(login_data, db)
        
        return {
            "success": True,
            "message": "Login successful",
            "data": token_response.dict()
        }
        
    except HTTPException as e:
        logger.error(f"Login failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to server error"
        )

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user)
):
    """
    Logout user and invalidate session
    """
    try:
        result = await auth_system.logout_user(current_user.id)
        
        return {
            "success": True,
            "message": "Logout successful"
        }
        
    except HTTPException as e:
        logger.error(f"Logout failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed due to server error"
        )

@router.post("/verify-email")
async def verify_email(
    token: str,
    db: Session = Depends(get_db)
):
    """
    Verify user email address with token
    
    - **token**: Email verification token
    """
    try:
        result = await auth_system.verify_email(token, db)
        
        return {
            "success": True,
            "message": "Email verified successfully",
            "data": result
        }
        
    except HTTPException as e:
        logger.error(f"Email verification failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during email verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification failed due to server error"
        )

@router.post("/forgot-password")
async def forgot_password(
    email: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Request password reset link
    
    - **email**: User's email address
    """
    try:
        result = await auth_system.reset_password_request(email, db, background_tasks)
        
        return {
            "success": True,
            "message": "If the email exists, a reset link has been sent",
            "data": result
        }
        
    except HTTPException as e:
        logger.error(f"Password reset request failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during password reset request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed due to server error"
        )

@router.post("/reset-password")
async def reset_password(
    token: str,
    new_password: str,
    db: Session = Depends(get_db)
):
    """
    Reset password using reset token
    
    - **token**: Password reset token
    - **new_password**: New password
    """
    try:
        result = await auth_system.reset_password(token, new_password, db)
        
        return {
            "success": True,
            "message": "Password reset successful",
            "data": result
        }
        
    except HTTPException as e:
        logger.error(f"Password reset failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during password reset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed due to server error"
        )

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change user password
    
    - **current_password**: Current password
    - **new_password**: New password
    - **confirm_password**: Confirm new password
    """
    try:
        result = await auth_system.change_password(current_user.id, password_data, db)
        
        return {
            "success": True,
            "message": "Password changed successfully",
            "data": result
        }
        
    except HTTPException as e:
        logger.error(f"Password change failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during password change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed due to server error"
        )

@router.get("/profile")
async def get_profile(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user profile
    """
    try:
        return {
            "success": True,
            "message": "Profile retrieved successfully",
            "data": {
                "id": current_user.id,
                "email": current_user.email,
                "first_name": current_user.first_name,
                "last_name": current_user.last_name,
                "phone": current_user.phone,
                "date_of_birth": current_user.date_of_birth,
                "gender": current_user.gender,
                "medical_conditions": current_user.medical_conditions,
                "emergency_contact": current_user.emergency_contact,
                "is_verified": current_user.is_verified,
                "is_active": current_user.is_active,
                "created_at": current_user.created_at,
                "last_login": current_user.last_login
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )

@router.put("/profile")
async def update_profile(
    profile_data: UserProfile,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update user profile
    
    - **first_name**: User's first name
    - **last_name**: User's last name
    - **phone**: Phone number
    - **date_of_birth**: Date of birth
    - **gender**: Gender
    - **medical_conditions**: List of medical conditions
    - **emergency_contact**: Emergency contact
    """
    try:
        # Update user profile
        current_user.first_name = profile_data.first_name
        current_user.last_name = profile_data.last_name
        current_user.phone = profile_data.phone
        current_user.date_of_birth = profile_data.date_of_birth
        current_user.gender = profile_data.gender
        current_user.medical_conditions = profile_data.medical_conditions
        current_user.emergency_contact = profile_data.emergency_contact
        
        db.commit()
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "data": {
                "id": current_user.id,
                "email": current_user.email,
                "first_name": current_user.first_name,
                "last_name": current_user.last_name,
                "phone": current_user.phone,
                "date_of_birth": current_user.date_of_birth,
                "gender": current_user.gender,
                "medical_conditions": current_user.medical_conditions,
                "emergency_contact": current_user.emergency_contact
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information
    """
    try:
        return {
            "success": True,
            "message": "User information retrieved successfully",
            "data": {
                "id": current_user.id,
                "email": current_user.email,
                "first_name": current_user.first_name,
                "last_name": current_user.last_name,
                "is_verified": current_user.is_verified,
                "is_active": current_user.is_active,
                "created_at": current_user.created_at,
                "last_login": current_user.last_login
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )

@router.delete("/account")
async def delete_account(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete user account (soft delete)
    """
    try:
        # Soft delete - deactivate account instead of deleting
        current_user.is_active = False
        current_user.updated_at = datetime.utcnow()
        db.commit()
        
        # Logout user
        await auth_system.logout_user(current_user.id)
        
        return {
            "success": True,
            "message": "Account deactivated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting account: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )

@router.post("/resend-verification")
async def resend_verification(
    email: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Resend email verification
    
    - **email**: User's email address
    """
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Don't reveal if user exists
            return {
                "success": True,
                "message": "If the email exists and is not verified, a verification link has been sent"
            }
        
        if user.is_verified:
            return {
                "success": True,
                "message": "Email is already verified"
            }
        
        # Generate new verification token
        import secrets
        verification_token = secrets.token_urlsafe(32)
        user.verification_token = verification_token
        db.commit()
        
        # Send verification email
        background_tasks.add_task(auth_system.send_verification_email, email, verification_token)
        
        return {
            "success": True,
            "message": "If the email exists and is not verified, a verification link has been sent"
        }
        
    except Exception as e:
        logger.error(f"Error resending verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resend verification email"
        )

@router.get("/check-auth")
async def check_authentication(
    current_user: User = Depends(get_current_active_user)
):
    """
    Check if user is authenticated
    """
    return {
        "success": True,
        "message": "User is authenticated",
        "data": {
            "authenticated": True,
            "user_id": current_user.id,
            "email": current_user.email,
            "is_verified": current_user.is_verified
        }
    }

@router.get("/sessions")
async def get_user_sessions(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user's active sessions (if Redis is available)
    """
    try:
        from auth_system import REDIS_AVAILABLE, redis_client
        
        if not REDIS_AVAILABLE:
            return {
                "success": True,
                "message": "Session management not available",
                "data": {"sessions": []}
            }
        
        # Get session data from Redis
        session_key = f"session:{current_user.id}"
        session_data = redis_client.get(session_key)
        
        if session_data:
            sessions = [{"session_id": session_key, "data": str(session_data)}]
        else:
            sessions = []
        
        return {
            "success": True,
            "message": "Sessions retrieved successfully",
            "data": {"sessions": sessions}
        }
        
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )

@router.post("/validate-token")
async def validate_token(
    current_user: User = Depends(get_current_user)
):
    """
    Validate JWT token
    """
    return {
        "success": True,
        "message": "Token is valid",
        "data": {
            "valid": True,
            "user_id": current_user.id,
            "email": current_user.email
        }
    }

# Health check endpoint for authentication system
@router.get("/health")
async def auth_health_check():
    """
    Health check for authentication system
    """
    return {
        "success": True,
        "message": "Authentication system is healthy",
        "data": {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    }
