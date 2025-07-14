"""
Payment processing system for Medical AI Assistant using Stripe.
"""

import os
import stripe
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from .models import User, Subscription, SubscriptionType
from .database import get_db

logger = logging.getLogger(__name__)

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

class PaymentProcessor:
    """Handles payment processing and subscription management."""
    
    def __init__(self):
        self.stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
        self.stripe_publishable_key = os.getenv("STRIPE_PUBLISHABLE_KEY")
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        if not self.stripe_secret_key:
            logger.warning("Stripe secret key not configured. Payment processing will be disabled.")
        
        # Subscription plans configuration
        self.subscription_plans = {
            "free": {
                "name": "Free Plan",
                "price": 0,
                "features": [
                    "5 voice interactions per day",
                    "Basic medical information",
                    "Limited chat history"
                ],
                "limits": {
                    "daily_interactions": 5,
                    "monthly_interactions": 150,
                    "chat_history_days": 7
                }
            },
            "premium": {
                "name": "Premium Plan",
                "price": 9.99,
                "stripe_price_id": "price_premium_monthly",  # Replace with actual Stripe price ID
                "features": [
                    "50 voice interactions per day",
                    "Advanced medical insights",
                    "30-day chat history",
                    "Priority support"
                ],
                "limits": {
                    "daily_interactions": 50,
                    "monthly_interactions": 1500,
                    "chat_history_days": 30
                }
            },
            "pro": {
                "name": "Pro Plan",
                "price": 19.99,
                "stripe_price_id": "price_pro_monthly",  # Replace with actual Stripe price ID
                "features": [
                    "Unlimited voice interactions",
                    "Advanced medical analytics",
                    "Unlimited chat history",
                    "24/7 priority support",
                    "API access"
                ],
                "limits": {
                    "daily_interactions": -1,  # Unlimited
                    "monthly_interactions": -1,  # Unlimited
                    "chat_history_days": -1  # Unlimited
                }
            }
        }
    
    def create_customer(self, user_email: str, user_name: str = None) -> str:
        """Create a Stripe customer."""
        try:
            customer = stripe.Customer.create(
                email=user_email,
                name=user_name,
                metadata={
                    "user_email": user_email
                }
            )
            return customer.id
        except stripe.error.StripeError as e:
            logger.error(f"Error creating Stripe customer: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create customer"
            )
    
    def create_checkout_session(
        self, 
        user_email: str, 
        plan_type: str, 
        success_url: str, 
        cancel_url: str
    ) -> Dict:
        """Create a Stripe checkout session for subscription."""
        try:
            if plan_type not in self.subscription_plans:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid subscription plan"
                )
            
            plan = self.subscription_plans[plan_type]
            
            if plan_type == "free":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Free plan doesn't require payment"
                )
            
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': plan["stripe_price_id"],
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                customer_email=user_email,
                metadata={
                    "plan_type": plan_type,
                    "user_email": user_email
                }
            )
            
            return {
                "checkout_url": session.url,
                "session_id": session.id
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error creating checkout session: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create checkout session"
            )
    
    def create_customer_portal_session(self, customer_id: str, return_url: str) -> Dict:
        """Create a Stripe customer portal session."""
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            
            return {
                "portal_url": session.url
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error creating customer portal session: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create customer portal session"
            )
    
    def handle_subscription_created(self, subscription_data: Dict, db: Session):
        """Handle successful subscription creation."""
        try:
            customer_email = subscription_data["customer_email"]
            plan_type = subscription_data["metadata"].get("plan_type")
            
            # Find user by email
            user = db.query(User).filter(User.email == customer_email).first()
            if not user:
                logger.error(f"User not found for email: {customer_email}")
                return
            
            # Update user subscription type
            user.subscription_type = SubscriptionType(plan_type)
            
            # Create subscription record
            subscription = Subscription(
                user_id=user.id,
                plan=plan_type,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=30)  # Monthly subscription
            )
            
            db.add(subscription)
            db.commit()
            
            logger.info(f"Subscription created for user {user.email}: {plan_type}")
            
        except Exception as e:
            logger.error(f"Error handling subscription creation: {e}")
            db.rollback()
    
    def handle_subscription_updated(self, subscription_data: Dict, db: Session):
        """Handle subscription updates."""
        try:
            # Handle subscription changes (upgrades/downgrades)
            customer_email = subscription_data["customer_email"]
            
            user = db.query(User).filter(User.email == customer_email).first()
            if not user:
                logger.error(f"User not found for email: {customer_email}")
                return
            
            # Update subscription based on new plan
            # This would involve checking the new plan and updating accordingly
            logger.info(f"Subscription updated for user {user.email}")
            
        except Exception as e:
            logger.error(f"Error handling subscription update: {e}")
    
    def handle_subscription_cancelled(self, subscription_data: Dict, db: Session):
        """Handle subscription cancellation."""
        try:
            customer_email = subscription_data["customer_email"]
            
            user = db.query(User).filter(User.email == customer_email).first()
            if not user:
                logger.error(f"User not found for email: {customer_email}")
                return
            
            # Downgrade user to free plan
            user.subscription_type = SubscriptionType.FREE
            
            # Update subscription record
            subscription = db.query(Subscription).filter(
                Subscription.user_id == user.id
            ).order_by(Subscription.start_date.desc()).first()
            
            if subscription:
                subscription.end_date = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"Subscription cancelled for user {user.email}")
            
        except Exception as e:
            logger.error(f"Error handling subscription cancellation: {e}")
            db.rollback()
    
    def get_subscription_plans(self) -> Dict:
        """Get available subscription plans."""
        return self.subscription_plans
    
    def check_usage_limits(self, user: User, interaction_type: str) -> bool:
        """Check if user has exceeded their usage limits."""
        plan = self.subscription_plans.get(user.subscription_type.value)
        if not plan:
            return False
        
        limits = plan["limits"]
        
        # For now, return True (allow usage)
        # In a real implementation, you'd check actual usage against limits
        return True
    
    def get_user_usage_stats(self, user_id: int, db: Session) -> Dict:
        """Get user usage statistics."""
        # This would query the interactions table to get usage stats
        # For now, return mock data
        return {
            "daily_interactions": 5,
            "monthly_interactions": 150,
            "remaining_daily": 10,
            "remaining_monthly": 300
        }

# Global payment processor instance
payment_processor = PaymentProcessor()
