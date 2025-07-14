#!/usr/bin/env python3
"""
Database initialization script for Medical AI Assistant.
This script sets up the database tables and creates initial data.
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add the backend directory to the Python path
backend_path = Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    from database import init_db, engine, Base
    from models import User, Interaction, Subscription, SubscriptionType
    from auth import get_password_hash
    from sqlalchemy.orm import sessionmaker
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def create_sample_data():
    """Create sample data for testing."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Check if sample user already exists
        existing_user = session.query(User).filter(User.email == "admin@medicalai.com").first()
        if existing_user:
            logger.info("Sample data already exists")
            return
        # Create a sample admin user
        admin_user = User(
            email="admin@medicalai.com",
            password_hash=get_password_hash("admin123"),
            subscription_type=SubscriptionType.PRO
        )
        session.add(admin_user)
        # Create a sample regular user
        regular_user = User(
            email="user@example.com",
            password_hash=get_password_hash("user123"),
            subscription_type=SubscriptionType.FREE
        )
        session.add(regular_user)
        session.commit()
        logger.info("Sample data created successfully")
        logger.info("Sample users created:")
        logger.info("  Admin: admin@medicalai.com / admin123")
        logger.info("  User: user@example.com / user123")
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating sample data: {e}")
        raise
    finally:
        session.close()

def main():
    """
    Main function to initialize the database. Supports optional sample data creation via CLI.
    """
    parser = argparse.ArgumentParser(description="Initialize the Medical AI Assistant database.")
    parser.add_argument(
        "--with-sample-data",
        action="store_true",
        help="Create sample data (admin and regular user) after initializing tables."
    )
    args = parser.parse_args()

    logger.info("Starting database initialization...")
    try:
        create_tables()
        if args.with_sample_data:
            create_sample_data()
        logger.info("Database initialization completed successfully!")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
