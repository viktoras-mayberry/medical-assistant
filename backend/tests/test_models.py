import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models import Base, User, Interaction, Subscription, SubscriptionType
from datetime import datetime

@pytest.fixture(scope="module")
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    engine.dispose()

def test_user_creation(test_db):
    user = User(email="test@example.com", password_hash="hashed", subscription_type=SubscriptionType.FREE)
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    assert user.id is not None
    assert user.is_active == 1
    assert user.subscription_type == SubscriptionType.FREE

def test_interaction_creation(test_db):
    user = User(email="interaction@example.com", password_hash="hashed")
    test_db.add(user)
    test_db.commit()
    interaction = Interaction(user_id=user.id, interaction_type="question", content="What is AI?")
    test_db.add(interaction)
    test_db.commit()
    test_db.refresh(interaction)
    assert interaction.id is not None
    assert interaction.user_id == user.id
    assert interaction.content == "What is AI?"

def test_subscription_creation(test_db):
    user = User(email="sub@example.com", password_hash="hashed")
    test_db.add(user)
    test_db.commit()
    subscription = Subscription(user_id=user.id, plan="pro", start_date=datetime.utcnow())
    test_db.add(subscription)
    test_db.commit()
    test_db.refresh(subscription)
    assert subscription.id is not None
    assert subscription.user_id == user.id
    assert subscription.plan == "pro" 