import pytest
from backend.auth import get_password_hash, verify_password, authenticate_user, get_current_active_user
from backend.models import User, SubscriptionType
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models import Base
from fastapi import HTTPException, Depends

@pytest.fixture(scope="module")
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    engine.dispose()

def test_password_hash_and_verify():
    password = "mysecretpassword"
    hashed = get_password_hash(password)
    assert verify_password(password, hashed)
    assert not verify_password("wrongpassword", hashed)

def test_authenticate_user_success_and_failure(test_db):
    user = User(email="auth@example.com", password_hash=get_password_hash("testpass"), is_active=1)
    test_db.add(user)
    test_db.commit()
    # Success
    result = authenticate_user(test_db, "auth@example.com", "testpass")
    assert result.email == "auth@example.com"
    # Failure: wrong password
    assert authenticate_user(test_db, "auth@example.com", "wrongpass") is False
    # Failure: non-existent user
    assert authenticate_user(test_db, "nouser@example.com", "testpass") is False

def test_get_current_active_user_inactive():
    class DummyUser:
        email = "inactive@example.com"
        is_active = 0
    with pytest.raises(HTTPException) as excinfo:
        get_current_active_user(DummyUser())
    assert excinfo.value.status_code == 403
    assert "Inactive user" in str(excinfo.value.detail) 