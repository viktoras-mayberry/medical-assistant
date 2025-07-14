import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json() or response.text.lower().find("ok") != -1

def test_chat_endpoint():
    # This test assumes /chat expects a JSON body with 'message' and returns a JSON with 'response'
    payload = {"message": "Hello, what is hypertension?"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)

def test_users_me_requires_auth():
    # Should return 401 if no token is provided
    response = client.get("/users/me")
    assert response.status_code == 401
    assert "detail" in response.json()

def test_auth_login_and_users_me():
    # Register a user first
    email = "apitest@example.com"
    password = "apitestpass"
    register_payload = {"email": email, "password": password}
    reg_response = client.post("/auth/register", json=register_payload)
    assert reg_response.status_code == 200 or reg_response.status_code == 400  # 400 if already registered
    # Login
    login_payload = {"username": email, "password": password}
    login_response = client.post("/auth/login", data=login_payload)
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    # Access /users/me with token
    headers = {"Authorization": f"Bearer {token}"}
    me_response = client.get("/users/me", headers=headers)
    assert me_response.status_code == 200
    data = me_response.json()
    assert data["email"] == email 