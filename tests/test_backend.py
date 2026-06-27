import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint for uptime probes"""
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "lung-cancer-detection-api"}

def test_root_endpoint():
    """Test the root welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_login_invalid_credentials():
    """Test that invalid credentials return 400 (or 401 depending on FastAPI security config)"""
    response = client.post("/auth/login", data={"username": "invalid_user", "password": "wrongpassword"})
    assert response.status_code in [400, 401]

def test_history_unauthenticated():
    """Test that accessing history without a JWT token returns 401 Unauthorized"""
    response = client.get("/history/")
    assert response.status_code == 401
