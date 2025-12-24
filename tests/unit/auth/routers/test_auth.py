import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.models.enums import UserRole
from auth.models.schemas import TokenOut, UserOut
from auth.routers.auth import router
from auth.services.auth import AuthService, get_auth_service, get_current_user_service


@pytest.fixture
def mock_auth_service():
    """Mock AuthService"""
    return AsyncMock(spec=AuthService)


@pytest.fixture
def app(mock_auth_service):
    """Create test app with dependency overrides"""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_auth_service] = lambda: mock_auth_service
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


def test_register_user_success(client, mock_auth_service):
    """Test POST /auth/register creates new user"""
    mock_auth_service.register_user.return_value = UserOut(
        id=uuid.uuid4(),
        email="new@example.com",
        is_active=True,
        role=UserRole.VIEWER,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    response = client.post(
        "/auth/register", json={"email": "new@example.com", "password": "Pass123!"}
    )

    assert response.status_code == 200
    assert response.json()["email"] == "new@example.com"


def test_login_success(client, mock_auth_service):
    """Test POST /auth/token returns JWT"""
    mock_auth_service.authenticate_user.return_value = TokenOut(
        access_token="jwt_abc123"
    )

    response = client.post(
        "/auth/token", json={"email": "user@example.com", "password": "pass123"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["access_token"] == "jwt_abc123"
    assert data["token_type"] == "bearer"


def test_get_me_returns_current_user(client, app):
    """Test GET /auth/me returns authenticated user"""
    mock_user = UserOut(
        id=uuid.uuid4(),
        email="current@example.com",
        is_active=True,
        role=UserRole.MEMBER,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    app.dependency_overrides[get_current_user_service] = lambda: mock_user

    response = client.get("/auth/me")

    assert response.status_code == 200
    assert response.json()["email"] == "current@example.com"
