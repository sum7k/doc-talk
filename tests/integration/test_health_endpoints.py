from unittest.mock import patch

from fastapi.testclient import TestClient

from core.exceptions import ReadinessError
from main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_when_db_healthy():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_db_fails():
    with patch("main.check_db_readiness", side_effect=ReadinessError("DB error")):
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready"}
