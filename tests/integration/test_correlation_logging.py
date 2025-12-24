from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_correlation_id_propagation(caplog):
    correlation_id = "test-correlation-id-123"

    with caplog.at_level("INFO"):
        response = client.get(
            "/health",
            headers={"X-Correlation-ID": correlation_id},
        )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == correlation_id

    logs = [record.message for record in caplog.records]

    # At least one log line must contain correlation_id
    assert any(correlation_id in log for log in logs)
