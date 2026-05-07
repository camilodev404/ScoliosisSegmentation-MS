from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_service_status() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "ScoliosisSegmentation-MS"
    assert "model_ready" in payload
    assert "missing_artifacts" in payload

