"""Without artifacts the API must start, report 'degraded', and return a clear
503 (never a stack trace) instead of crashing."""

from fastapi.testclient import TestClient


def test_missing_artifacts_gives_clear_503(tmp_path, monkeypatch):
    from backend.services import model_service

    monkeypatch.setattr(model_service, "ARTIFACTS_DIR", tmp_path)
    from backend.main import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert "run_pipeline.py" in health["detail"]

        r = client.get("/models")
        assert r.status_code == 503
        assert "run_pipeline.py" in r.json()["detail"]
        assert "Traceback" not in r.text
