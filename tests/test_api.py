"""End-to-end API tests against the real artifacts from ml/run_pipeline.py."""

import math
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "random_forest.joblib").exists(),
    reason="Run `python ml/run_pipeline.py` first to generate model artifacts.",
)

CLASSES = ["Pure Fresh", "Good", "Stale to Spoiled"]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:  # context manager runs the startup lifespan
        yield c


@pytest.fixture(scope="module")
def sample(client):
    return client.get("/dataset/samples/0").json()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_dataset(client):
    body = client.get("/dataset").json()
    assert body["samples"] == 555
    assert body["features"] == 26
    assert body["classes"] == CLASSES
    assert body["class_distribution"] == {"Pure Fresh": 69, "Good": 144, "Stale to Spoiled": 342}
    assert body["test_size"] == 111


def test_models_match_artifacts(client):
    body = client.get("/models").json()
    names = [m["name"] for m in body["models"]]
    assert names == ["Random Forest", "SVM", "XGBoost"]
    assert body["selection"]["test_set_used_for_selection"] is False
    selected = [m["name"] for m in body["models"] if m["selected_for_xai"]]
    assert selected == [body["selection"]["xai_selected_model"]]
    for m in body["models"]:
        assert len(m["confusion_matrix"]) == 3
        assert 0 <= m["accuracy"] <= 1


def test_predict_default_and_each_model(client, sample):
    for model in (None, "Random Forest", "SVM", "XGBoost"):
        payload = {"features": sample["features"]}
        if model:
            payload["model"] = model
        body = client.post("/predict", json=payload).json()
        assert body["predicted_class"] in CLASSES
        assert math.isclose(sum(body["probabilities"].values()), 1.0, abs_tol=1e-6)
        assert body["confidence"] == max(body["probabilities"].values())
        if model:
            assert body["model"] == model


def test_predict_rejects_missing_feature(client, sample):
    features = dict(sample["features"])
    features.pop("MQ2_mean")
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 422
    assert "MQ2_mean" in r.text


def test_predict_rejects_unknown_model(client, sample):
    r = client.post("/predict", json={"features": sample["features"], "model": "KNN"})
    assert r.status_code == 422


def test_feature_importance(client):
    body = client.get("/feature-importance").json()
    assert len(body["features"]) == 26
    scores = [f["importance"] for f in body["features"]]
    assert scores == sorted(scores, reverse=True)


def test_lime_by_sample_index(client):
    body = client.post("/explain/lime", json={"sample_index": 6}).json()
    assert body["explanation_type"] == "local"
    assert body["actual_class"] in CLASSES
    assert body["prediction"] in CLASSES
    assert 0 < len(body["contributions"]) <= 10


def test_shap_local_is_additive(client, sample):
    body = client.post("/explain/shap", json={"features": sample["features"]}).json()
    for c in CLASSES:
        assert len(body["shap_values"][c]) == 26
        total = body["base_values"][c] + sum(body["shap_values"][c])
        assert math.isclose(total, body["output_values"][c], abs_tol=1e-6)


def test_shap_global(client):
    body = client.get("/explain/shap/global", params={"class_name": "Good"}).json()
    assert body["explanation_type"] == "global"
    assert len(body["importance"]) == 26
    assert len(body["summary"][0]["points"]) == 111


def test_explain_requires_exactly_one_input(client, sample):
    both = {"features": sample["features"], "sample_index": 0}
    assert client.post("/explain/lime", json=both).status_code == 422
    assert client.post("/explain/lime", json={}).status_code == 422


def test_unknown_sample_is_404(client):
    assert client.post("/explain/shap", json={"sample_index": 9999}).status_code == 404
