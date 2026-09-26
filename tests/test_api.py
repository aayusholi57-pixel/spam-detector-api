from types import SimpleNamespace

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_without_model_returns_service_unavailable(monkeypatch):
    def missing_model():
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(main, "load_model", missing_model)

    response = client.post("/predict", json={"text": "hello"})
    assert response.status_code == 503
    assert response.json()["detail"] == "model missing"


def test_predict_returns_label_and_confidence(monkeypatch):
    fake_model = SimpleNamespace(
        classes_=["ham", "spam"],
        predict_proba=lambda texts: [[0.08, 0.92]],
    )

    monkeypatch.setattr(main, "load_model", lambda: fake_model)

    response = client.post("/predict", json={"text": "free prize now"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"
    assert response.json()["confidence_score"] == 0.92


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
