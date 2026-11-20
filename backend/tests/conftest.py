import pytest
from fastapi.testclient import TestClient

from app import config


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Cada test usa una carpeta de experimentos temporal para no ensuciar backend/experiments/.
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    from app.main import app

    return TestClient(app)


@pytest.fixture()
def fast_experiment(client):
    body = {"name": "test rapido", "target": "soh", "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 20}, "test_batteries": ["B0006"]}
    r = client.post("/api/v1/experiments", json=body)
    assert r.status_code == 201, r.text
    return r.json()
