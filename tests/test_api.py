import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by fastapi.testclient

from fastapi.testclient import TestClient  # noqa: E402

from tests.conftest import SCAM  # noqa: E402


@pytest.fixture()
def client(tiny_models_dir, monkeypatch):
    import src.predict as predict_mod
    from api.main import app

    predict_mod.get_predictor.cache_clear()
    monkeypatch.setattr(predict_mod, "get_predictor",
                        lambda: predict_mod.Predictor(models_dir=tiny_models_dir))
    import api.main as main_mod

    monkeypatch.setattr(main_mod, "get_predictor",
                        lambda: predict_mod.Predictor(models_dir=tiny_models_dir))
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json={"text": SCAM, "email": "hr@gmail.com"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Fake"
    assert 0 <= body["risk_score"] <= 100
    assert isinstance(body["red_flags"], list) and body["red_flags"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}])
def test_predict_rejects_empty(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_too_long(client):
    assert client.post("/predict", json={"text": "a " * 15_000}).status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200 and "results" in r.json()
