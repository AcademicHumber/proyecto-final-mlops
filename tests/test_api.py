from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from serving.app import FEATURES, app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # el contexto ejecuta el lifespan (carga el modelo)
        yield c


def _rows(n):
    df = pd.read_csv(ROOT / "data" / "datos.csv", nrows=n)   # primeras filas del dump (base)
    # NaN del CSV -> None para que sea JSON válido
    return [{k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in df[FEATURES].to_dict("records")]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_info(client):
    body = client.get("/info").json()
    assert body["model_name"] == "adult-income-classifier"
    assert body["alias"] == "production"
    assert body["model_version"] and body["model_run_id"] and body["pod"]


def test_predict_real_rows(client):
    r = client.post("/predict", json=_rows(5))
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 5
    for p in body["predictions"]:
        assert p["prediction"] in ("<=50K", ">50K")
        assert 0.0 <= p["probability_over_50k"] <= 1.0
    assert body["pod"] and body["model_version"]


def test_predict_null_categoricals(client):
    row = _rows(1)[0]
    for k in ("workclass", "occupation", "native-country"):
        row[k] = None
    r = client.post("/predict", json=[row])
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 1


def test_predict_empty_list_rejected(client):
    assert client.post("/predict", json=[]).status_code == 422
