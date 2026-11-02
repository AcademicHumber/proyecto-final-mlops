"""API de inferencia: sirve el modelo registrado en MLflow, resuelto siempre por alias.

Variables de entorno:
  MLFLOW_TRACKING_URI  (defecto: http://localhost:5000)
  MODEL_URI            (defecto: models:/adult-income-classifier@production)
"""
import os
import re
import socket
from contextlib import asynccontextmanager
from typing import List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/adult-income-classifier@production")

# Orden de columnas idéntico al del entrenamiento (train/data.py). Se duplica a propósito
# para que la imagen de serving no dependa del código de entrenamiento.
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
FEATURES = NUMERIC + CATEGORICAL
LABELS = {0: "<=50K", 1: ">50K"}


def pod_name() -> str:
    return os.getenv("HOSTNAME") or socket.gethostname()


class Record(BaseModel):
    """Una persona. Los nombres con guion se aceptan por alias; las categóricas pueden ser null."""
    model_config = ConfigDict(populate_by_name=True)

    age: float
    fnlwgt: float
    education_num: float = Field(alias="education-num")
    capital_gain: float = Field(alias="capital-gain")
    capital_loss: float = Field(alias="capital-loss")
    hours_per_week: float = Field(alias="hours-per-week")
    workclass: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = Field(default=None, alias="marital-status")
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    native_country: Optional[str] = Field(default=None, alias="native-country")


class Prediction(BaseModel):
    prediction: str
    probability_over_50k: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_name: str
    model_version: str
    model_run_id: str
    pod: str


def parse_model_uri(uri: str):
    m = re.fullmatch(r"models:/([^/@]+)@([^/@]+)", uri)
    if not m:
        raise ValueError(f"MODEL_URI debe ser por alias (models:/<nombre>@<alias>), recibido: {uri}")
    return m.group(1), m.group(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    try:
        name, alias = parse_model_uri(MODEL_URI)
        mlflow.set_tracking_uri(TRACKING_URI)
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        # Se carga el flavor sklearn (no pyfunc) para poder usar predict_proba.
        app.state.model = mlflow.sklearn.load_model(MODEL_URI)
        app.state.info = {"model_name": name, "alias": alias,
                          "model_version": str(mv.version), "model_run_id": mv.run_id}
    except Exception as exc:  # el servicio arranca; /health informará 503
        app.state.load_error = repr(exc)
    yield


app = FastAPI(title="Adult Income - API de inferencia", lifespan=lifespan)


def _require_model():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail=f"Modelo no cargado: {getattr(app.state, 'load_error', '')}")


@app.get("/health")
def health():
    _require_model()
    return {"status": "ok", "pod": pod_name()}


@app.get("/info")
def info():
    _require_model()
    return {**app.state.info, "pod": pod_name()}


@app.post("/predict", response_model=PredictResponse)
def predict(records: List[Record]):
    _require_model()
    if not records:
        raise HTTPException(status_code=422, detail="La lista de registros está vacía")
    df = pd.DataFrame([r.model_dump(by_alias=True) for r in records])[FEATURES]
    df[NUMERIC] = df[NUMERIC].astype("float64")
    # None -> NaN: el SimpleImputer del pipeline solo reconoce np.nan como faltante.
    df[CATEGORICAL] = df[CATEGORICAL].astype(object).where(df[CATEGORICAL].notna(), np.nan)

    proba = app.state.model.predict_proba(df)[:, 1]
    preds = [Prediction(prediction=LABELS[int(p >= 0.5)], probability_over_50k=round(float(p), 4))
             for p in proba]
    return PredictResponse(predictions=preds, pod=pod_name(),
                           **{k: app.state.info[k] for k in ("model_name", "model_version", "model_run_id")})
