"""Carga y división reproducible del dataset Adult Income.

Split documentado (estratificado por la etiqueta, semilla fija):
  - 60 % train       -> entrena el modelo y es el baseline de referencia del drift
  - 20 % test        -> evalúa el modelo
  - 20 % producción  -> simula el tráfico futuro (lotes para drift)
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from train.config import GLOBAL

SEED = GLOBAL["seed"]
TARGET = "income"
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
FEATURES = NUMERIC + CATEGORICAL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_adult() -> pd.DataFrame:
    """Descarga Adult desde OpenML y devuelve X + columna binaria `income` (1 = >50K)."""
    raw = fetch_openml("adult", version=2, as_frame=True).frame
    df = raw[FEATURES].copy()
    for col in CATEGORICAL:
        df[col] = df[col].astype("object").replace("?", pd.NA)
    # float64 en todas las numéricas: evita errores de esquema en MLflow cuando hay NaN/enteros
    df[NUMERIC] = df[NUMERIC].astype("float64")
    df[TARGET] = (raw["class"].astype(str).str.strip() == ">50K").astype(int)
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Devuelve (train, test, produccion) con proporciones 60/20/20."""
    train, rest = train_test_split(df, test_size=0.4, random_state=SEED, stratify=df[TARGET])
    test, prod = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=rest[TARGET])
    return (train.reset_index(drop=True), test.reset_index(drop=True),
            prod.reset_index(drop=True))


def save_splits(train, test, prod) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    prod.to_csv(DATA_DIR / "produccion.csv", index=False)
