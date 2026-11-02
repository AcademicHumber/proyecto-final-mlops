"""Detección de concept drift: seguimiento del F1 por lotes sucesivos de producción.

Requiere etiquetas reales, que en la práctica llegan con retraso (ver docs/drift.md).

Criterio de reentrenamiento (explícito):
  F1 del lote < F1_base * (1 - DROP)  durante  CONSECUTIVE lotes seguidos.
  - F1_base: F1 del modelo en test (referencia sin drift).
  - DROP = 10 %: la caída que fijamos como inaceptable en el caso de negocio.
  - CONSECUTIVE = 3: exige caída sostenida, para no reentrenar por el ruido de un lote.
"""
import os

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from drift.inject import inject_concept_drift
from train.data import DATA_DIR, FEATURES, TARGET

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DROP = 0.10
CONSECUTIVE = 3
MODEL_URI = "models:/adult-income-classifier@production"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_production_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


def f1_of(model, df: pd.DataFrame) -> float:
    return float(f1_score(df[TARGET], model.predict(df[FEATURES])))


def make_stream(prod: pd.DataFrame, n_batches: int = 12, drift_start: int | None = None,
                max_strength: float = 1.0, seed: int = 0) -> list[pd.DataFrame]:
    """Parte producción en lotes consecutivos. Desde `drift_start` el concepto cambia y la
    intensidad sube linealmente hasta `max_strength` en el último lote. Sin `drift_start`, todo limpio."""
    bounds = np.linspace(0, len(prod), n_batches + 1).astype(int)
    batches = [prod.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    if drift_start is None:
        return batches
    steps = n_batches - drift_start
    out = []
    for i, b in enumerate(batches):
        s = 0.0 if i < drift_start else max_strength * (i - drift_start + 1) / steps
        out.append(inject_concept_drift(b, s, seed=seed + i))
    return out


def batch_f1s(model, batches: list[pd.DataFrame]) -> list[float]:
    return [f1_of(model, b) for b in batches]


def retrain_threshold(f1_base: float, drop: float = DROP) -> float:
    return f1_base * (1 - drop)


def first_alert(f1s: list[float], threshold: float, consecutive: int = CONSECUTIVE) -> int | None:
    """Índice del lote en que se confirma la alerta (el último de la racha), o None."""
    run = 0
    for i, f in enumerate(f1s):
        run = run + 1 if f < threshold else 0
        if run >= consecutive:
            return i
    return None


def plot(f1s: list[float], f1_base: float, threshold: float, alert: int | None, path) -> None:
    x = np.arange(1, len(f1s) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, f1s, "o-", label="F1 por lote")
    ax.axhline(f1_base, color="gray", ls="--", label=f"F1 base (test) = {f1_base:.3f}")
    ax.axhline(threshold, color="red", ls="--", label=f"Umbral de reentrenamiento = {threshold:.3f}")
    if alert is not None:
        ax.axvline(alert + 1, color="red", alpha=0.4)
        ax.annotate(f"alerta (lote {alert + 1})", (alert + 1, threshold), xytext=(8, 14),
                    textcoords="offset points", color="red")
    ax.set_xlabel("Lote de producción (orden temporal)")
    ax.set_ylabel("F1")
    ax.set_xticks(x)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    model = load_production_model()
    f1_base = f1_of(model, pd.read_csv(DATA_DIR / "test.csv"))
    thr = retrain_threshold(f1_base)
    prod = pd.read_csv(DATA_DIR / "produccion.csv")
    for nombre, kw in [("limpio", {}), ("con concept drift", {"drift_start": 4})]:
        f1s = batch_f1s(model, make_stream(prod, **kw))
        al = first_alert(f1s, thr)
        print(f"{nombre:18s} F1 por lote: {[round(f, 3) for f in f1s]}  alerta en lote: "
              f"{None if al is None else al + 1}")
    print(f"F1 base {f1_base:.4f}  umbral {thr:.4f}")
    img = DATA_DIR.parent / "docs" / "img"
    img.mkdir(exist_ok=True)
    plot(f1s, f1_base, thr, al, img / "concept_drift.png")   # f1s/al: flujo con drift (último del bucle)
    print(f"Gráfica: {img / 'concept_drift.png'}")
