"""Entrenamiento con MLflow. Qué se entrena lo define configs/experiments.yaml.

Criterio de selección: la métrica `selection_metric` del YAML (F1 medio en validación
cruzada sobre train). El conjunto de test solo se usa para reportar, no para elegir.

Uso:  python -m train.train
      python -m train.train --only rf-n50-d5,lr-C1 --experiment-name prueba
Requiere el servidor MLflow activo (ver docs/mlflow-server.md): http://localhost:5000
"""
import argparse
import hashlib
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train.config import CONFIG_PATH, GLOBAL, MODELS, RUNS
from train.data import CATEGORICAL, DUMP, FEATURES, NUMERIC, TARGET, cargar_dump, particion

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SEED = GLOBAL["seed"]


def dataframe_hash(df: pd.DataFrame) -> str:
    """Huella corta del contenido de los datos: dos runs con el mismo hash usaron los mismos datos."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


def run_tags(run: dict, data_hash: str) -> dict:
    """Tags descriptivos y fijos: hechos del momento del entrenamiento, sin mantenimiento posterior."""
    return {
        "model_family": run["family"],
        "uses_class_weight": str("class_weight" in run["params"]).lower(),
        "data_hash": data_hash,
        "sklearn_version": sklearn.__version__,
        "selection_metric": GLOBAL["selection_metric"],
    }


def precision_at_top(y_true, proba, fraction=GLOBAL["top_fraction"]) -> float:
    """De los prospectos con mayor probabilidad (top `fraction`), qué proporción es realmente >50K."""
    k = max(1, int(len(y_true) * fraction))
    top = np.argsort(proba)[::-1][:k]
    return float(np.asarray(y_true)[top].mean())


def build_pipeline(model_name: str, params: dict) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([("prep", preprocess), ("model", MODELS[model_name](random_state=SEED, **params))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="nombres de runs separados por coma (por defecto, todos)")
    parser.add_argument("--experiment-name", default=GLOBAL["experiment_name"])
    args = parser.parse_args()

    runs = RUNS
    if args.only:
        wanted = set(args.only.split(","))
        runs = [r for r in RUNS if r["name"] in wanted]
        unknown = wanted - {r["name"] for r in runs}
        if unknown:
            raise SystemExit(f"Runs inexistentes en {CONFIG_PATH.name}: {sorted(unknown)}")

    if not DUMP.exists():
        raise SystemExit(f"Falta {DUMP}. Recupéralo con `dvc pull` (o regéneralo: python -m drift.escenarios --desde-openml).")
    base = cargar_dump(hasta_periodo=0)   # el modelo solo ve la base: el periodo 0 del dump
    train, test = particion(base, "train"), particion(base, "test")
    print(f"train={len(train)} test={len(test)} (periodo 0 del dump)")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    data_hash = dataframe_hash(train)
    print(f"data_hash={data_hash}")

    for run in runs:
        name, model_name, params = run["name"], run["model"], run["params"]
        with mlflow.start_run(run_name=name):
            mlflow.set_tags(run_tags(run, data_hash))
            mlflow.set_tag(MLFLOW_RUN_NOTE, run["description"])  # campo "Description" de la interfaz
            mlflow.log_artifact(str(CONFIG_PATH), artifact_path="config")  # configuración exacta del experimento

            pipe = build_pipeline(model_name, params)
            # Validación cruzada solo sobre train: criterio de selección sin tocar test
            cv = StratifiedKFold(GLOBAL["cv_folds"], shuffle=True, random_state=SEED)
            cv_f1 = cross_val_score(build_pipeline(model_name, params), X_train, y_train,
                                    cv=cv, scoring="f1", n_jobs=-1)

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics = {
                "f1": f1_score(y_test, pred),
                "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred),
                "accuracy": accuracy_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba),
                "precision_top10": precision_at_top(y_test, proba),
                "cv_f1_mean": float(cv_f1.mean()),
                "cv_f1_std": float(cv_f1.std()),
            }
            if GLOBAL["selection_metric"] not in metrics:
                raise SystemExit(f"selection_metric {GLOBAL['selection_metric']!r} no es una métrica registrada: "
                                 f"{sorted(metrics)}")

            mlflow.log_param("model_type", model_name)
            mlflow.log_params(params)
            mlflow.log_params({"seed": SEED, "split": "60/20/20 estratificado",
                               "n_train": len(train), "n_test": len(test)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipe, name="model",
                # MLflow 3.x usa skops por defecto y bloquea los árboles de sklearn como
                # "tipos no confiables". Modelo propio y de origen confiable: usamos cloudpickle.
                serialization_format="cloudpickle",
                signature=infer_signature(X_train, pipe.predict(X_train)),
                input_example=X_train.head(3),
            )
            print(f"{name:22s} cv_f1={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f} "
                  f"f1={metrics['f1']:.4f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"auc={metrics['roc_auc']:.4f} p@10={metrics['precision_top10']:.3f}")


if __name__ == "__main__":
    main()
