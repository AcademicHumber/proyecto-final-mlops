"""Entrenamiento con MLflow: 9 ejecuciones (RandomForest, GradientBoosting, Regresión Logística).

Criterio de selección: F1 medio en validación cruzada sobre train (`cv_f1_mean`).
El conjunto de test solo se usa para reportar, no para elegir.

Uso:  python -m train.train
Requiere el servidor MLflow activo (ver docs/mlflow-server.md): http://localhost:5000
"""
import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train.data import CATEGORICAL, FEATURES, NUMERIC, SEED, TARGET, load_adult, save_splits, split_data

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "adult-income"

# (nombre del run, clase del modelo, hiperparámetros)
CONFIGS = [
    ("rf-n50-d5", RandomForestClassifier, {"n_estimators": 50, "max_depth": 5}),
    ("rf-n100-d10", RandomForestClassifier, {"n_estimators": 100, "max_depth": 10}),
    ("rf-n200-d20", RandomForestClassifier, {"n_estimators": 200, "max_depth": 20}),
    ("gb-n50-lr0.1-d3", GradientBoostingClassifier, {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}),
    ("gb-n100-lr0.1-d3", GradientBoostingClassifier, {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
    ("gb-n200-lr0.05-d5", GradientBoostingClassifier, {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5}),
    # Línea base simple e interpretable, y prueba del desbalance de clases (class_weight)
    ("lr-C1", LogisticRegression, {"C": 1.0, "max_iter": 1000}),
    ("lr-C1-balanced", LogisticRegression, {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
    ("rf-n200-d20-balanced", RandomForestClassifier, {"n_estimators": 200, "max_depth": 20, "class_weight": "balanced"}),
]

CV_FOLDS = 5
TOP_FRACTION = 0.10  # campaña con presupuesto limitado: se contacta al 10 % mejor puntuado


def precision_at_top(y_true, proba, fraction=TOP_FRACTION) -> float:
    """De los prospectos con mayor probabilidad (top `fraction`), qué proporción es realmente >50K."""
    k = max(1, int(len(y_true) * fraction))
    top = np.argsort(proba)[::-1][:k]
    return float(np.asarray(y_true)[top].mean())


def build_pipeline(model_cls, params) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([("prep", preprocess), ("model", model_cls(random_state=SEED, **params))])


def main() -> None:
    df = load_adult()
    train, test, prod = split_data(df)
    save_splits(train, test, prod)
    print(f"train={len(train)} test={len(test)} produccion={len(prod)}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    for run_name, model_cls, params in CONFIGS:
        with mlflow.start_run(run_name=run_name):
            pipe = build_pipeline(model_cls, params)
            # Validación cruzada solo sobre train: criterio de selección sin tocar test
            cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
            cv_f1 = cross_val_score(build_pipeline(model_cls, params), X_train, y_train,
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

            mlflow.log_param("model_type", model_cls.__name__)
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
            print(f"{run_name:22s} cv_f1={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f} "
                  f"f1={metrics['f1']:.4f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"auc={metrics['roc_auc']:.4f} p@10={metrics['precision_top10']:.3f}")


if __name__ == "__main__":
    main()
