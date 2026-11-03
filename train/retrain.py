"""Modelo corregido (v3): reentrena el Champion con los datos de concept drift (data-v2).

Reparto (ver docs/correccion.md):
  - entrena con  train.csv + lotes 1-9 de produccion_concept_drift.csv
                 (los lotes etiquetados hasta que salta la alarma)
  - evalúa con   lotes 10-12 (los más afectados por el drift, nunca vistos al entrenar)
  - referencia   test.csv (el test original, para ver cuánto se pierde en el mundo "viejo")

Se prueban 3 variantes de cómo usar los lotes 1-9 y solo la mejor (F1 en lotes 10-12) se
registra en el Registry:
  pesos     train + lotes 1-9, con los lotes 1-9 pesando igual en total que train
  recientes solo lotes 1-9 (se descarta el train viejo)
  drift     solo lotes 5-9 (los que ya tienen el concepto nuevo)

Usa los mismos hiperparámetros que el modelo en producción (alias `production`): lo que
cambia son los datos, no el modelo. Registra la versión nueva SIN mover el alias.

Uso:  python -m train.retrain
Requiere MLflow activo (http://localhost:5000) y los datos de data-v2 (`dvc pull`).
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from train.config import CONFIG_PATH, GLOBAL, RUNS
from train.data import DATA_DIR, FEATURES, TARGET
from train.register import ALIAS, MODEL_NAME
from train.train import SEED, TRACKING_URI, build_pipeline, dataframe_hash

EXPERIMENT = "adult-income-correccion"
CONCEPT_FILE = "produccion_concept_drift.csv"
TRAIN_BATCHES = range(1, 10)   # lotes 1-9
EVAL_BATCHES = range(10, 13)   # lotes 10-12


def dvc_md5(name: str) -> str:
    """Hash DVC del archivo de datos (lo que fija su versión exacta)."""
    meta = yaml.safe_load((DATA_DIR / f"{name}.dvc").read_text(encoding="utf-8"))
    return meta["outs"][0]["md5"]


def scores(y, pred, proba) -> dict:
    return {"f1": f1_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "accuracy": accuracy_score(y, pred),
            "roc_auc": roc_auc_score(y, proba)}


def production_run(client: MlflowClient):
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    return mv, client.get_run(mv.run_id)


VARIANTS = {
    "pesos": "train + lotes 1-9, con los lotes 1-9 pesando lo mismo en total que train",
    "recientes": "solo lotes 1-9",
    "drift": "solo lotes 5-9 (concepto ya cambiado)",
}


def build_training(variant: str, base: pd.DataFrame, concept: pd.DataFrame):
    """Devuelve (datos de entrenamiento, pesos por fila o None) según la variante."""
    new = concept[concept["lote"].isin(TRAIN_BATCHES)][base.columns]
    if variant == "pesos":
        data = pd.concat([base, new], ignore_index=True)
        w = np.r_[np.ones(len(base)), np.full(len(new), len(base) / len(new))]
        return data, w
    if variant == "recientes":
        return new.reset_index(drop=True), None
    if variant == "drift":
        return concept[concept["lote"].isin(range(5, 10))][base.columns].reset_index(drop=True), None
    raise ValueError(variant)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    prod_mv, prod_run = production_run(client)
    cfg = next(r for r in RUNS if r["name"] == prod_run.info.run_name)
    print(f"Modelo en produccion: v{prod_mv.version} ({cfg['name']}). Mismos parámetros, datos distintos.")

    concept = pd.read_csv(DATA_DIR / CONCEPT_FILE)
    base = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    ev = concept[concept["lote"].isin(EVAL_BATCHES)]
    old_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{prod_mv.version}")
    old = scores(ev[TARGET], old_model.predict(ev[FEATURES]), old_model.predict_proba(ev[FEATURES])[:, 1])
    print(f"Referencia v{prod_mv.version} en lotes 10-12: F1={old['f1']:.4f}   (eval={len(ev)} filas)")

    mlflow.set_experiment(EXPERIMENT)
    results = {}
    for variant, desc in VARIANTS.items():
        train, w = build_training(variant, base, concept)
        X_train, y_train = train[FEATURES], train[TARGET]
        pipe = build_pipeline(cfg["model"], cfg["params"])
        pipe.fit(X_train, y_train, **({"model__sample_weight": w} if w is not None else {}))
        new = scores(ev[TARGET], pipe.predict(ev[FEATURES]), pipe.predict_proba(ev[FEATURES])[:, 1])
        on_test = scores(test[TARGET], pipe.predict(test[FEATURES]), pipe.predict_proba(test[FEATURES])[:, 1])
        with mlflow.start_run(run_name=f"{cfg['name']}-v3-{variant}") as run:
            mlflow.set_tags({
                "model_family": cfg["family"],
                "uses_class_weight": str("class_weight" in cfg["params"]).lower(),
                "data_hash": dataframe_hash(train),
                "sklearn_version": sklearn.__version__,
                "selection_metric": "f1_lotes_10_12",
                "correccion_de": f"v{prod_mv.version}",
                "variante": variant,
                "drift_aplicado": "concept",
                "dvc_train_md5": dvc_md5("train.csv"),
                "dvc_concept_md5": dvc_md5(CONCEPT_FILE),
                "data_version": "data-v2",
            })
            mlflow.set_tag(MLFLOW_RUN_NOTE, f"Corrección de concept drift, variante '{variant}': {desc}. "
                           "Evaluada en los lotes 10-12.")
            mlflow.log_artifact(str(CONFIG_PATH), artifact_path="config")
            mlflow.log_param("model_type", cfg["model"])
            mlflow.log_params(cfg["params"])
            mlflow.log_params({"seed": SEED, "n_train": len(train), "n_eval": len(ev), "variante": variant})
            mlflow.log_metrics({**new, "f1_test_original": on_test["f1"],
                                f"f1_v{prod_mv.version}_mismos_lotes": old["f1"]})
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle",
                                     signature=infer_signature(X_train, pipe.predict(X_train)),
                                     input_example=X_train.head(3))
        results[variant] = (new, on_test, run.info.run_id, len(train))
        print(f"{variant:10s} n_train={len(train):6d}  F1 lotes 10-12={new['f1']:.4f}  "
              f"(P={new['precision']:.3f} R={new['recall']:.3f})  F1 test original={on_test['f1']:.4f}")

    best = max(results, key=lambda v: results[v][0]["f1"])
    new, on_test, run_id, _ = results[best]
    if new["f1"] <= old["f1"]:
        print("Ninguna variante supera al modelo en producción; no se registra nada.")
        return
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.update_model_version(MODEL_NAME, mv.version,
                                f"Corrección de concept drift, variante '{best}' ({VARIANTS[best]}). "
                                f"Candidato a sustituir a v{prod_mv.version}.")
    for k, v in {"run_id": run_id, "variante": best, "f1_lotes_10_12": f"{new['f1']:.4f}",
                 "f1_test": f"{on_test['f1']:.4f}", "data_version": "data-v2",
                 "drift_aplicado": "concept"}.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, k, v)
    print(f"Mejor variante: {best}. Registrada v{mv.version}; el alias '{ALIAS}' sigue en v{prod_mv.version}.")


if __name__ == "__main__":
    main()
