"""Registra modelos en el Model Registry de MLflow y asigna el alias `production`.

  v1 = mejor Regresión Logística (modelo base simple e interpretable)
  v2 = Champion: mejor run por F1 en validación cruzada -> alias `production`

Uso:  python -m train.register
"""
import mlflow
from mlflow import MlflowClient

from train.config import GLOBAL
from train.train import TRACKING_URI

MODEL_NAME = "adult-income-classifier"
ALIAS = "production"
EXPERIMENT = GLOBAL["experiment_name"]
SELECTION_METRIC = GLOBAL["selection_metric"]  # el mismo criterio con el que se entrenó


def best_run(client: MlflowClient, experiment_id: str, model_type: str | None = None):
    flt = f"params.model_type = '{model_type}'" if model_type else ""
    runs = client.search_runs([experiment_id], filter_string=flt,
                              order_by=[f"metrics.{SELECTION_METRIC} DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"No hay runs para {model_type or 'el experimento'}; ejecuta primero train.train")
    return runs[0]


def register(client: MlflowClient, run, note: str):
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    client.update_model_version(MODEL_NAME, mv.version, description=note)
    client.set_model_version_tag(MODEL_NAME, mv.version, "run_id", run.info.run_id)
    client.set_model_version_tag(MODEL_NAME, mv.version, SELECTION_METRIC, f"{run.data.metrics[SELECTION_METRIC]:.4f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "f1_test", f"{run.data.metrics['f1']:.4f}")
    print(f"v{mv.version} <- run {run.info.run_name} ({run.info.run_id[:8]}) {SELECTION_METRIC}={run.data.metrics[SELECTION_METRIC]:.4f} f1_test={run.data.metrics['f1']:.4f}")
    return mv


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    if client.search_model_versions(f"name='{MODEL_NAME}'"):
        raise SystemExit(f"'{MODEL_NAME}' ya tiene versiones registradas; no se duplican. "
                         "Borra mlflow.db para empezar de cero.")

    exp = client.get_experiment_by_name(EXPERIMENT)
    best = best_run(client, exp.experiment_id)
    baseline = GLOBAL["baseline_model"]
    base_type = ("RandomForestClassifier" if best.data.params["model_type"] == baseline else baseline)
    base = best_run(client, exp.experiment_id, base_type)

    register(client, base, f"Modelo base: mejor {base_type}")
    mv = register(client, best, "Champion: mejor F1 en validación cruzada")

    client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    print(f"alias '{ALIAS}' -> v{mv.version}")


if __name__ == "__main__":
    main()
