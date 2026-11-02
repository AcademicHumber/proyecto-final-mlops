"""Carga y valida la configuración del experimento (configs/experiments.yaml).

El YAML decide *qué* se entrena; el código decide *cómo*. Los modelos permitidos
son una lista cerrada: el YAML elige entre ellos, no importa código arbitrario.
"""
import os
from pathlib import Path

import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CONFIG_PATH = Path(os.getenv("EXPERIMENT_CONFIG",
                             Path(__file__).resolve().parent.parent / "configs" / "experiments.yaml"))

MODELS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}
FAMILIES = {"linear", "bagging", "boosting"}
GLOBAL_KEYS = {"experiment_name", "seed", "cv_folds", "top_fraction", "selection_metric", "baseline_model"}
RUN_KEYS = {"name", "model", "family", "params", "description"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = GLOBAL_KEYS - set(cfg.get("global", {}))
    if missing:
        raise ValueError(f"Faltan claves en 'global': {sorted(missing)}")

    names = set()
    for run in cfg.get("runs", []):
        if RUN_KEYS - set(run):
            raise ValueError(f"Run incompleto {run.get('name')!r}: faltan {sorted(RUN_KEYS - set(run))}")
        if run["model"] not in MODELS:
            raise ValueError(f"{run['name']}: modelo desconocido {run['model']!r}; válidos: {sorted(MODELS)}")
        if run["family"] not in FAMILIES:
            raise ValueError(f"{run['name']}: family {run['family']!r} no está en {sorted(FAMILIES)}")
        if run["name"] in names:
            raise ValueError(f"Nombre de run repetido: {run['name']!r}")
        names.add(run["name"])
        run["description"] = " ".join(run["description"].split())  # plegar saltos de línea
    if not names:
        raise ValueError("El archivo no define ningún run")
    if cfg["global"]["baseline_model"] not in MODELS:
        raise ValueError(f"baseline_model desconocido: {cfg['global']['baseline_model']!r}")
    return cfg


CONFIG = load_config()
GLOBAL = CONFIG["global"]
RUNS = CONFIG["runs"]
