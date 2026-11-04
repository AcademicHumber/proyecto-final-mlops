"""Monitor de drift sobre el dump temporal (`data/datos.csv`).

Para cada lote de llegadas (periodos 1 a N) calcula:
  - data drift: cuántas de las 14 variables salen en rojo frente al baseline (partición `train`);
  - concept drift: F1 del modelo en producción (vía la API) frente al F1 base (partición `test`).
Veredicto: hay data drift si algún lote tiene variables en rojo; hay concept drift si el F1 cae por debajo del
90 % del base durante 3 lotes seguidos (criterio de drift/concept_drift.py).

    python -m drift.monitor [--hasta-periodo N] [--via api|mlflow] [--no-guardar]

Código de salida: 0 sin alertas, 1 si hay alguna alerta (data o concept drift).

Cada ejecución se guarda en `resultados/drift/<fecha>-p<N>.json` (ignorado por Git, lo lee el servidor MCP) y,
si MLflow responde, en el experimento `adult-income-drift` (un run por ejecución). La lógica vive aquí; la
terminal y el MCP solo la llaman.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from drift.concept_drift import CONSECUTIVE, DROP, first_alert, retrain_threshold
from drift.data_drift import detect_data_drift
from train.data import DATA_DIR, FEATURES, TARGET, cargar_dump, llegadas, particion

ROOT = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = Path(os.getenv("RESULTADOS_DIR", ROOT / "resultados" / "drift"))
EXPERIMENTO = "adult-income-drift"
API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
MAX_PERIODO = 12
LOTE_API = 500   # filas por petición a /predict

# Un predictor recibe un DataFrame con las 14 variables y devuelve (predicciones 0/1, info del modelo).
Predictor = Callable[[pd.DataFrame], "tuple[np.ndarray, dict]"]


def predictor_api(url: str = API_URL) -> Predictor:
    """Predice con la API desplegada (la misma que usan los clientes)."""
    import requests

    def predecir(df: pd.DataFrame):
        preds, info = [], {}
        for i in range(0, len(df), LOTE_API):
            filas = [{k: (None if pd.isna(v) else v) for k, v in r.items()}
                     for r in df[FEATURES].iloc[i:i + LOTE_API].to_dict("records")]
            r = requests.post(f"{url}/predict", json=filas, timeout=60)
            r.raise_for_status()
            cuerpo = r.json()
            preds += [1 if p["prediction"] == ">50K" else 0 for p in cuerpo["predictions"]]
            info = {"version": cuerpo["model_version"], "run_id": cuerpo["model_run_id"],
                    "nombre": cuerpo["model_name"], "fuente": f"api {url}"}
        return np.array(preds), info

    return predecir


def predictor_mlflow() -> Predictor:
    """Predice cargando el modelo `production` del Registry (no necesita el clúster)."""
    from mlflow import MlflowClient

    from drift.concept_drift import MODEL_URI, TRACKING_URI, load_production_model

    modelo = load_production_model()
    mv = MlflowClient(TRACKING_URI).get_model_version_by_alias(*MODEL_URI.removeprefix("models:/").split("@"))
    info = {"version": str(mv.version), "run_id": mv.run_id, "nombre": mv.name, "fuente": f"mlflow {TRACKING_URI}"}
    return lambda df: (modelo.predict(df[FEATURES]), info)


def _f1(df: pd.DataFrame, predictor: Predictor) -> "tuple[float, dict]":
    pred, info = predictor(df)
    from sklearn.metrics import f1_score
    return float(f1_score(df[TARGET], pred)), info


def hash_dump(df: pd.DataFrame) -> str:
    """md5 del CSV de lo analizado. Con el dump completo coincide con el hash de DVC de `datos.csv`; con un
    corte (--hasta-periodo N) coincide con el de la etiqueta de datos de ese momento."""
    csv = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.md5(csv).hexdigest()


def analizar(dump: pd.DataFrame, predictor: Predictor) -> dict:
    """Analiza todos los lotes del dump que se le pase (la base más las llegadas disponibles)."""
    train, test = particion(dump, "train"), particion(dump, "test")
    if train.empty or test.empty:
        raise ValueError("El dump no tiene la base (periodo 0 con train y test).")
    f1_base, info = _f1(test, predictor)
    umbral = retrain_threshold(f1_base)

    lotes = []
    for n in sorted(p for p in dump["periodo"].unique() if p > 0):
        lote = llegadas(dump, n, n)
        rep = detect_data_drift(train, lote[FEATURES])
        rojas = rep.loc[rep["drift"], "feature"].tolist()
        f1, info = _f1(lote, predictor)
        lotes.append({"lote": int(n), "filas": len(lote), "variables_en_rojo": len(rojas), "variables": rojas,
                      "data_drift": bool(rojas), "f1": round(f1, 4), "bajo_umbral": f1 < umbral})

    alerta = first_alert([x["f1"] for x in lotes], umbral)
    lotes_dd = [x["lote"] for x in lotes if x["data_drift"]]
    return {
        "baseline": {"filas_train": len(train), "filas_test": len(test), "f1_base": round(f1_base, 4),
                     "umbral_f1": round(umbral, 4), "regla": f"F1 < {1 - DROP:.0%} del base en {CONSECUTIVE} lotes seguidos"},
        "modelo": info,
        "dump": {"hasta_periodo": int(dump["periodo"].max()), "filas": len(dump), "md5": hash_dump(dump)},
        "lotes": lotes,
        "veredicto": {"data_drift": bool(lotes_dd), "lotes_con_data_drift": lotes_dd,
                      "concept_drift": alerta is not None,
                      "lote_alerta": None if alerta is None else lotes[alerta]["lote"],
                      "alerta": bool(lotes_dd) or alerta is not None},
    }


# ---------- guardado ----------

def _a_mlflow(res: dict, ruta: Path) -> "str | None":
    """Un run por ejecución. Si MLflow no responde no se interrumpe: el JSON ya es la fuente de verdad."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    try:
        import mlflow
        from drift.concept_drift import TRACKING_URI
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENTO)
        v, d = res["veredicto"], res["dump"]
        with mlflow.start_run(run_name=ruta.stem) as run:
            mlflow.set_tags({"periodo": str(d["hasta_periodo"]), "dump_md5": d["md5"],
                             "modelo_version": res["modelo"].get("version", "?"),
                             "alerta": str(v["alerta"]).lower()})
            mlflow.log_metrics({"f1_base": res["baseline"]["f1_base"], "umbral_f1": res["baseline"]["umbral_f1"],
                                "lotes_con_data_drift": len(v["lotes_con_data_drift"]),
                                "lote_alerta_concept_drift": v["lote_alerta"] or 0})
            for x in res["lotes"]:
                mlflow.log_metric("f1_lote", x["f1"], step=x["lote"])
                mlflow.log_metric("variables_en_rojo", x["variables_en_rojo"], step=x["lote"])
            mlflow.log_artifact(str(ruta))
            return run.info.run_id
    except Exception as e:  # noqa: BLE001 - MLflow es secundario
        print(f"(aviso) No se guardó en MLflow: {type(e).__name__}", file=sys.stderr)
        return None


def guardar(res: dict, directorio: Path = RESULTADOS_DIR, mlflow_: bool = True) -> Path:
    directorio.mkdir(parents=True, exist_ok=True)
    ahora = datetime.now()
    res["fecha"] = ahora.isoformat(timespec="seconds")
    ruta = directorio / f"{ahora:%Y%m%d-%H%M%S}-p{res['dump']['hasta_periodo']}.json"
    res["mlflow_run_id"] = None
    _escribir(res, ruta)                      # primero el JSON: es la fuente de verdad
    if mlflow_:
        res["mlflow_run_id"] = _a_mlflow(res, ruta)
        _escribir(res, ruta)                  # y se vuelve a escribir con el id del run
    return ruta


def _escribir(res: dict, ruta: Path) -> None:
    ruta.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


def ejecutar(hasta_periodo: "int | None" = None, predictor: "Predictor | None" = None,
             guardar_resultado: bool = True, ruta_dump: Path = DATA_DIR / "datos.csv",
             directorio: Path = RESULTADOS_DIR, mlflow_: bool = True) -> dict:
    """Analiza el dump (hasta un periodo) y guarda el resultado. Devuelve el resultado con `archivo`."""
    if hasta_periodo is not None and not 0 <= int(hasta_periodo) <= MAX_PERIODO:
        raise ValueError(f"hasta_periodo debe estar entre 0 y {MAX_PERIODO}.")
    dump = cargar_dump(ruta_dump, None if hasta_periodo is None else int(hasta_periodo))
    res = analizar(dump, predictor or predictor_api())
    if guardar_resultado:
        res["archivo"] = guardar(res, directorio, mlflow_).name
    return res


# ---------- lectura de lo guardado ----------

def historial(directorio: Path = RESULTADOS_DIR) -> list[dict]:
    """Resumen de cada ejecución guardada, la más reciente primero."""
    out = []
    for f in sorted(directorio.glob("*.json"), reverse=True) if directorio.exists() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        out.append({"archivo": f.name, "fecha": r.get("fecha"), "hasta_periodo": r["dump"]["hasta_periodo"],
                    "dump_md5": r["dump"]["md5"], **r["veredicto"], "modelo_version": r["modelo"].get("version")})
    return out


def ultimo(directorio: Path = RESULTADOS_DIR) -> "dict | None":
    archivos = sorted(directorio.glob("*.json"), reverse=True) if directorio.exists() else []
    if not archivos:
        return None
    return {"archivo": archivos[0].name, **json.loads(archivos[0].read_text(encoding="utf-8"))}


# ---------- terminal ----------

def imprimir(res: dict) -> None:
    b, v, d = res["baseline"], res["veredicto"], res["dump"]
    print(f"Dump: periodos 0-{d['hasta_periodo']} ({d['filas']} filas, md5 {d['md5'][:8]})   "
          f"Modelo: v{res['modelo'].get('version')} ({res['modelo'].get('fuente')})")
    print(f"F1 base (test) {b['f1_base']:.3f}   umbral {b['umbral_f1']:.3f}   regla: {b['regla']}\n")
    print(f"{'lote':>4} {'filas':>5} {'vars en rojo':>12} {'F1':>7}  estado")
    for x in res["lotes"]:
        estado = ["DATA DRIFT"] * x["data_drift"] + ["F1 bajo umbral"] * x["bajo_umbral"]
        print(f"{x['lote']:>4} {x['filas']:>5} {x['variables_en_rojo']:>6}/14    {x['f1']:>7.3f}  "
              f"{', '.join(estado) or 'ok'}")
    print()
    print(f"Data drift:    {'SÍ, lotes ' + str(v['lotes_con_data_drift']) if v['data_drift'] else 'no'}")
    print(f"Concept drift: {'SÍ, alerta en el lote ' + str(v['lote_alerta']) if v['concept_drift'] else 'no'}")
    print(f"Veredicto:     {'ALERTA' if v['alerta'] else 'todo en orden'}")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hasta-periodo", type=int, help=f"analiza solo hasta este periodo (0 a {MAX_PERIODO})")
    ap.add_argument("--via", choices=["api", "mlflow"], default="api", help="cómo obtener las predicciones")
    ap.add_argument("--no-guardar", action="store_true", help="no escribe JSON ni MLflow")
    args = ap.parse_args(argv)
    pred = predictor_api() if args.via == "api" else predictor_mlflow()
    res = ejecutar(args.hasta_periodo, pred, guardar_resultado=not args.no_guardar)
    imprimir(res)
    if "archivo" in res:
        print(f"Guardado en resultados/drift/{res['archivo']}"
              + (f" y en MLflow (run {res['mlflow_run_id'][:8]})" if res.get("mlflow_run_id") else ""))
    return 1 if res["veredicto"]["alerta"] else 0


if __name__ == "__main__":
    sys.exit(main())
