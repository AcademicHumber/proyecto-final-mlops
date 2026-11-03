"""Capa de predicción del servidor MCP: cliente de la API desplegada en Kubernetes.

La URL sale de la variable de entorno `API_URL`; no es un parámetro de las herramientas, para que
el servidor no pueda usarse para llamar a direcciones arbitrarias.
"""
import os

import requests

from ui.perfiles import EDUCACION_NUM, OPCIONES, a_registro, accion, prioridad

API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
TIMEOUT = 30
MAX_PROSPECTOS = 500


class ApiNoDisponible(RuntimeError):
    pass


def _llamar(metodo: str, ruta: str, **kw):
    try:
        r = requests.request(metodo, f"{API_URL}{ruta}", timeout=TIMEOUT, **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise ApiNoDisponible(
            f"No se pudo usar la API en {API_URL} ({e}). ¿Está el clúster encendido y el túnel abierto "
            f"(`minikube service adult-api --url`)? Las consultas al dataset siguen disponibles.") from e


def info() -> dict:
    return _llamar("GET", "/info")


def _validar(v: dict) -> dict:
    if v.get("education") not in EDUCACION_NUM:
        raise ValueError(f"education debe ser una de: {', '.join(EDUCACION_NUM)}")
    for campo, opciones in OPCIONES.items():
        val = v.get(campo)
        if val is not None and val not in opciones and campo != "education":
            raise ValueError(f"{campo}={val!r} no es válido. Opciones: {', '.join(opciones)}")
    return v


def predecir(prospectos: list[dict]) -> dict:
    """Puntúa prospectos (valores en formato del formulario) y añade la prioridad comercial."""
    if not prospectos:
        raise ValueError("La lista de prospectos está vacía.")
    if len(prospectos) > MAX_PROSPECTOS:
        raise ValueError(f"Máximo {MAX_PROSPECTOS} prospectos por llamada.")
    registros = [a_registro(_validar(p)) for p in prospectos]
    resp = _llamar("POST", "/predict", json=registros)
    items = []
    for p, pred in zip(prospectos, resp["predictions"]):
        prob = pred["probability_over_50k"]
        items.append({"prediccion": pred["prediction"], "probabilidad_mayor_50k": prob,
                      "prioridad": prioridad(prob), "accion": accion(prob), "entrada": p})
    return {"resultados": items, "modelo": resp["model_name"], "version": resp["model_version"],
            "run_id": resp["model_run_id"], "pod": resp["pod"]}
