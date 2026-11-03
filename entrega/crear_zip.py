"""Genera el ZIP de entrega: documentación, código, manifiestos y evidencia del estado final.

Uso:  python entrega/crear_zip.py
Opcional: con MLflow (http://localhost:5000) y el clúster activos, captura también el estado
del Registry y de los pods. Lo que no esté disponible se anota en el archivo de evidencia.
Salida: entrega/proyecto-final-entrega.zip (no se versiona en Git).
"""
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "entrega" / "proyecto-final-entrega.zip"
TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
INCLUDE = ["README.md", "instrucciones.txt", "requirements.txt", "dvc.yaml", ".dvc/config",
           "docs", "k8s", "docker", "configs", "train", "serving", "drift", "tests", "ui", "mcp_server", ".mcp.json",
           "data/train.csv.dvc", "data/test.csv.dvc", "data/produccion.csv.dvc",
           "data/produccion_concept_drift.csv.dvc", "data/produccion_data_drift.csv.dvc"]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}

REGISTRY_SNIPPET = f"""
from mlflow import MlflowClient
c = MlflowClient({TRACKING!r})
for mv in sorted(c.search_model_versions("name='adult-income-classifier'"), key=lambda m: int(m.version)):
    run = c.get_run(mv.run_id)
    m = {{k: round(v, 4) for k, v in run.data.metrics.items() if k in ("f1", "cv_f1_mean", "roc_auc")}}
    print(f"v{{mv.version}}  run={{run.info.run_name}}  id={{mv.run_id}}  {{m}}")
a = c.get_model_version_by_alias("adult-income-classifier", "production")
print(f"alias production -> v{{a.version}}  run_id={{a.run_id}}")
"""


def sh(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300,
                           env={**os.environ, "PYTHONUTF8": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"})
        return (r.stdout + r.stderr).strip() or "(sin salida)"
    except Exception as e:  # noqa: BLE001 - la evidencia es opcional, se anota el motivo
        return f"(no disponible: {e})"


def shown(cmd: list[str]) -> str:
    if cmd[0] == sys.executable:
        return "python " + (" ".join(cmd[1:]) if cmd[1] != "-c" else "(consulta al Registry con MlflowClient)")
    return " ".join(cmd)


def evidence() -> str:
    parts = [f"# Evidencia del estado final ({datetime.now():%Y-%m-%d %H:%M})"]
    for title, cmd in [
        ("Model Registry de MLflow", [sys.executable, "-c", REGISTRY_SNIPPET]),
        ("Pods del clúster", ["kubectl", "get", "pods", "-o", "wide"]),
        ("Deployments y Services", ["kubectl", "get", "deployment,service"]),
        ("Pruebas automáticas", [sys.executable, "-m", "pytest", "-q"]),
        ("Versiones de datos y checkpoints (etiquetas de Git)", ["git", "tag", "-n1", "--sort=creatordate"]),
        ("Historial de commits", ["git", "log", "--format=%h %ad %an  %s", "--date=short"]),
    ]:
        out = "\n".join(l for l in sh(cmd).splitlines() if "INFO" not in l[:40])
        parts.append(f"\n## {title}\n\n$ {shown(cmd)}\n\n{out}")
    return "\n".join(parts) + "\n"


def files():
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.parts) & SKIP_DIRS) and f.suffix != ".pyc":
                    yield f


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for f in files():
            z.write(f, f.relative_to(ROOT).as_posix())
            n += 1
        z.writestr("evidencia/estado-final.txt", evidence())
    print(f"{OUT.relative_to(ROOT)}: {n} archivos + evidencia/estado-final.txt, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
