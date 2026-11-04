"""Crea las etiquetas de datos: el dump completo se versiona "creciendo" por momentos.

Para cada momento deja en `data/datos.csv` solo las filas hasta ese periodo, lo registra con DVC, hace un
commit y le pone una etiqueta anotada. Se ejecuta **una sola vez, al final**, sobre el código ya terminado,
de modo que en cada etiqueta está el mismo código y solo cambian los datos. El último momento (el dump
completo) es el estado de la rama en la que se ejecuta.

    python -m drift.versionar_dump --dry-run     # solo muestra lo que haría
    python -m drift.versionar_dump               # crea commits y etiquetas (no sube nada)
    git push origin <rama> --tags; dvc push      # publicar, a mano

Cada archivo intermedio es un prefijo exacto del completo (las filas están ordenadas por periodo).
"""
import argparse
import subprocess
import sys

import pandas as pd

from train.data import DATA_DIR

RUTA = DATA_DIR / "datos.csv"
RUTA_DVC = DATA_DIR / "datos.csv.dvc"

# (etiqueta, hasta_periodo, descripción)
MOMENTOS = [
    ("datos-t0-base", 0, "T0: base (train + test); baseline y modelo v2"),
    ("datos-t1-limpio", 2, "T1: + lotes 1-2 limpios (sin drift)"),
    ("datos-t2-data-drift", 4, "T2: + lotes 3-4 con data drift"),
    ("datos-t3-concept-drift", 9, "T3: + lotes 5-9 con concept drift; salta la alerta"),
    ("datos-t4-completo", 12, "T4: + lotes 10-12; dump completo"),
]


def _run(*cmd: str, check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode != 0:
        sys.exit(f"Falló `{' '.join(cmd)}`:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def lineas_hasta(texto: str, periodos: pd.Series, hasta: int) -> str:
    """Cabecera y las filas de periodo <= hasta (prefijo exacto del archivo completo)."""
    n = int((periodos <= hasta).sum())
    lineas = texto.splitlines(keepends=True)
    return "".join(lineas[:n + 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    completo = RUTA.read_text(encoding="utf-8")
    periodos = pd.read_csv(RUTA, usecols=["periodo"])["periodo"]
    if not periodos.is_monotonic_increasing:
        sys.exit("El dump debe estar ordenado por periodo.")

    if not args.dry_run:
        if _run("git", "status", "--porcelain"):
            sys.exit("Hay cambios sin guardar: haz commit antes (las etiquetas deben colgar de código terminado).")
        existentes = [t for t, _, _ in MOMENTOS if _run("git", "tag", "-l", t, check=False)]
        if existentes:
            sys.exit(f"Ya existen las etiquetas {existentes}; no se sobrescriben.")

    for etiqueta, hasta, desc in MOMENTOS:
        filas = int((periodos <= hasta).sum())
        print(f"{etiqueta:26s} periodo <= {hasta:2d}  {filas:6d} filas  {desc}")
        if args.dry_run:
            continue
        # El último momento es el dump completo: vuelve a dejar exactamente el archivo original.
        RUTA.write_text(lineas_hasta(completo, periodos, hasta), encoding="utf-8", newline="")
        _run("dvc", "add", str(RUTA))
        _run("git", "add", str(RUTA_DVC), str(DATA_DIR / ".gitignore"))
        _run("git", "commit", "-q", "-m", f"Datos {etiqueta}: {desc}")
        _run("git", "tag", "-a", etiqueta, "-m", desc)

    if not args.dry_run:
        print("\nListo. Publicar a mano: `git push origin <rama> --tags` y `dvc push`.")


if __name__ == "__main__":
    main()
