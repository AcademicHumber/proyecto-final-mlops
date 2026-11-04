"""Dump temporal de la "base de datos": un solo archivo que crece con el tiempo.

Modelo (ver docs/datos.md): `data/datos.csv` representa el dump de la base de datos.
  - periodo 0 : la base con la que se entrena (train + test, con la asignación original en `particion`).
  - periodo 1-12 : lotes de llegadas, en orden temporal (`particion` = "llegada").
Línea de tiempo (cada lote es un periodo):
  - lotes 1-2  : limpios (sin drift);
  - lotes 3-4  : data drift (cambia la población; las etiquetas no cambian);
  - lotes 5-12 : concept drift con intensidad creciente (cambia P(y|X); las X no cambian).

Es determinista: las mismas entradas dan los mismos bytes. Sale de los CSV de la versión v1 de los datos
(train.csv, test.csv y produccion.csv), con las mismas semillas que ya usaban los escenarios anteriores,
de modo que el modelo v2 (entrenado con la partición `train`) sigue siendo válido.

    python -m drift.escenarios --salida data/datos.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from drift.concept_drift import make_stream
from drift.inject import inject_data_drift
from train.data import DATA_DIR, FEATURES, TARGET, load_adult, split_data

N_LOTES = 12
INICIO_CONCEPT_DRIFT = 4          # índice base 0: el lote 5 es el primero con concept drift
LOTES_DATA_DRIFT = (3, 4)         # lotes (base 1) con data drift
SEED = 0
COLUMNAS = FEATURES + [TARGET, "periodo", "particion"]


def construir_dump(train: pd.DataFrame, test: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    """Une la base (periodo 0) y las llegadas (periodos 1 a 12) en un único DataFrame."""
    base = pd.concat([train.assign(particion="train"), test.assign(particion="test")], ignore_index=True)
    base["periodo"] = 0

    lotes = make_stream(prod, n_batches=N_LOTES, drift_start=INICIO_CONCEPT_DRIFT, seed=SEED)

    # Data drift sobre los lotes indicados, aplicado una sola vez al bloque y repartido de nuevo.
    idx = [n - 1 for n in LOTES_DATA_DRIFT]
    tamanos = [len(lotes[i]) for i in idx]
    bloque = inject_data_drift(pd.concat([lotes[i] for i in idx], ignore_index=True), seed=SEED)
    inicio = 0
    for i, n in zip(idx, tamanos):
        lotes[i] = bloque.iloc[inicio:inicio + n]
        inicio += n

    llegadas = pd.concat([b.assign(periodo=n + 1, particion="llegada") for n, b in enumerate(lotes)],
                         ignore_index=True)
    return pd.concat([base, llegadas], ignore_index=True)[COLUMNAS]


def construir_desde_disco(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    leer = lambda n: pd.read_csv(data_dir / f"{n}.csv")   # noqa: E731
    return construir_dump(leer("train"), leer("test"), leer("produccion"))


def construir_desde_openml() -> pd.DataFrame:
    """Regenera el dump desde cero (descarga Adult de OpenML y reparte 60/20/20 con la semilla del proyecto).
    Es lo que usa la integración continua, donde no hay almacén de DVC."""
    return construir_dump(*split_data(load_adult()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salida", default=str(DATA_DIR / "datos.csv"))
    ap.add_argument("--desde-openml", action="store_true",
                    help="descarga Adult de OpenML en vez de leer los CSV de la v1 (para CI)")
    args = ap.parse_args()
    dump = construir_desde_openml() if args.desde_openml else construir_desde_disco()
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    dump.to_csv(args.salida, index=False, lineterminator="\n")   # mismos bytes en Windows y Linux
    por_periodo = dump.groupby("periodo").size()
    print(f"{args.salida}: {len(dump)} filas, {dump.shape[1]} columnas")
    print(f"periodo 0: {por_periodo[0]} filas; periodos 1-{N_LOTES}: {por_periodo[1]} a {por_periodo[N_LOTES]} filas")
