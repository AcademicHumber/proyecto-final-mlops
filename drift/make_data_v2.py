"""Genera la versión 2 de los datos: producción con drift aplicado, a partir de la v1 versionada con DVC.

Entrada:  data/produccion.csv (v1, sin drift)
Salida:
  - data/produccion_concept_drift.csv: los 12 lotes de producción en orden temporal, con la columna
    `lote` (1..12). Los lotes 1-4 están limpios; desde el 5 cambia P(y|X) con intensidad creciente.
  - data/produccion_data_drift.csv: toda producción con las X alteradas (las etiquetas no cambian).

Es determinista (semillas fijas): regenerar da los mismos bytes. Después de ejecutarlo:
    dvc add data/produccion_concept_drift.csv data/produccion_data_drift.csv
"""
import pandas as pd

from drift.concept_drift import make_stream
from drift.inject import inject_data_drift
from train.data import DATA_DIR

N_BATCHES = 12
DRIFT_START = 4     # índice base 0: el lote 5 es el primero con drift
SEED = 0


def build_v2(prod: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lotes = make_stream(prod, n_batches=N_BATCHES, drift_start=DRIFT_START, seed=SEED)
    concept = pd.concat([b.assign(lote=i + 1) for i, b in enumerate(lotes)], ignore_index=True)
    data = inject_data_drift(prod, seed=SEED).reset_index(drop=True)
    return {"produccion_concept_drift.csv": concept, "produccion_data_drift.csv": data}


if __name__ == "__main__":
    prod = pd.read_csv(DATA_DIR / "produccion.csv")
    for nombre, df in build_v2(prod).items():
        df.to_csv(DATA_DIR / nombre, index=False)
        print(f"{nombre}: {len(df)} filas, {df.shape[1]} columnas")
