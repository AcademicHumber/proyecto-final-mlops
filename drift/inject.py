"""Inyección controlada de drift sobre datos de producción simulada.

- Data drift: cambia la distribución de las variables X (las etiquetas no se tocan).
- Concept drift: cambia la relación X -> y (las X no se tocan, se modifican las etiquetas).

Todas las funciones devuelven una copia y son reproducibles con `seed`.
"""
import numpy as np
import pandas as pd

TARGET = "income"


def shift_numeric(df: pd.DataFrame, col: str, add: float = 0.0, mult: float = 1.0,
                  lower: float | None = None) -> pd.DataFrame:
    """col -> col * mult + add (con límite inferior opcional)."""
    out = df.copy()
    out[col] = out[col] * mult + add
    if lower is not None:
        out[col] = out[col].clip(lower=lower)
    return out


def shift_categorical(df: pd.DataFrame, col: str, value: str, frac: float,
                      seed: int = 0) -> pd.DataFrame:
    """Reemplaza por `value` una fracción `frac` de las filas de `col`."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    out.loc[rng.random(len(out)) < frac, col] = value
    return out


def inject_data_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Escenario de data drift: población más mayor, con menos horas, más ganancias de
    capital y más autónomos y mujeres que en el entrenamiento."""
    rng = np.random.default_rng(seed)
    out = shift_numeric(df, "age", add=8, lower=17)
    out = shift_numeric(out, "hours-per-week", add=-6, lower=1)
    # capital-gain: los que ya tenían ganancias ganan más, y algunos de los que no tenían empiezan a tener
    nonzero = out["capital-gain"] > 0
    out.loc[nonzero, "capital-gain"] *= 2.5
    nuevos = (~nonzero) & (rng.random(len(out)) < 0.15)
    out.loc[nuevos, "capital-gain"] = rng.choice(df.loc[df["capital-gain"] > 0, "capital-gain"].to_numpy(),
                                                  size=int(nuevos.sum()))
    out = shift_categorical(out, "workclass", "Self-emp-not-inc", 0.30, seed)
    out = shift_categorical(out, "sex", "Female", 0.30, seed + 1)
    return out


def inject_concept_drift(df: pd.DataFrame, strength: float, seed: int = 0) -> pd.DataFrame:
    """Cambia P(y | X) sin tocar X. `strength` en [0, 1] es la probabilidad de voltear la etiqueta
    en las dos reglas que cambian:
      - con estudios superiores (education-num >= 13) el ingreso >50K deja de darse (>50K -> <=50K);
      - jóvenes (< 30 años) con jornada larga (>= 45 h) pasan a ganar >50K (<=50K -> >50K).
    """
    out = df.copy()
    if strength <= 0:
        return out
    rng = np.random.default_rng(seed)
    flip = rng.random(len(out)) < strength
    r1 = (out["education-num"] >= 13) & (out[TARGET] == 1)
    r2 = (out["age"] < 30) & (out["hours-per-week"] >= 45) & (out[TARGET] == 0)
    out.loc[r1 & flip, TARGET] = 0
    out.loc[r2 & flip, TARGET] = 1
    return out
