"""Detección de data drift: compara una muestra nueva contra el baseline (train).

Pruebas por tipo de variable:
  - numéricas: Kolmogorov-Smirnov de dos muestras (p-valor);
  - capital-gain y capital-loss: PSI (tienen ~90 % de ceros y KS se comporta mal con ese empate);
  - categóricas: chi-cuadrado sobre la tabla de frecuencias baseline vs nueva.

Umbrales (explícitos):
  - KS y chi-cuadrado: p < ALPHA / n_pruebas (corrección de Bonferroni para no dar falsas alarmas
    al probar muchas variables a la vez).
  - PSI >= 0.2 (convención: <0.1 estable, 0.1-0.2 vigilar, >=0.2 cambio relevante).
Hay drift del conjunto si al menos una variable lo tiene.
"""
import numpy as np
import pandas as pd
from scipy import stats

from train.data import CATEGORICAL, NUMERIC

ALPHA = 0.01
PSI_THRESHOLD = 0.2
PSI_FEATURES = ["capital-gain", "capital-loss"]
KS_FEATURES = [c for c in NUMERIC if c not in PSI_FEATURES]
RARE_SHARE = 0.005   # categorías con menos de 0.5 % del baseline se agrupan en "other"


def psi(ref: pd.Series, cur: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index. Bin propio para el 0 y deciles del baseline para el resto."""
    ref, cur = ref.dropna().to_numpy(), cur.dropna().to_numpy()
    pos = ref[ref > 0]
    edges = np.unique(np.quantile(pos, np.linspace(0, 1, n_bins + 1))) if len(pos) else np.array([0.0, 1.0])
    edges[0], edges[-1] = 0.0, np.inf

    def shares(x):
        counts = np.append((x == 0).sum(), np.histogram(x[x > 0], bins=edges)[0]).astype(float)
        return np.clip(counts / counts.sum(), 1e-4, None)   # evita log(0)

    p, q = shares(ref), shares(cur)
    return float(np.sum((q - p) * np.log(q / p)))


def _chi2(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    ref = ref.fillna("missing").astype(str)
    cur = cur.fillna("missing").astype(str)
    share = ref.value_counts(normalize=True)
    keep = set(share[share >= RARE_SHARE].index)
    ref = ref.where(ref.isin(keep), "other")
    cur = cur.where(cur.isin(keep), "other")
    table = pd.concat([ref.value_counts(), cur.value_counts()], axis=1).fillna(0)
    chi2, p, _, _ = stats.chi2_contingency(table.to_numpy())
    return float(chi2), float(p)


def detect_data_drift(ref: pd.DataFrame, cur: pd.DataFrame, alpha: float = ALPHA,
                      psi_threshold: float = PSI_THRESHOLD) -> pd.DataFrame:
    """Un renglón por variable: prueba, estadístico, p-valor/PSI, umbral y si hay drift."""
    n_tests = len(KS_FEATURES) + len(CATEGORICAL)
    p_limit = alpha / n_tests
    rows = []
    for col in KS_FEATURES:
        res = stats.ks_2samp(ref[col].dropna(), cur[col].dropna())
        rows.append((col, "KS", res.statistic, res.pvalue, p_limit, res.pvalue < p_limit))
    for col in PSI_FEATURES:
        v = psi(ref[col], cur[col])
        rows.append((col, "PSI", v, np.nan, psi_threshold, v >= psi_threshold))
    for col in CATEGORICAL:
        chi2, p = _chi2(ref[col], cur[col])
        rows.append((col, "chi2", chi2, p, p_limit, p < p_limit))
    return pd.DataFrame(rows, columns=["feature", "test", "statistic", "p_value", "threshold", "drift"])


def has_drift(report: pd.DataFrame) -> bool:
    return bool(report["drift"].any())


if __name__ == "__main__":
    from drift.inject import inject_data_drift
    from train.data import DATA_DIR

    train = pd.read_csv(DATA_DIR / "train.csv")
    prod = pd.read_csv(DATA_DIR / "produccion.csv").sample(2000, random_state=0)
    pd.set_option("display.width", 140)
    for nombre, cur in [("DATOS LIMPIOS", prod), ("DATOS CON DRIFT", inject_data_drift(prod, seed=0))]:
        rep = detect_data_drift(train, cur)
        print(f"\n== {nombre}: {'DRIFT' if has_drift(rep) else 'sin drift'} "
              f"({int(rep['drift'].sum())}/{len(rep)} variables) ==")
        print(rep.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
