"""Envia N peticiones a /predict y cuenta que pod responde cada una.

Uso: python k8s/probe.py URL [N]
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]


def main(url: str, n: int) -> None:
    row = pd.read_csv(ROOT / "data" / "test.csv").head(1)[FEATURES].to_dict("records")[0]
    row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    pods, errores = Counter(), 0
    for _ in range(n):
        try:
            r = requests.post(f"{url}/predict", json=[row], timeout=5)
            r.raise_for_status()
            pods[r.json()["pod"]] += 1
        except requests.RequestException:
            errores += 1
    print(f"{n} peticiones, {errores} errores")
    for pod, c in sorted(pods.items()):
        print(f"  {pod}: {c}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
