import numpy as np
import pandas as pd
import pytest

from drift.data_drift import detect_data_drift, has_drift, psi
from drift.inject import inject_data_drift
from train.data import DATA_DIR


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(DATA_DIR / "train.csv")


@pytest.fixture(scope="module")
def prod():
    return pd.read_csv(DATA_DIR / "produccion.csv")


def _drifted_features(rep):
    return set(rep.loc[rep["drift"], "feature"])


def test_verde_con_datos_limpios(train, prod):
    rep = detect_data_drift(train, prod.sample(2000, random_state=0))
    assert not has_drift(rep), rep[rep["drift"]]


@pytest.mark.parametrize("seed", range(10))
def test_sin_falsas_alarmas_en_varias_muestras(train, prod, seed):
    rep = detect_data_drift(train, prod.sample(1000, random_state=seed))
    assert not has_drift(rep), rep[rep["drift"]]


def test_rojo_con_datos_derivados(train, prod):
    cur = inject_data_drift(prod.sample(2000, random_state=0), seed=0)
    rep = detect_data_drift(train, cur)
    assert has_drift(rep)
    # cada tipo de prueba detecta su variable alterada
    assert {"age", "hours-per-week"} <= _drifted_features(rep)    # KS
    assert "capital-gain" in _drifted_features(rep)                # PSI
    assert {"workclass", "sex"} <= _drifted_features(rep)          # chi-cuadrado


def test_no_marca_variables_sin_tocar(train, prod):
    cur = inject_data_drift(prod.sample(2000, random_state=0), seed=0)
    rep = detect_data_drift(train, cur)
    sin_tocar = {"fnlwgt", "education-num", "capital-loss", "education", "marital-status",
                 "occupation", "relationship", "race", "native-country"}
    assert not (_drifted_features(rep) & sin_tocar)


def test_psi_identico_es_cero_y_cambio_grande_lo_supera(train):
    cg = train["capital-gain"]
    assert psi(cg, cg) == pytest.approx(0.0, abs=1e-9)
    assert psi(cg, cg * 3 + (np.arange(len(cg)) % 5 == 0) * 2000) >= 0.2


def test_inject_no_modifica_el_original(prod):
    antes = prod.copy()
    inject_data_drift(prod, seed=0)
    pd.testing.assert_frame_equal(prod, antes)
