import pandas as pd
import pytest

from drift.concept_drift import (batch_f1s, f1_of, first_alert, load_production_model, make_stream,
                                 retrain_threshold)
from drift.inject import inject_concept_drift
from train.data import DATA_DIR, FEATURES, TARGET, cargar_dump, llegadas, particion

DATOS = DATA_DIR / "datos.csv"


# --- Lógica pura (no necesita modelo ni MLflow) ---------------------------------------------

def test_alerta_exige_racha_sostenida():
    thr = 0.64
    assert first_alert([0.70, 0.60, 0.70, 0.60, 0.70, 0.60], thr) is None      # dips aislados
    assert first_alert([0.70, 0.60, 0.60, 0.70, 0.60, 0.60, 0.60], thr) == 6   # 3 seguidos


def test_umbral_es_diez_por_ciento_bajo_la_base():
    assert retrain_threshold(0.70) == pytest.approx(0.63)


def _lote_limpio():
    """Lotes 1-2 del dump (sin drift): materia prima para probar la inyección."""
    return llegadas(cargar_dump(DATOS), 1, 2).drop(columns="periodo")


@pytest.mark.skipif(not DATOS.exists(), reason="falta data/datos.csv (dvc pull)")
def test_make_stream_parte_en_lotes_consecutivos():
    prod = _lote_limpio()
    lotes = make_stream(prod, n_batches=4)
    assert len(lotes) == 4 and sum(len(b) for b in lotes) == len(prod)
    assert pd.concat(lotes).index.is_monotonic_increasing


@pytest.mark.skipif(not DATOS.exists(), reason="falta data/datos.csv (dvc pull)")
def test_concept_drift_cambia_etiquetas_y_no_las_x():
    prod = _lote_limpio()
    out = inject_concept_drift(prod, strength=1.0, seed=0)
    pd.testing.assert_frame_equal(out[FEATURES], prod[FEATURES])
    assert (out[TARGET] != prod[TARGET]).any()
    pd.testing.assert_frame_equal(inject_concept_drift(prod, 0.0), prod)


# --- Con el modelo en producción (requiere MLflow accesible, p. ej. port-forward a :5000) ---

@pytest.fixture(scope="module")
def model():
    try:
        return load_production_model()
    except Exception as e:  # MLflow no accesible o sin modelo registrado
        pytest.skip(f"No se pudo cargar el modelo de producción: {e}")


@pytest.fixture(scope="module")
def dump():
    if not DATOS.exists():
        pytest.skip("falta data/datos.csv (dvc pull)")
    return cargar_dump(DATOS)


@pytest.fixture(scope="module")
def f1_base(model, dump):
    return f1_of(model, particion(dump, "test"))


def test_verde_sin_concept_drift(model, f1_base, dump):
    """Hasta el periodo 4 (lotes 1-4, incluido el data drift) el F1 no da alerta."""
    f1s = batch_f1s(model, [llegadas(dump, n, n) for n in range(1, 5)])
    assert first_alert(f1s, retrain_threshold(f1_base)) is None


def test_rojo_con_concept_drift(model, f1_base, dump):
    f1s = batch_f1s(model, [llegadas(dump, n, n) for n in range(1, 13)])
    assert first_alert(f1s, retrain_threshold(f1_base)) is not None
    assert f1s[-1] < 0.5 * f1s[0]                       # degradación clara
    assert max(f1s[:4]) > retrain_threshold(f1_base)    # antes del drift el modelo estaba sano
