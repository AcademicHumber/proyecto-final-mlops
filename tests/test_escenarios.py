"""El dump temporal cumple la línea de tiempo (docs/datos.md) y reproduce los CSV de la v1.

Las pruebas marcadas `anclaje` comparan con los CSV antiguos (etiqueta `data-v1`/`data-v2`): corren solo si esos
archivos están en `data/` (en local, tras `git switch data-v2 && dvc checkout`); en CI se omiten.
"""
import pandas as pd
import pytest

from drift.data_drift import detect_data_drift, has_drift
from drift.escenarios import COLUMNAS, LOTES_DATA_DRIFT, N_LOTES, construir_desde_disco, construir_dump
from train.data import DATA_DIR, FEATURES, TARGET, cargar_dump, llegadas, particion

V1 = FEATURES + [TARGET]
ANTIGUOS = ("train", "test", "produccion", "produccion_concept_drift")
anclaje = pytest.mark.skipif(not all((DATA_DIR / f"{n}.csv").exists() for n in ANTIGUOS),
                             reason="faltan los CSV antiguos (data-v2)")


@pytest.fixture(scope="module")
def csv():
    return {n: pd.read_csv(DATA_DIR / f"{n}.csv") for n in ANTIGUOS}


@pytest.fixture(scope="module")
def dump():
    if not (DATA_DIR / "datos.csv").exists():
        pytest.skip("falta data/datos.csv (dvc pull)")
    return cargar_dump()


def test_estructura(dump):
    assert list(dump.columns) == COLUMNAS
    assert sorted(dump["periodo"].unique()) == list(range(N_LOTES + 1))
    assert set(dump.loc[dump["periodo"] == 0, "particion"]) == {"train", "test"}
    assert set(dump.loc[dump["periodo"] > 0, "particion"]) == {"llegada"}
    assert dump["periodo"].is_monotonic_increasing                      # cada momento es un prefijo del archivo
    assert dump.loc[dump["periodo"] > 0, "periodo"].value_counts().between(800, 820).all()   # lotes de ~814


@anclaje
def test_tamanos_coinciden_con_los_csv_antiguos(dump, csv):
    assert (dump["periodo"] == 0).sum() == len(csv["train"]) + len(csv["test"])
    assert (dump["periodo"] > 0).sum() == len(csv["produccion"])


@anclaje
def test_la_base_reproduce_train_y_test_exactamente(dump, csv):
    """El modelo v2 se entrenó con train.csv: sus filas deben seguir intactas y en el mismo orden."""
    pd.testing.assert_frame_equal(particion(dump, "train"), csv["train"])
    pd.testing.assert_frame_equal(particion(dump, "test"), csv["test"])


@anclaje
def test_lotes_sin_data_drift_reproducen_el_concept_drift_anterior(dump, csv):
    """Los lotes 1-2 y 5-12 son idénticos a produccion_concept_drift.csv: los resultados documentados siguen valiendo."""
    antiguo = csv["produccion_concept_drift"]
    for lote in [n for n in range(1, N_LOTES + 1) if n not in LOTES_DATA_DRIFT]:
        nuevo = llegadas(dump, lote, lote).drop(columns="periodo")
        viejo = antiguo[antiguo["lote"] == lote][V1].reset_index(drop=True)
        pd.testing.assert_frame_equal(nuevo, viejo, obj=f"lote {lote}")


@anclaje
def test_lotes_con_data_drift_cambian_las_x_y_no_las_etiquetas(dump, csv):
    antiguo = csv["produccion_concept_drift"]
    for lote in LOTES_DATA_DRIFT:
        nuevo = llegadas(dump, lote, lote)
        viejo = antiguo[antiguo["lote"] == lote].reset_index(drop=True)
        pd.testing.assert_series_equal(nuevo[TARGET], viejo[TARGET])      # etiquetas intactas
        assert nuevo["age"].mean() > viejo["age"].mean() + 5               # población más mayor


@anclaje
def test_es_determinista(dump, csv):
    otro = construir_dump(csv["train"], csv["test"], csv["produccion"])
    pd.testing.assert_frame_equal(dump, otro)


def test_hasta_periodo_reconstruye_cada_momento(dump, tmp_path):
    ruta = tmp_path / "datos.csv"
    dump.to_csv(ruta, index=False)
    t0, t2, t12 = (cargar_dump(ruta, hasta_periodo=n) for n in (0, 2, 12))
    assert set(t0["periodo"]) == {0} and set(t2["periodo"]) == {0, 1, 2}
    assert len(t12) == len(dump) and len(t0) < len(t2) < len(t12)
    # cada momento es un prefijo del siguiente: el dump solo crece
    pd.testing.assert_frame_equal(t2, t12.iloc[:len(t2)].reset_index(drop=True))


def test_semaforo_de_data_drift_por_lote(dump):
    """Con el baseline de train: limpios en verde y con data drift en rojo (como pide el apartado 6.1)."""
    base = particion(dump, "train")
    sanos = llegadas(dump, 1, 2)
    con_drift = llegadas(dump, 3, 4)
    assert not has_drift(detect_data_drift(base, sanos[FEATURES]))
    rep = detect_data_drift(base, con_drift[FEATURES])
    assert has_drift(rep) and rep["drift"].sum() >= 3, rep[rep["drift"]]
    # el concept drift no toca las X: los lotes 5-12 no deben dar data drift
    assert not has_drift(detect_data_drift(base, llegadas(dump, 5, 9)[FEATURES]))


@anclaje
def test_datos_csv_versionado_es_el_dump_generado(dump):
    """data/datos.csv no es opaco: es exactamente lo que genera drift.escenarios desde los CSV de la v1."""
    pd.testing.assert_frame_equal(dump, construir_desde_disco())
