"""Monitor de drift: veredictos por periodo, guardado de resultados y lectura del historial."""
import json

import numpy as np
import pytest

from drift import monitor
from drift.versionar_dump import lineas_hasta
from train.data import DATA_DIR, cargar_dump

pytestmark = pytest.mark.skipif(not (DATA_DIR / "datos.csv").exists(), reason="falta data/datos.csv (dvc pull)")


def predictor_falso(df):
    """Regla fija (sin API ni MLflow): >50K si tiene estudios altos o capital. Suficiente para la lógica."""
    pred = ((df["education-num"] >= 13) | (df["capital-gain"] > 5000)).astype(int).to_numpy()
    return pred, {"version": "test", "run_id": "x", "nombre": "falso", "fuente": "prueba"}


def correr(tmp_path, hasta, **kw):
    return monitor.ejecutar(hasta, predictor_falso, directorio=tmp_path, mlflow_=False, **kw)


def test_hasta_periodo_2_esta_en_verde(tmp_path):
    r = correr(tmp_path, 2)
    assert [x["lote"] for x in r["lotes"]] == [1, 2]
    assert not r["veredicto"]["data_drift"] and r["veredicto"]["alerta"] is False


def test_hasta_periodo_4_detecta_el_data_drift_de_los_lotes_3_y_4(tmp_path):
    r = correr(tmp_path, 4)
    v = r["veredicto"]
    assert v["data_drift"] and v["lotes_con_data_drift"] == [3, 4] and v["alerta"]
    assert all(x["variables_en_rojo"] >= 3 for x in r["lotes"] if x["lote"] in (3, 4))
    assert all(x["variables_en_rojo"] == 0 for x in r["lotes"] if x["lote"] in (1, 2))


def test_el_periodo_0_no_tiene_lotes(tmp_path):
    r = correr(tmp_path, 0)
    assert r["lotes"] == [] and r["veredicto"]["alerta"] is False


def test_periodo_fuera_de_rango_se_rechaza(tmp_path):
    for malo in (-1, 13):
        with pytest.raises(ValueError):
            correr(tmp_path, malo)


def test_el_hash_de_un_corte_es_el_del_prefijo_del_archivo(tmp_path):
    """Para que coincida con el hash DVC de la etiqueta de datos de ese momento."""
    import hashlib
    ruta = DATA_DIR / "datos.csv"
    texto = ruta.read_text(encoding="utf-8")
    periodos = cargar_dump(ruta)["periodo"]
    esperado = hashlib.md5(lineas_hasta(texto, periodos, 4).encode("utf-8")).hexdigest()
    assert correr(tmp_path, 4)["dump"]["md5"] == esperado


def test_guarda_json_y_se_puede_leer(tmp_path):
    r = correr(tmp_path, 4)
    archivos = list(tmp_path.glob("*-p4.json"))
    assert len(archivos) == 1 and r["archivo"] == archivos[0].name
    assert json.loads(archivos[0].read_text(encoding="utf-8"))["veredicto"] == r["veredicto"]
    ult = monitor.ultimo(tmp_path)
    assert ult["archivo"] == r["archivo"] and ult["dump"]["hasta_periodo"] == 4
    h = monitor.historial(tmp_path)
    assert len(h) == 1 and h[0]["hasta_periodo"] == 4 and h[0]["lotes_con_data_drift"] == [3, 4]


def test_no_guardar_no_escribe(tmp_path):
    correr(tmp_path / "nada", 2, guardar_resultado=False)
    assert not (tmp_path / "nada").exists()


def test_sin_resultados_ultimo_es_none_e_historial_vacio(tmp_path):
    assert monitor.ultimo(tmp_path / "x") is None and monitor.historial(tmp_path / "x") == []


def test_alerta_de_concept_drift_con_f1_que_cae(tmp_path):
    """Si el predictor empeora a partir del lote 7, la alerta se confirma tras 3 lotes bajo el umbral."""
    def degradante(df):
        pred, info = predictor_falso(df)
        if "periodo" in df.columns and df["periodo"].iloc[0] >= 7:
            pred = np.zeros_like(pred)
        return pred, info
    r = monitor.ejecutar(12, degradante, directorio=tmp_path, mlflow_=False)
    assert r["veredicto"]["concept_drift"] and r["veredicto"]["lote_alerta"] == 9
