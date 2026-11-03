"""Lógica de la interfaz (sin levantar Streamlit) y un arranque de la app sin API."""
import math
from pathlib import Path

import pytest

from ui.perfiles import (DESCONOCIDO, EDUCACION_NUM, NUMERICAS, OPCIONES, PERFILES, a_registro, fila_a_valores,
                         perfil_completo, prioridad)

CAMPOS_API = {"age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "workclass",
              "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"}


@pytest.mark.parametrize("nombre", list(PERFILES))
def test_perfiles_son_validos(nombre):
    valores = perfil_completo(nombre)
    for campo, opciones in OPCIONES.items():
        assert valores[campo] in opciones, f"{nombre}: {campo}={valores[campo]!r} no es una opción"
    for campo, (lo, hi, _) in NUMERICAS.items():
        assert lo <= valores[campo] <= hi
    assert set(a_registro(valores)) == CAMPOS_API


def test_education_num_sale_de_education():
    reg = a_registro(perfil_completo("Ejecutivo"))   # Masters
    assert reg["education-num"] == EDUCACION_NUM["Masters"] == 14.0


def test_desconocido_se_envia_como_null():
    valores = {**perfil_completo("Oficio estable"), "occupation": DESCONOCIDO}
    assert a_registro(valores)["occupation"] is None


def test_fila_real_con_faltantes_y_valores_fuera_de_rango():
    fila = {**perfil_completo("Oficio estable"), "workclass": float("nan"), "age": 120, "native-country": "Atlantis"}
    v = fila_a_valores(fila)
    assert v["workclass"] == DESCONOCIDO
    assert v["age"] == NUMERICAS["age"][1]
    assert v["native-country"] in OPCIONES["native-country"]
    assert not any(isinstance(x, float) and math.isnan(x) for x in v.values())


@pytest.mark.parametrize("p, esperado", [(0.95, "Alta"), (0.70, "Alta"), (0.69, "Media"), (0.40, "Media"),
                                         (0.39, "Baja"), (0.0, "Baja")])
def test_prioridad(p, esperado):
    assert prioridad(p) == esperado


def test_la_app_arranca_sin_api(monkeypatch):
    """Sin API la interfaz debe avisar, no romperse."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")   # puerto cerrado
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "ui" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Sin conexión" in e.value for e in at.sidebar.error)
