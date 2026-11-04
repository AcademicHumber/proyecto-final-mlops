"""Servidor MCP: guardas del SQL, predicción con la API simulada y una prueba extremo a extremo en memoria."""
import asyncio

import pytest

from mcp_server import datos, modelo
from mcp_server.datos import ConsultaInvalida

pytestmark = pytest.mark.skipif(not (datos.DATA_DIR / datos.ARCHIVO).exists(), reason="faltan los CSV (dvc pull)")


def test_select_basico():
    r = datos.consultar("SELECT education, COUNT(*) n FROM train GROUP BY education ORDER BY n DESC")
    assert r["columnas"] == ["education", "n"] and r["filas"]


def test_limite_de_filas():
    r = datos.consultar("SELECT * FROM train", max_filas=7)
    assert len(r["filas"]) == 7 and r["truncado"]


@pytest.mark.parametrize("sql", [
    "DROP TABLE train", "DELETE FROM train", "SELECT 1; SELECT 2", "", "PRAGMA table_info(train)",
    "ATTACH DATABASE 'x.db' AS x", "SELECT load_extension('x')", "SELECT readfile('/etc/passwd')",
    "WITH a AS (SELECT 1) DELETE FROM train", "INSERT INTO train SELECT * FROM test", "DROP VIEW train",
])
def test_consultas_peligrosas_rechazadas(sql):
    with pytest.raises(ConsultaInvalida):
        datos.consultar(sql)


def test_describir_incluye_la_tabla_y_las_vistas():
    d = datos.describir()
    assert set(datos.TABLAS) <= set(d) and {"datos", "train", "test", "llegadas"} == set(datos.TABLAS)
    assert {"periodo", "particion"} <= set(d["datos"]["columnas"])
    assert "periodo" in d["llegadas"]["columnas"] and "periodo" not in d["train"]["columnas"]
    assert d["train"]["filas"] + d["test"]["filas"] + d["llegadas"]["filas"] == d["datos"]["filas"]


def _falsa_api(monkeypatch):
    def llamar(metodo, ruta, **kw):
        if ruta == "/info":
            return {"model_name": "m", "alias": "production", "model_version": "2", "model_run_id": "r", "pod": "p1"}
        probs = [0.9 if r["age"] > 40 else 0.1 for r in kw["json"]]
        return {"predictions": [{"prediction": ">50K" if p >= .5 else "<=50K", "probability_over_50k": p} for p in probs],
                "model_name": "m", "model_version": "2", "model_run_id": "r", "pod": "p1"}
    monkeypatch.setattr(modelo, "_llamar", llamar)


def test_predecir_con_prioridad(monkeypatch):
    _falsa_api(monkeypatch)
    from ui.perfiles import perfil_completo
    r = modelo.predecir([perfil_completo("Ejecutivo")])
    assert r["resultados"][0]["prioridad"] == "Alta" and r["pod"] == "p1" and r["version"] == "2"


def test_predecir_rechaza_valores_invalidos(monkeypatch):
    _falsa_api(monkeypatch)
    from ui.perfiles import perfil_completo
    with pytest.raises(ValueError):
        modelo.predecir([{**perfil_completo("Ejecutivo"), "sex": "X"}])
    with pytest.raises(ValueError):
        modelo.predecir([])


def test_extremo_a_extremo_con_cliente_mcp(monkeypatch):
    _falsa_api(monkeypatch)
    from mcp import Client
    from mcp_server.server import mcp

    async def main():
        async with Client(mcp) as c:
            nombres = {t.name for t in (await c.list_tools()).tools}
            assert {"predecir_prospecto", "priorizar_campana", "info_modelo", "consultar_datos",
                    "describir_datos"} <= nombres
            r = await c.call_tool("predecir_prospecto", {"age": 50, "education": "Masters"})
            assert not r.is_error and "Alta" in r.content[0].text
            r = await c.call_tool("consultar_datos", {"sql": "DROP TABLE train"})
            assert r.is_error and "SELECT" in r.content[0].text
            r = await c.call_tool("priorizar_campana", {"prospectos": [{"age": 20}, {"age": 60}], "porcentaje_presupuesto": 50})
            assert not r.is_error
    asyncio.run(main())


def test_herramientas_de_drift(monkeypatch, tmp_path):
    """ejecutar_drift escribe solo en resultados/drift (aquí un directorio temporal) y las de lectura lo devuelven."""
    from drift import monitor
    from tests.test_monitor import predictor_falso
    from mcp import Client
    from mcp_server.server import mcp

    monkeypatch.setattr(monitor, "RESULTADOS_DIR", tmp_path)
    monkeypatch.setattr(monitor, "predictor_api", lambda url=None: predictor_falso)
    monkeypatch.setattr(monitor, "_a_mlflow", lambda res, ruta: None)

    async def main():
        async with Client(mcp) as c:
            r = await c.call_tool("ultimo_resultado_drift", {})
            assert r.is_error and "ejecutar_drift" in r.content[0].text           # aún no hay nada
            r = await c.call_tool("ejecutar_drift", {"hasta_periodo": 4})
            assert not r.is_error and "lotes_con_data_drift" in r.content[0].text
            r = await c.call_tool("ejecutar_drift", {"hasta_periodo": 99})
            assert r.is_error
            r = await c.call_tool("ultimo_resultado_drift", {})
            assert not r.is_error and '"hasta_periodo": 4' in r.content[0].text
            r = await c.call_tool("historial_drift", {})
            assert not r.is_error
    asyncio.run(main())
    assert len(list(tmp_path.glob("*.json"))) == 1
