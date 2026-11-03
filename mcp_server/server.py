"""Servidor MCP (stdio): predicción contra la API de Kubernetes y consultas de solo lectura al dataset.

Ejecutar desde la raíz del repositorio:  python -m mcp_server.server
Variables: API_URL (por defecto http://localhost:8080), DATA_DIR (por defecto ./data).
"""
from typing import Literal, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import datos, modelo
from ui.perfiles import OPCIONES, VALORES_BASE

mcp = MCPServer("adult-income")


def _seguro(fn, *a, **kw):
    """Los errores esperados (SQL no permitido, API caída, valores inválidos) llegan al cliente con su mensaje."""
    try:
        return fn(*a, **kw)
    except (datos.ConsultaInvalida, modelo.ApiNoDisponible, ValueError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e

Workclass = Literal[tuple(OPCIONES["workclass"])]
Education = Literal[tuple(OPCIONES["education"])]
Marital = Literal[tuple(OPCIONES["marital-status"])]
Occupation = Literal[tuple(OPCIONES["occupation"])]
Relationship = Literal[tuple(OPCIONES["relationship"])]
Race = Literal[tuple(OPCIONES["race"])]
Sex = Literal[tuple(OPCIONES["sex"])]
Country = Literal[tuple(OPCIONES["native-country"])]


@mcp.tool()
def predecir_prospecto(age: int, education: Education, hours_per_week: int = 40,
                       workclass: Optional[Workclass] = "Private",
                       marital_status: Marital = "Married-civ-spouse",
                       occupation: Optional[Occupation] = None,
                       relationship: Relationship = "Not-in-family",
                       sex: Sex = "Male", race: Race = "White",
                       native_country: Optional[Country] = "United-States",
                       capital_gain: int = 0, capital_loss: int = 0, fnlwgt: int = VALORES_BASE["fnlwgt"]) -> dict:
    """Estima la probabilidad de que un prospecto gane más de 50K con el modelo en producción
    (alias `production`, servido desde Kubernetes). Devuelve probabilidad, prioridad comercial
    Alta/Media/Baja, versión del modelo y pod que respondió. Es una guía para priorizar, no una
    decisión automática. Las variables sensibles (sex, race, native_country) las usa el modelo."""
    p = {"age": age, "education": education, "hours-per-week": hours_per_week, "workclass": workclass,
         "marital-status": marital_status, "occupation": occupation, "relationship": relationship,
         "sex": sex, "race": race, "native-country": native_country, "capital-gain": capital_gain,
         "capital-loss": capital_loss, "fnlwgt": fnlwgt}
    r = _seguro(modelo.predecir, [p])
    return {**r["resultados"][0], "modelo": r["modelo"], "version": r["version"], "pod": r["pod"]}


@mcp.tool()
def priorizar_campana(prospectos: list[dict], porcentaje_presupuesto: float = 10.0) -> dict:
    """Puntúa hasta 500 prospectos y devuelve a quién contactar según el presupuesto (% mejor
    puntuado, 10 por defecto). Cada prospecto usa las claves del formulario: age, education,
    hours-per-week, workclass, marital-status, occupation, relationship, sex, race, native-country,
    capital-gain, capital-loss, fnlwgt (las que falten toman valores base)."""
    if not 0 < porcentaje_presupuesto <= 100:
        raise ToolError("porcentaje_presupuesto debe estar entre 0 y 100.")
    completos = [{**VALORES_BASE, **p} for p in prospectos]
    r = _seguro(modelo.predecir, completos)
    orden = sorted(range(len(completos)), key=lambda i: -r["resultados"][i]["probabilidad_mayor_50k"])
    n = max(1, round(len(orden) * porcentaje_presupuesto / 100))
    ranking = [{"posicion": k + 1, "indice_original": i, "contactar": k < n,
                **{c: r["resultados"][i][c] for c in ("probabilidad_mayor_50k", "prioridad")}}
               for k, i in enumerate(orden)]
    return {"contactar": n, "total": len(orden), "ranking": ranking,
            "version": r["version"], "pod": r["pod"]}


@mcp.tool()
def info_modelo() -> dict:
    """Modelo desplegado en el clúster: nombre, alias, versión, run_id y pod que respondió."""
    return _seguro(modelo.info)


@mcp.tool()
def consultar_datos(sql: str, max_filas: int = 50) -> dict:
    """Ejecuta una consulta SQL SELECT (SQLite, solo lectura, una sentencia, máximo 200 filas) sobre
    las tablas train, test, produccion, produccion_data_drift y produccion_concept_drift.
    income: 1 = >50K. Usa `describir_datos` para ver columnas. Las tablas *_drift son simuladas."""
    return _seguro(datos.consultar, sql, max_filas)


@mcp.tool()
def describir_datos() -> dict:
    """Tablas disponibles con descripción, número de filas, columnas y proporción de >50K."""
    return _seguro(datos.describir)


@mcp.resource("datos://esquema")
def esquema() -> dict:
    """Esquema de todas las tablas consultables."""
    return _seguro(datos.describir)


@mcp.resource("datos://{tabla}/muestra")
def muestra(tabla: str) -> dict:
    """Cinco filas de ejemplo de una tabla."""
    if tabla not in datos.TABLAS:
        raise ValueError(f"Tabla desconocida: {tabla}. Opciones: {', '.join(datos.TABLAS)}")
    return datos.consultar(f"SELECT * FROM {tabla} LIMIT 5")


@mcp.prompt()
def priorizar_prospectos(objetivo: str = "campaña del producto premium") -> str:
    """Guía para priorizar prospectos con el modelo y los datos."""
    return (f"Objetivo: {objetivo}. 1) Llama a describir_datos y a info_modelo. 2) Explora con "
            "consultar_datos los perfiles relevantes (income = 1 es >50K). 3) Evalúa candidatos con "
            "predecir_prospecto o priorizar_campana. 4) Resume a quién contactar y recuerda que la "
            "prioridad es una guía, no una decisión automática, y que el modelo usa variables "
            "sensibles (sex, race, native-country). 5) Indica versión del modelo y pod.")


if __name__ == "__main__":
    mcp.run("stdio")
