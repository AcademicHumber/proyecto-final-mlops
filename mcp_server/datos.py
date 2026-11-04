"""Capa de datos del servidor MCP: SQLite en memoria, solo lectura y con autorizador.

Los CSV de `data/` se cargan una vez en una base en memoria. Después se instala un autorizador que
solo permite SELECT y funciones de la lista blanca; así una consulta no puede escribir, adjuntar
archivos ni cargar extensiones. Se acepta una única sentencia y el resultado se limita en filas.
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MAX_FILAS = 200

ARCHIVO = "datos.csv"
VARS = ('age, workclass, fnlwgt, education, "education-num", "marital-status", occupation, relationship, race, '
        'sex, "capital-gain", "capital-loss", "hours-per-week", "native-country", income')

# Una tabla (`datos`, el dump completo) y tres vistas que la recortan.
TABLAS = {
    "datos": "Dump completo de la base: base (periodo 0) + llegadas (periodos 1-12). Columnas extra: periodo, particion.",
    "train": "Vista: filas de la base con particion='train' (baseline del drift y datos del modelo).",
    "test": "Vista: filas de la base con particion='test' (evaluación del modelo).",
    "llegadas": "Vista: filas que llegaron después de la base (periodo 1-12); data drift en los periodos 3-4 y concept drift desde el 5.",
}
VISTAS = {"train": f"SELECT {VARS} FROM datos WHERE particion = 'train'",
          "test": f"SELECT {VARS} FROM datos WHERE particion = 'test'",
          "llegadas": f"SELECT {VARS}, periodo FROM datos WHERE particion = 'llegada'"}

FUNCIONES_OK = {"count", "sum", "avg", "min", "max", "round", "abs", "coalesce", "ifnull", "nullif", "cast",
                "lower", "upper", "length", "substr", "trim", "total", "iif", "typeof", "group_concat",
                "printf", "like", "glob", "instr", "replace"}


class ConsultaInvalida(ValueError):
    pass


def _autorizador(accion, arg1, arg2, db, origen):
    if accion in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ):
        return sqlite3.SQLITE_OK
    if accion == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in FUNCIONES_OK else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def _cargar() -> sqlite3.Connection:
    ruta = DATA_DIR / ARCHIVO
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Ejecuta `dvc pull`.")
    con = sqlite3.connect(":memory:", check_same_thread=False)
    pd.read_csv(ruta).to_sql("datos", con, index=False)
    for nombre, sql in VISTAS.items():
        con.execute(f"CREATE VIEW {nombre} AS {sql}")
    con.set_authorizer(_autorizador)   # desde aquí solo se puede leer
    return con


_con: sqlite3.Connection | None = None


def conexion() -> sqlite3.Connection:
    global _con
    if _con is None:
        _con = _cargar()
    return _con


def consultar(sql: str, max_filas: int = MAX_FILAS) -> dict:
    """Ejecuta un SELECT de solo lectura. Devuelve columnas, filas y si se truncó."""
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        raise ConsultaInvalida("La consulta está vacía.")
    if ";" in sql:
        raise ConsultaInvalida("Solo se admite una sentencia.")
    if sql.split(None, 1)[0].lower() not in ("select", "with"):
        raise ConsultaInvalida("Solo se admiten consultas SELECT.")
    max_filas = max(1, min(int(max_filas), MAX_FILAS))
    try:
        cur = conexion().execute(sql)
        filas = cur.fetchmany(max_filas + 1)
    except sqlite3.Error as e:
        raise ConsultaInvalida(f"Consulta rechazada o con error: {e}") from e
    columnas = [d[0] for d in cur.description or []]
    truncado = len(filas) > max_filas
    return {"columnas": columnas, "filas": [list(f) for f in filas[:max_filas]],
            "truncado": truncado, "max_filas": max_filas}


def describir() -> dict:
    """Tablas, columnas con su tipo, número de filas y proporción de >50K (income = 1)."""
    con = conexion()
    out = {}
    for t, nota in TABLAS.items():
        cur = con.execute(f"SELECT * FROM {t} LIMIT 0")
        cols = [d[0] for d in cur.description]
        n, p = con.execute(f"SELECT COUNT(*), AVG(income) FROM {t}").fetchone()
        out[t] = {"descripcion": nota, "filas": n, "proporcion_mayor_50k": round(p, 4), "columnas": cols}
    out["_notas"] = ["income: 1 = >50K, 0 = <=50K.",
                     "El dump crece con el tiempo: periodo 0 = base; 1-12 = llegadas (simuladas: data drift en 3-4, "
                     "concept drift desde el 5).",
                     "Valores faltantes de workclass, occupation y native-country son NULL."]
    return out
