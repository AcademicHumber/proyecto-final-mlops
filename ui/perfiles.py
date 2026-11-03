"""Lógica de la interfaz sin Streamlit: opciones, perfiles rápidos, prioridad y armado del JSON.

Se separa de `app.py` para poder probarla sin levantar la interfaz (tests/test_ui.py).
"""
import math

DESCONOCIDO = "(desconocido)"   # la API acepta null en las categóricas; el modelo las imputa

EDUCACION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, "11th": 7,
    "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}

OPCIONES = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                  "Without-pay", "Never-worked"],
    "education": list(EDUCACION_NUM),
    "marital-status": ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing",
                   "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv",
                   "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving"],
    "relationship": ["Husband", "Wife", "Not-in-family", "Own-child", "Unmarried", "Other-relative"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "sex": ["Male", "Female"],
    "native-country": ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "India", "Cuba", "England", "China", "Jamaica", "Italy", "Japan", "Vietnam", "South", "Dominican-Republic", "El-Salvador", "Guatemala", "Poland", "Columbia", "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "Greece", "France", "Ecuador", "Ireland", "Hong", "Thailand", "Cambodia", "Trinadad&Tobago", "Laos", "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Honduras", "Hungary", "Scotland", "Holand-Netherlands"],
}
# Las categorías con valores faltantes en los datos reales
CON_DESCONOCIDO = ("workclass", "occupation", "native-country")

NUMERICAS = {  # (mínimo, máximo, valor por defecto)
    "age": (17, 90, 38),
    "hours-per-week": (1, 99, 40),
    "capital-gain": (0, 99999, 0),
    "capital-loss": (0, 4356, 0),
    "fnlwgt": (12285, 1500000, 178319),
}

VALORES_BASE = {
    "age": 38, "hours-per-week": 40, "capital-gain": 0, "capital-loss": 0, "fnlwgt": 178319,
    "workclass": "Private", "education": "HS-grad", "marital-status": "Married-civ-spouse",
    "occupation": "Craft-repair", "relationship": "Husband", "race": "White", "sex": "Male",
    "native-country": "United-States",
}

# Perfiles de atajo: puntos de partida para la demo, no personas reales.
PERFILES = {
    "Ejecutivo": {"age": 45, "workclass": "Private", "education": "Masters", "marital-status": "Married-civ-spouse",
                  "occupation": "Exec-managerial", "relationship": "Husband", "sex": "Male", "hours-per-week": 50},
    "Profesional con inversiones": {"age": 52, "workclass": "Self-emp-inc", "education": "Prof-school",
                                    "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty",
                                    "relationship": "Husband", "sex": "Male", "hours-per-week": 55,
                                    "capital-gain": 15024},
    "Recién graduada": {"age": 24, "workclass": "Private", "education": "Bachelors", "marital-status": "Never-married",
                        "occupation": "Tech-support", "relationship": "Not-in-family", "sex": "Female",
                        "hours-per-week": 40},
    "Oficio estable": {"age": 38, "workclass": "Private", "education": "HS-grad", "marital-status": "Married-civ-spouse",
                       "occupation": "Craft-repair", "relationship": "Husband", "sex": "Male", "hours-per-week": 40},
    "Joven medio tiempo": {"age": 19, "workclass": "Private", "education": "Some-college",
                           "marital-status": "Never-married", "occupation": "Other-service",
                           "relationship": "Own-child", "sex": "Male", "hours-per-week": 20},
    "Jubilada, pocas horas": {"age": 68, "workclass": "Private", "education": "HS-grad", "marital-status": "Widowed",
                              "occupation": "Adm-clerical", "relationship": "Not-in-family", "sex": "Female",
                              "hours-per-week": 15},
}

# Umbrales de prioridad comercial. Son un supuesto de la interfaz (no del modelo, que decide con 0,5).
PRIORIDAD_ALTA = 0.70
PRIORIDAD_MEDIA = 0.40


def perfil_completo(nombre: str) -> dict:
    """Valores base + los del perfil: todos los campos del formulario quedan definidos."""
    return {**VALORES_BASE, **PERFILES[nombre]}


def prioridad(p: float) -> str:
    if p >= PRIORIDAD_ALTA:
        return "Alta"
    return "Media" if p >= PRIORIDAD_MEDIA else "Baja"


def accion(p: float) -> str:
    return {"Alta": "Contactar primero.",
            "Media": "Contactar si el presupuesto de la campaña lo permite.",
            "Baja": "No priorizar."}[prioridad(p)]


def _faltante(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or v == DESCONOCIDO


def a_registro(valores: dict) -> dict:
    """Valores del formulario -> registro JSON para POST /predict (14 variables, nombres con guion)."""
    reg = {k: float(valores[k]) for k in NUMERICAS}
    reg["education-num"] = float(EDUCACION_NUM[valores["education"]])
    for k in OPCIONES:
        v = valores.get(k)
        reg[k] = None if _faltante(v) else str(v)
    return reg


def fila_a_valores(fila: dict) -> dict:
    """Fila de los datos (con NaN y países fuera de la lista) -> valores del formulario."""
    out = {}
    for k in NUMERICAS:
        lo, hi, _ = NUMERICAS[k]
        out[k] = int(min(max(fila[k], lo), hi))
    for k, opciones in OPCIONES.items():
        v = fila.get(k)
        if _faltante(v):
            out[k] = DESCONOCIDO if k in CON_DESCONOCIDO else VALORES_BASE[k]
        else:
            out[k] = v if v in opciones else VALORES_BASE[k]
    return out
