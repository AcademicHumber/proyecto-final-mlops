"""Interfaz para priorizar prospectos: consume la API desplegada en Kubernetes.

Caso de negocio (docs/crisp-dm.md): una entidad financiera decide a quién contactar primero para
un producto premium. El modelo estima la probabilidad de que el prospecto gane más de 50K; la
interfaz la traduce a una prioridad comercial. Es una herramienta de priorización, no una decisión
automática de aprobar o rechazar.

Uso:
    minikube service adult-api --url          # copiar la URL que imprime
    $env:API_URL = "<esa URL>"
    streamlit run ui/app.py
"""
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui.perfiles import (CON_DESCONOCIDO, DESCONOCIDO, NUMERICAS, OPCIONES, PERFILES, VALORES_BASE,  # noqa: E402
                         a_registro, accion, fila_a_valores, perfil_completo, prioridad)

DEFAULT_URL = os.getenv("API_URL", "http://localhost:8080")
DATOS_CSV = ROOT / "data" / "datos.csv"
LOTE = 500   # filas por petición al enviar una campaña
ETIQUETAS = {"age": "Edad", "hours-per-week": "Horas por semana", "capital-gain": "Ganancia de capital",
             "capital-loss": "Pérdida de capital", "fnlwgt": "Peso muestral (fnlwgt)",
             "workclass": "Tipo de empleo", "education": "Educación", "marital-status": "Estado civil",
             "occupation": "Ocupación", "relationship": "Relación familiar", "race": "Raza", "sex": "Sexo",
             "native-country": "País de origen"}

st.set_page_config(page_title="Priorización de prospectos", page_icon="🎯", layout="wide")


# ---------- acceso a la API ----------
def api_get(url: str, ruta: str):
    r = requests.get(f"{url.rstrip('/')}{ruta}", timeout=5)
    r.raise_for_status()
    return r.json()


def api_predict(url: str, registros: list[dict]) -> dict:
    """Una petición nueva por llamada (sin sesión), para que el Service reparta entre pods."""
    r = requests.post(f"{url.rstrip('/')}/predict", json=registros, timeout=60)
    r.raise_for_status()
    return r.json()


def registrar_pod(resp: dict) -> None:
    st.session_state.pods[resp["pod"]] += 1


@st.cache_data(show_spinner=False)
def cargar_test() -> pd.DataFrame | None:
    """Filas reales de ejemplo: la partición test de la base (periodo 0 del dump)."""
    if not DATOS_CSV.exists():
        return None
    d = pd.read_csv(DATOS_CSV)
    return d[d["particion"] == "test"].drop(columns=["periodo", "particion"]).reset_index(drop=True)


# ---------- estado del formulario ----------
def fijar_formulario(valores: dict) -> None:
    for k, v in valores.items():
        st.session_state[f"f_{k}"] = v


def cb_perfil(nombre: str) -> None:
    fijar_formulario(perfil_completo(nombre))


def cb_fila_real(clase: int | None) -> None:
    df = cargar_test()
    if df is None:
        return
    sub = df if clase is None else df[df["income"] == clase]
    fila = sub.sample(1).iloc[0]
    fijar_formulario(fila_a_valores(fila.to_dict()))
    st.session_state.verdad = int(fila["income"])


def cb_limpiar() -> None:
    fijar_formulario(VALORES_BASE)
    st.session_state.pop("verdad", None)
    st.session_state.pop("resultado", None)


if "pods" not in st.session_state:
    st.session_state.pods = Counter()
    st.session_state.historial = []
    fijar_formulario(VALORES_BASE)

# ---------- barra lateral: conexión con la API ----------
with st.sidebar:
    st.header("Servicio")
    api_url = st.text_input("URL de la API", DEFAULT_URL,
                            help="Con minikube: `minikube service adult-api --url`")
    try:
        info = api_get(api_url, "/info")
        st.success("Conectado")
        st.markdown(f"**Modelo:** `{info['model_name']}`  \n**Versión:** {info['model_version']} "
                    f"(alias `{info['alias']}`)  \n**Run:** `{info['model_run_id'][:8]}…`")
    except requests.RequestException as e:
        info = None
        st.error("Sin conexión con la API")
        st.caption(f"{type(e).__name__}. Revisa la URL y que los pods estén `1/1`.")
    if st.session_state.pods:
        st.divider()
        st.subheader("Pods que han respondido")
        st.caption("Cada consulta puede atenderla un pod distinto: así se ve el balanceo.")
        st.bar_chart(pd.Series(dict(st.session_state.pods)), horizontal=True)

st.title("🎯 Priorización de prospectos")
st.caption("Producto premium: ¿a quién contactar primero? El modelo estima la probabilidad de ingresos "
           "mayores a 50K. Es una guía para priorizar, no una decisión automática.")

tab_uno, tab_lote = st.tabs(["Un prospecto", "Campaña (lote)"])

# ---------- pestaña 1: un prospecto ----------
with tab_uno:
    st.markdown("**Atajos para rellenar**")
    cols = st.columns(len(PERFILES))
    for c, nombre in zip(cols, PERFILES):
        c.button(nombre, key=f"perfil_{nombre}", on_click=cb_perfil, args=(nombre,), width="stretch")

    test = cargar_test()
    a1, a2, a3, a4 = st.columns(4)
    a1.button("🎲 Prospecto real al azar", on_click=cb_fila_real, args=(None,), disabled=test is None,
              width="stretch", help="Una fila de la partición test de data/datos.csv")
    a2.button("🎲 Real que gana >50K", on_click=cb_fila_real, args=(1,), disabled=test is None,
              width="stretch")
    a3.button("🎲 Real que gana ≤50K", on_click=cb_fila_real, args=(0,), disabled=test is None,
              width="stretch")
    a4.button("🧹 Limpiar", on_click=cb_limpiar, width="stretch")
    if test is None:
        st.caption("Los atajos con datos reales necesitan `data/datos.csv` (`dvc pull`).")

    st.divider()
    izq, der = st.columns([3, 2])
    with izq:
        c1, c2, c3 = st.columns(3)
        c1.number_input(ETIQUETAS["age"], NUMERICAS["age"][0], NUMERICAS["age"][1], key="f_age")
        c2.number_input(ETIQUETAS["hours-per-week"], NUMERICAS["hours-per-week"][0],
                        NUMERICAS["hours-per-week"][1], key="f_hours-per-week")
        c3.selectbox(ETIQUETAS["sex"], OPCIONES["sex"], key="f_sex")

        def selector(col, campo):
            ops = ([DESCONOCIDO] if campo in CON_DESCONOCIDO else []) + OPCIONES[campo]
            col.selectbox(ETIQUETAS[campo], ops, key=f"f_{campo}")

        c1, c2, c3 = st.columns(3)
        selector(c1, "education")
        selector(c2, "workclass")
        selector(c3, "occupation")
        c1, c2, c3 = st.columns(3)
        selector(c1, "marital-status")
        selector(c2, "relationship")
        selector(c3, "race")
        c1, c2, c3 = st.columns(3)
        selector(c1, "native-country")
        c2.number_input(ETIQUETAS["capital-gain"], NUMERICAS["capital-gain"][0],
                        NUMERICAS["capital-gain"][1], key="f_capital-gain")
        c3.number_input(ETIQUETAS["capital-loss"], NUMERICAS["capital-loss"][0],
                        NUMERICAS["capital-loss"][1], key="f_capital-loss")
        with st.expander("Avanzado"):
            st.number_input(ETIQUETAS["fnlwgt"], NUMERICAS["fnlwgt"][0], NUMERICAS["fnlwgt"][1], key="f_fnlwgt",
                            help="Peso estadístico del censo. Sin efecto práctico en un prospecto real.")

        if st.button("Evaluar prospecto", type="primary", disabled=info is None):
            valores = {k: st.session_state[f"f_{k}"] for k in VALORES_BASE}
            try:
                resp = api_predict(api_url, [a_registro(valores)])
                registrar_pod(resp)
                p = resp["predictions"][0]["probability_over_50k"]
                st.session_state.resultado = {"p": p, "pod": resp["pod"], "version": resp["model_version"]}
                st.session_state.historial.insert(0, {
                    "Edad": valores["age"], "Educación": valores["education"],
                    "Ocupación": valores["occupation"], "Probabilidad >50K": p,
                    "Prioridad": prioridad(p), "Pod": resp["pod"], "Modelo": f"v{resp['model_version']}"})
            except requests.RequestException as e:
                st.error(f"No se pudo consultar la API: {e}")

    with der:
        res = st.session_state.get("resultado")
        if res:
            p = res["p"]
            st.metric("Probabilidad de ganar >50K", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
            nivel = prioridad(p)
            {"Alta": st.success, "Media": st.warning, "Baja": st.info}[nivel](
                f"**Prioridad {nivel}.** {accion(p)}")
            st.caption(f"Atendido por el pod `{res['pod']}` · modelo v{res['version']}")
            if "verdad" in st.session_state:
                real = ">50K" if st.session_state.verdad == 1 else "≤50K"
                pred = ">50K" if p >= 0.5 else "≤50K"
                st.caption(f"Ingreso real de esta fila: **{real}** · el modelo dice **{pred}** "
                           f"({'acierta' if real == pred else 'falla'}).")
        else:
            st.info("Elige un atajo o completa el formulario y pulsa **Evaluar prospecto**.")

    if st.session_state.historial:
        st.subheader("Consultas recientes")
        st.dataframe(pd.DataFrame(st.session_state.historial[:10]), hide_index=True, width="stretch",
                     column_config={"Probabilidad >50K": st.column_config.ProgressColumn(
                         min_value=0.0, max_value=1.0, format="%.2f")})

# ---------- pestaña 2: campaña ----------
with tab_lote:
    st.markdown("Puntúa una lista de prospectos y marca a quién contactar según el presupuesto de la campaña.")
    origen = st.radio("Origen de los prospectos", ["Filas al azar de la partición test", "Subir un CSV"], horizontal=True)
    df_lote = None
    if origen == "Filas al azar de la partición test":
        if test is None:
            st.warning("No hay `data/datos.csv` (ejecuta `dvc pull`).")
        else:
            n = st.slider("Cantidad de prospectos", 50, 2000, 500, step=50)
            if st.button("Armar muestra"):
                st.session_state.muestra = test.sample(n, random_state=None).reset_index(drop=True)
            df_lote = st.session_state.get("muestra")
    else:
        archivo = st.file_uploader("CSV con las 14 variables (las 14 variables, como en data/datos.csv)", type="csv")
        if archivo:
            df_lote = pd.read_csv(archivo)

    if df_lote is not None:
        st.caption(f"{len(df_lote)} prospectos listos.")
        presupuesto = st.slider("Presupuesto: contactar al mejor X %", 1, 50, 10,
                                help="10 % es la campaña de referencia del proyecto (precision@10 %).")
        if st.button("Puntuar campaña", type="primary", disabled=info is None):
            columnas = list(a_registro(VALORES_BASE))
            faltan = [c for c in columnas if c not in df_lote.columns]
            if faltan:
                st.error(f"Faltan columnas: {faltan}")
            else:
                registros = [{k: (None if pd.isna(v) else v) for k, v in fila.items()}
                             for fila in df_lote[columnas].to_dict("records")]
                probas = []
                try:
                    barra = st.progress(0.0, text="Consultando la API…")
                    for i in range(0, len(registros), LOTE):
                        resp = api_predict(api_url, registros[i:i + LOTE])
                        registrar_pod(resp)
                        probas += [x["probability_over_50k"] for x in resp["predictions"]]
                        barra.progress(min(1.0, (i + LOTE) / len(registros)))
                    barra.empty()
                    out = df_lote.copy()
                    out.insert(0, "probabilidad_>50K", probas)
                    out = out.sort_values("probabilidad_>50K", ascending=False).reset_index(drop=True)
                    out.insert(1, "prioridad", out["probabilidad_>50K"].map(prioridad))
                    k = max(1, int(len(out) * presupuesto / 100))
                    out["contactar"] = out.index < k
                    st.session_state.campana = (out, k, resp["model_version"])
                except requests.RequestException as e:
                    st.error(f"No se pudo consultar la API: {e}")

        if "campana" in st.session_state:
            out, k, version = st.session_state.campana
            m1, m2, m3 = st.columns(3)
            m1.metric("Prospectos a contactar", f"{k} de {len(out)}")
            m2.metric("Probabilidad mínima del grupo", f"{out['probabilidad_>50K'].iloc[k - 1]:.2f}")
            if "income" in out.columns:
                acierto = out["income"].iloc[:k].mean()
                base = out["income"].mean()
                m3.metric("Realmente >50K en el grupo", f"{acierto:.0%}", f"{acierto - base:+.0%} vs. toda la lista")
            else:
                m3.metric("Modelo usado", f"v{version}")
            cuenta = out["prioridad"].value_counts().reindex(["Alta", "Media", "Baja"]).fillna(0)
            st.bar_chart(cuenta)
            st.dataframe(out.head(200), hide_index=True, width="stretch")
            st.download_button("Descargar lista priorizada (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="campana_priorizada.csv", mime="text/csv")
