import time

import streamlit as st

from modulos.motor_bayesiano import MotorDiagnostico
from modulos.traducciones import DICCIONARIO_ENFERMEDADES, DICCIONARIO_SINTOMAS
from modulos import ui_utils as ui_utils_module
from modulos.visualizador import generar_grafo_consulta, generar_grafo_top3
from modulos.app_helpers import (
    mostrar_desglose_multiplicativo,
    mostrar_desarrollo_bayesiano_inferencia,
    mostrar_desarrollo_bayesiano_estandar,
    mostrar_desarrollo_bayesiano_causal,
)
from modulos.secciones import (
    render_diagnostico_estandar,
    render_inferencia_diagnostica,
    render_causalidad_clinica,
)


def render_sidebar():
    # Barra lateral fija con la ficha académica del proyecto.
    with st.sidebar:
        # Imagen institucional centrada.
        col_img = st.columns([1, 6, 1])[1]
        col_img.image("./imagenes/logo-unjbg.png", use_container_width=True)

        st.markdown("<h3 style='text-align: center;'>Universidad Nacional<br>Jorge Basadre Grohmann</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888888;'>Ingeniería en Informática y Sistemas</p>", unsafe_allow_html=True)
        st.divider()

        st.markdown("#### :material/info: Ficha del Proyecto")
        st.markdown("**Curso:** Sistemas Expertos")
        st.markdown("**Tema:** Motor de Inferencia Médica usando Redes Bayesianas")
        st.markdown("**Docente:** Ing. Reynaldo Alfonte Zapana")
        st.divider()

        st.markdown("#### :material/group: Integrantes")
        st.markdown(
            """
            * **Marco Luis Samuel Valdivia Ticona** (`2022-119036`)
            * **Alex Javier Flores Ticona** (`2022-119072`)
            * **Jhon Alexander Mamani Arrazola** (`2022-119050`)
            * **Snais Ladinson Mamani Astete** (`2022-119011`)
            """
        )


@st.cache_resource
def iniciar_servicio():
    # Carga y entrenamiento inicial del modelo bayesiano.
    time.sleep(3)
    motor = MotorDiagnostico(ruta_datos='datos/dataset_enfermedades.csv')
    cols = motor.entrenar_modelo()
    return motor, cols


def render_app():
    # Se reutilizan las utilidades de UI ya importadas por el módulo.
    ui_utils = ui_utils_module
    mostrar_resultados_top = ui_utils.mostrar_resultados_top
    mostrar_ranking_enfermedades = ui_utils.mostrar_ranking_enfermedades
    mostrar_tablas_cpt = ui_utils.mostrar_tablas_cpt

    # Sidebar + carga del motor antes de mostrar la navegación principal.
    render_sidebar()

    with st.spinner("Entrenando red bayesiana con dataset clínico... (3s)"):
        motor, columnas = iniciar_servicio()

    mapa_sintomas = {DICCIONARIO_SINTOMAS.get(s, s.replace('_', ' ').title()): s for s in columnas}  # Traduce los síntomas al español.
    opciones_sintomas = list(mapa_sintomas.keys())

    st.title(":material/vital_signs: Sistema Experto de Diagnóstico Bayesiano")
    st.divider()

    # Selector central para movernos entre las tres vistas.
    seccion_activa = st.radio(
        "Navegación principal",
        options=[
            "Diagnóstico Estándar",
            "Análisis de Inferencia Diagnóstica",
            "Análisis de Causalidad Clínica",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if seccion_activa == "Diagnóstico Estándar":
        # Vista principal de diagnóstico con ranking, CPT y grafo.
        render_diagnostico_estandar(
            motor,
            mapa_sintomas,
            opciones_sintomas,
            DICCIONARIO_ENFERMEDADES,
            mostrar_resultados_top,
            mostrar_ranking_enfermedades,
            mostrar_tablas_cpt,
            generar_grafo_top3,
            mostrar_desglose_multiplicativo,
            mostrar_desarrollo_bayesiano_estandar,
        )
    elif seccion_activa == "Análisis de Inferencia Diagnóstica":
        # Vista de inferencia: posterior Bayesiano para una patología elegida.
        render_inferencia_diagnostica(
            motor,
            mapa_sintomas,
            opciones_sintomas,
            DICCIONARIO_ENFERMEDADES,
            DICCIONARIO_SINTOMAS,
            mostrar_tablas_cpt,
            generar_grafo_consulta,
            mostrar_desglose_multiplicativo,
            mostrar_desarrollo_bayesiano_inferencia,
        )
    else:
        # Vista causal: probabilidad de síntomas condicionada al diagnóstico.
        render_causalidad_clinica(
            motor,
            mapa_sintomas,
            opciones_sintomas,
            DICCIONARIO_ENFERMEDADES,
            DICCIONARIO_SINTOMAS,
            mostrar_tablas_cpt,
            generar_grafo_consulta,
            mostrar_desglose_multiplicativo,
            mostrar_desarrollo_bayesiano_causal,
        )
