import streamlit as st

from modulos.app_layout import render_app  # Punto de entrada de toda la interfaz.


st.set_page_config(
    page_title="Agente Médico Bayesiano",
    page_icon="./imagenes/ia.png",
    layout="wide",
)


render_app()
