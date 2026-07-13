import streamlit as st
import pandas as pd
from modulos.traducciones import DICCIONARIO_ENFERMEDADES

def mostrar_resultados_top(resultados, diccionario_enf):
    st.subheader(":material/analytics: Resultados de Inferencia (Top 3)")
    cols = st.columns(3)
    
    for i in range(3):
        enfermedad_raw, prob = resultados[i]
        enf_limpia = enfermedad_raw.strip()
        enf_traducida = diccionario_enf.get(enf_limpia, enf_limpia)
        
        with cols[i]:
            st.metric(label=f"Diagnóstico {i+1}: {enf_traducida}", value=f"{prob * 100:.2f}%")
    
    # Mensaje principal
    top_enf = diccionario_enf.get(resultados[0][0].strip(), resultados[0][0].strip())
    if resultados[0][1] > 0.4:
        st.success(f"Diagnóstico principal: **{top_enf}**", icon=":material/check_circle:")
    else:
        st.info("La evidencia es ambigua.", icon=":material/info:")


def mostrar_ranking_enfermedades(resultados, diccionario_enf):
    """Muestra el ranking completo de enfermedades ordenado por score posterior."""
    st.subheader(":material/list: Ranking completo de enfermedades")

    filas = []
    for indice, (enfermedad_raw, prob) in enumerate(resultados, start=1):
        enf_limpia = enfermedad_raw.strip()
        enf_traducida = diccionario_enf.get(enf_limpia, enf_limpia)
        filas.append({
            "Ranking": indice,
            "Enfermedad": enf_traducida,
            "Score": prob,
            "Score (%)": prob * 100,
        })

    df_ranking = pd.DataFrame(filas)
    st.dataframe(
        df_ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.6f"),
            "Score (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
        
# modulos/ui_utils.py (Añade esta función al archivo existente)

def mostrar_tablas_cpt(sintomas_seleccionados, mapa_sintomas, motor_principal, diccionario_enfermedades):
    """Renderiza pestañas dinámicas con las CPT de los síntomas evaluados."""
    st.divider()
    st.subheader(":material/table_chart: Tablas de Probabilidad Condicional (CPT) Evaluadas")
    st.write(
        "A continuación se muestran los parámetros matemáticos subyacentes ($P(\\text{Síntoma} \\mid \\text{Patología})$) "
        "que el motor multiplicó internamente para resolver el diagnóstico."
    )

    if not sintomas_seleccionados:
        st.info("Selecciona síntomas para visualizar las CPT asociadas.")
        return
    
    # Creamos una pestaña interactiva por cada síntoma seleccionado
    tabs = st.tabs(sintomas_seleccionados)
    
    for idx, nombre_sintoma_es in enumerate(sintomas_seleccionados):
        with tabs[idx]:
            sintoma_en = mapa_sintomas[nombre_sintoma_es]
            try:
                # Obtenemos el DataFrame del motor
                df_cpt = motor_principal.obtener_cpt_dataframe(sintoma_en, diccionario_enfermedades)
                
                st.caption(f"Distribución probabilística para el nodo: **{nombre_sintoma_es}**")
                # Mostramos la matriz completa en un contenedor con scroll
                st.dataframe(df_cpt, use_container_width=True)
                
            except Exception as e:
                st.error(f"No se pudo cargar la CPT para este síntoma: {e}")