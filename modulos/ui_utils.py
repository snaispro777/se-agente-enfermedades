import streamlit as st
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