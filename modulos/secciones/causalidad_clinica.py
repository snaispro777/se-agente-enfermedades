import streamlit as st


def render_causalidad_clinica(
    motor,
    mapa_sintomas,
    opciones_sintomas,
    diccionario_enfermedades,
    diccionario_sintomas,
    mostrar_tablas_cpt,
    generar_grafo_consulta,
    mostrar_desglose_multiplicativo,
    mostrar_desarrollo_bayesiano_causal,
):
    # Formulario de causalidad clínica: diagnóstico fijo y síntomas esperados.
    st.subheader(":material/psychology: Análisis de Causalidad Clínica")
    st.markdown("### Calcular probabilidad de síntomas dado un Diagnóstico")
    enf_causal_es = st.selectbox("Si el paciente tuviera:", options=sorted(list(diccionario_enfermedades.values())))
    sintomas_causal_es = st.multiselect("¿Probabilidad de presentar estos síntomas?", options=opciones_sintomas)

    if st.button("Calcular Probabilidad Generativa"):
        # El diagnóstico se traduce al nombre interno usado por la red.
        mapa_inv_enf = {v.strip(): k for k, v in diccionario_enfermedades.items()}
        enf_en = mapa_inv_enf.get(enf_causal_es.strip())
        sintomas_en = [mapa_sintomas[s] for s in sintomas_causal_es]

        try:
            if not enf_en:
                raise ValueError(f"No se encontró la traducción al inglés para '{enf_causal_es}'.")

            # Producto de CPTs para obtener la verosimilitud generativa.
            desglose_causalidad = motor.obtener_desglose_causalidad(enf_en, sintomas_en)
            prob = desglose_causalidad["puntaje"]
            st.metric(f"Probabilidad de los síntomas dado {enf_causal_es}", f"{prob * 100:.6f}%")

            # Desarrollo matemático, CPT y descomposición paso a paso.
            mostrar_desarrollo_bayesiano_causal(enf_causal_es, sintomas_causal_es, desglose_causalidad)
            mostrar_tablas_cpt(sintomas_causal_es, mapa_sintomas, motor, diccionario_enfermedades)

            st.caption("Ruta de cálculo de la causalidad clínica.")
            mostrar_desglose_multiplicativo(
                f"Ruta de cálculo para {enf_causal_es}",
                desglose_causalidad["pasos"],
                prob,
                "Probabilidad generativa",
            )

            # Grafo con la evidencia causal resaltada.
            st.caption("Red bayesiana usada en la consulta causal, con síntomas seleccionados resaltados.")
            if sintomas_causal_es:
                sintomas_resaltados_en = [mapa_sintomas[s] for s in sintomas_causal_es]
                fig = generar_grafo_consulta(
                    sintomas_resaltados_en,
                    sintomas_resaltados=sintomas_resaltados_en,
                    nodo_consulta="prognosis",
                    diccionario_sintomas=diccionario_sintomas,
                    etiqueta_consulta=f"{enf_causal_es}\nScore {prob * 100:.2f}%",
                    detalle_nodos=desglose_causalidad["pasos"],
                )
                st.pyplot(fig)
            else:
                st.info("Selecciona síntomas para mostrar el fragmento de la red.")
        except Exception as e:
            st.error(f"Error: {e}")
