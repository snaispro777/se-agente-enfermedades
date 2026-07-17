import time

import streamlit as st


def render_inferencia_diagnostica(
    motor,
    mapa_sintomas,
    opciones_sintomas,
    diccionario_enfermedades,
    diccionario_sintomas,
    mostrar_tablas_cpt,
    generar_grafo_consulta,
    mostrar_desglose_multiplicativo,
    mostrar_desarrollo_bayesiano_inferencia,
):
    # Formulario de consulta personalizada para inferencia bayesiana.
    st.subheader(":material/calculate: Análisis de Inferencia Diagnóstica")
    st.markdown("### Configurar consulta personalizada")
    target = st.selectbox("Patología:", options=list(diccionario_enfermedades.values()))
    fixed = st.multiselect("Evidencia:", options=opciones_sintomas)

    if st.button("Calcular Precisión"):
        inicio_respuesta = time.perf_counter()

        # La evidencia se pasa al motor como diccionario binario.
        evidencia_fija = {mapa_sintomas[s]: 1 for s in fixed}
        mapa_inv = {v.strip(): k for k, v in diccionario_enfermedades.items()}
        target_en = mapa_inv.get(target.strip())

        try:
            if not target_en:
                raise ValueError(f"No se encontró la traducción al inglés para '{target}'.")

            desglose_inferencia = motor.obtener_desglose_inferencia(evidencia_fija)
            detalle_target = next(
                (item for item in desglose_inferencia if item["enfermedad"].strip() == target_en.strip()),
                None,
            )

            if not detalle_target:
                raise ValueError(f"No se encontró el desglose para '{target}'.")

            # Inferencia exacta sobre la red entrenada.
            res = motor.inferencia.query(variables=['prognosis'], evidence=evidencia_fija)
            estados = [s.strip() for s in res.state_names['prognosis']]
            prob = res.values[estados.index(target_en.strip())]
            st.metric(f"Probabilidad de {target}", f"{prob*100:.2f}%")

            # Fórmulas y desarrollo matemático para esta patología.
            mostrar_desarrollo_bayesiano_inferencia(target, fixed, detalle_target)
            # CPT visibles para que el usuario vea los parámetros usados.
            mostrar_tablas_cpt(fixed, mapa_sintomas, motor, diccionario_enfermedades)

            st.caption("Ruta de cálculo de la inferencia para la patología seleccionada.")
            mostrar_desglose_multiplicativo(
                f"Ruta de cálculo para {target}",
                detalle_target["pasos"],
                detalle_target["posterior"],
                "Posterior normalizado",
            )

            # Grafo con la evidencia resaltada.
            st.caption("Red bayesiana usada en la consulta, con síntomas consultados resaltados.")
            if fixed:
                sintomas_resaltados_en = [mapa_sintomas[s] for s in fixed]
                fig = generar_grafo_consulta(
                    sintomas_resaltados_en,
                    sintomas_resaltados=sintomas_resaltados_en,
                    nodo_consulta="prognosis",
                    diccionario_sintomas=diccionario_sintomas,
                    etiqueta_consulta=f"{target}\nPosterior {prob*100:.2f}%",
                    detalle_nodos=detalle_target["pasos"][1:],
                )
                st.pyplot(fig)
            else:
                st.info("Agrega síntomas en la evidencia para ver el fragmento de la red.")

            tiempo_respuesta_ms = (time.perf_counter() - inicio_respuesta) * 1000
            st.metric("Tiempo de respuesta (ms)", f"{tiempo_respuesta_ms:.0f}")
        except Exception as e:
            st.error(f"Error: {e}")
