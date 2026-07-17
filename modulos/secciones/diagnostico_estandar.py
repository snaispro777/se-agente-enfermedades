import time

import streamlit as st


def render_diagnostico_estandar(
    motor,
    mapa_sintomas,
    opciones_sintomas,
    diccionario_enfermedades,
    mostrar_resultados_top,
    mostrar_ranking_enfermedades,
    mostrar_tablas_cpt,
    generar_grafo_top3,
    mostrar_desglose_multiplicativo,
    mostrar_desarrollo_bayesiano_estandar,
):
    # Formulario principal del diagnóstico estándar.
    st.subheader(":material/list_alt: Cuadro Clínico del Paciente")
    seleccionados = st.multiselect("Síntomas observados:", options=opciones_sintomas, max_selections=15)

    if st.button("Ejecutar Diagnóstico", type="primary"):
        if not seleccionados:
            st.warning("Seleccione síntomas.")
            return

        inicio_respuesta = time.perf_counter()

        # Evidencia binaria: el síntoma seleccionado se marca como presente.
        evidencia = {mapa_sintomas[s]: 1 for s in seleccionados}
        resultados_ordenados = motor.calcular_probabilidades(evidencia)

        # Ranking principal y vista resumida.
        mostrar_resultados_top(resultados_ordenados, diccionario_enfermedades)
        mostrar_ranking_enfermedades(resultados_ordenados, diccionario_enfermedades)

        # Buscamos el diagnóstico top para explicar su camino matemático.
        top_enfermedad_raw, _ = resultados_ordenados[0]
        desglose_inferencia = motor.obtener_desglose_inferencia(evidencia)
        detalle_top = next(
            (item for item in desglose_inferencia if item["enfermedad"].strip() == top_enfermedad_raw.strip()),
            None,
        )

        if detalle_top:
            top_enfermedad_es = diccionario_enfermedades.get(top_enfermedad_raw.strip(), top_enfermedad_raw.strip())
            # Bloque bayesiano explicativo del diagnóstico principal.
            mostrar_desarrollo_bayesiano_estandar(
                top_enfermedad_es,
                seleccionados,
                detalle_top,
                len(seleccionados),
            )

        # CPT asociadas a cada síntoma seleccionado.
        mostrar_tablas_cpt(seleccionados, mapa_sintomas, motor, diccionario_enfermedades)

        if detalle_top:
            st.caption("Ruta de cálculo del diagnóstico principal.")
            top_enfermedad_es = diccionario_enfermedades.get(top_enfermedad_raw.strip(), top_enfermedad_raw.strip())
            mostrar_desglose_multiplicativo(
                f"Ruta de cálculo para {top_enfermedad_es}",
                detalle_top["pasos"],
                detalle_top["posterior"],
                "Posterior normalizado",
            )

        # Grafo resumido con el top 3 del diagnóstico.
        st.caption("Red bayesiana resumida con los tres diagnósticos más probables.")
        fig_top3 = generar_grafo_top3(
            resultados_ordenados[:3],
            diccionario_enfermedades,
            etiqueta_consulta="Diagnóstico",
            conteo_evidencia=len(seleccionados),
        )
        st.pyplot(fig_top3)

        tiempo_respuesta_ms = (time.perf_counter() - inicio_respuesta) * 1000
        st.metric("Tiempo de respuesta (ms)", f"{tiempo_respuesta_ms:.0f}")
