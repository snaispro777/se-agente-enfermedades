import pandas as pd
import streamlit as st


def mostrar_desglose_multiplicativo(titulo, pasos, resultado_final, etiqueta_final):
    # Resume el producto secuencial de factores y su acumulado.
    st.markdown(f"##### {titulo}")

    if not pasos:
        st.info("No hay pasos disponibles para mostrar.")
        return

    filas = []
    cadena_formula = []
    for indice, paso in enumerate(pasos):
        # Cada fila muestra un factor, su valor y el acumulado parcial.
        etiqueta = paso["factor"]
        probabilidad = paso["probabilidad"]
        acumulado = paso["acumulado"]
        cadena_formula.append(f"{probabilidad:.6f}")
        filas.append({
            "Paso": indice + 1,
            "Factor": etiqueta,
            "Probabilidad": f"{probabilidad:.6f}",
            "Acumulado": f"{acumulado:.12f}",
        })

    st.code(" × ".join(cadena_formula), language="text")
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    st.success(f"{etiqueta_final}: {resultado_final * 100:.6f}%")


def mostrar_desarrollo_bayesiano_inferencia(patologia_es, evidencia_es, detalle_target):
    # Bloque explicativo del posterior bayesiano para una patología concreta.
    st.subheader(":material/functions: Desarrollo matemático de la inferencia")
    st.markdown(
        "Se aplica el teorema de Bayes sobre una red tipo Naive Bayes. "
        "La probabilidad posterior se obtiene al multiplicar el prior de la patología por las CPT de los síntomas observados y luego normalizar."
    )
    st.markdown(
        "En esta sección, el **numerador** representa lo que explica una patología concreta para los síntomas observados. "
        "El **denominador** suma la explicación de *todas* las patologías posibles, y por eso convierte el valor bruto en una probabilidad comparable."
    )
    st.latex(r"P(H \mid E) = \frac{P(H, E)}{P(E)}")
    st.latex(r"P(H \mid E) = \alpha \sum_{\mathbf{z}} P(H, E, \mathbf{z})")
    st.latex(r"P(H \mid E) = \frac{P(H) \prod_{i=1}^{n} P(e_i \mid H)}{\sum_{h \in \mathcal{H}} P(h) \prod_{i=1}^{n} P(e_i \mid h)}")
    st.caption(f"H corresponde a la patología evaluada: {patologia_es}.")
    st.caption("E representa la evidencia clínica observada.")
    st.markdown(
        "Lectura rápida: primero se toma la probabilidad base de la enfermedad, luego se multiplican los síntomas que la respaldan, "
        "y al final se divide entre la suma de todas las alternativas posibles para obtener una probabilidad final entre 0 y 1."
    )

    pasos = detalle_target["pasos"]
    if not pasos:
        st.info("No hay pasos matemáticos disponibles para mostrar.")
        return

    st.markdown("**Desarrollo numérico:**")
    st.markdown(
        "Cada línea de abajo muestra el valor acumulado hasta ese momento. "
        "Esto es útil porque permite ver cómo el resultado cambia al incorporar el prior y luego cada síntoma observado."
    )
    paso_prior = pasos[0]  # Primer término: prior de la patología.
    st.latex(rf"\text{{Paso 1: Prior}}\quad P(H) = {paso_prior['probabilidad']:.6f}")

    acumulado_prev = paso_prior["acumulado"]
    for indice, paso in enumerate(pasos[1:], start=2):
        # Multiplicación incremental de cada evidencia observada.
        nombre_factor = paso["factor"].replace("_", r"\_")
        st.latex(
            rf"\text{{Paso {indice}:}}\quad {acumulado_prev:.12f} \times {paso['probabilidad']:.6f} = {paso['acumulado']:.12f}"
        )
        st.markdown(f"- Factor usado: `{nombre_factor}`")
        acumulado_prev = paso["acumulado"]

    st.latex(r"\text{Normalización} = \sum_{h \in \mathcal{H}} P(h) \prod_{i=1}^{n} P(e_i \mid h)")
    st.latex(rf"\text{{Posterior final}} = {detalle_target['posterior']:.6f}")


def mostrar_desarrollo_bayesiano_estandar(patologia_es, evidencia_es, detalle_target, cantidad_sintomas):
    # Misma explicación bayesiana, pero aplicada al flujo principal del diagnóstico.
    st.subheader(":material/functions: Desarrollo matemático del diagnóstico estándar")
    st.markdown(
        "En el diagnóstico estándar se aplica Bayes para estimar la patología más probable a partir de la evidencia clínica. "
        "El motor calcula una puntuación para cada enfermedad usando su prior y la contribución de cada síntoma observado."
    )
    st.markdown(
        "La idea central es comparar qué tan bien explica cada patología la evidencia. "
        "Por eso se calcula una expresión proporcional al posterior y luego se normaliza entre todas las enfermedades posibles."
    )
    st.latex(r"P(H \mid E) = \frac{P(E \mid H)P(H)}{P(E)}")
    st.latex(r"P(H \mid E) = \frac{P(H) \prod_{i=1}^{n} P(e_i \mid H)}{\sum_{h \in \mathcal{H}} P(h) \prod_{i=1}^{n} P(e_i \mid h)}")
    st.caption(f"H corresponde a la patología evaluada: {patologia_es}.")
    st.caption(f"E agrupa los {cantidad_sintomas} síntomas seleccionados como evidencia.")
    st.markdown(
        "Primero aparece el prior $P(H)$, después se multiplican las CPT de los síntomas observados, "
        "y finalmente se divide entre la suma total de todas las patológicas candidatas para obtener una probabilidad comparable."
    )

    pasos = detalle_target["pasos"]
    if not pasos:
        st.info("No hay pasos matemáticos disponibles para mostrar.")
        return

    st.markdown("**Desarrollo numérico:**")
    st.markdown(
        "La siguiente secuencia muestra cómo el valor acumulado cambia después de cada factor de la evidencia. "
        "Esto permite ver exactamente qué aporta cada síntoma al diagnóstico final."
    )
    paso_prior = pasos[0]  # Prior de la mejor patología detectada.
    st.latex(rf"\text{{Paso 1: Prior}}\quad P(H) = {paso_prior['probabilidad']:.6f}")

    acumulado_prev = paso_prior["acumulado"]
    for indice, paso in enumerate(pasos[1:], start=2):
        # Cada síntoma refuerza o debilita el diagnóstico final.
        st.latex(
            rf"\text{{Paso {indice}:}}\quad {acumulado_prev:.12f} \times {paso['probabilidad']:.6f} = {paso['acumulado']:.12f}"
        )
        st.markdown(f"- Factor usado: `{paso['factor']}`")
        st.caption(
            f"Este factor viene de la CPT del síntoma **{paso['detalle']}** para la patología {patologia_es}."
        )
        acumulado_prev = paso["acumulado"]

    st.latex(r"\text{Normalización} = \sum_{h \in \mathcal{H}} P(h) \prod_{i=1}^{n} P(e_i \mid h)")
    st.latex(rf"\text{{Posterior final}} = {detalle_target['posterior']:.6f}")


def mostrar_desarrollo_bayesiano_causal(patologia_es, evidencia_es, desglose_causalidad):
    # Desarrollo generativo: síntomas condicionados al diagnóstico.
    st.subheader(":material/functions: Desarrollo matemático de la causalidad")
    st.markdown(
        "En esta sección se calcula la probabilidad generativa de los síntomas condicionada al diagnóstico. "
        "Como la red es de tipo Naive Bayes, el resultado se obtiene como una multiplicación secuencial de las CPT de cada síntoma."
    )
    st.markdown(
        "Aquí no se busca la enfermedad más probable, sino cuánto explican los síntomas a partir de un diagnóstico fijo. "
        "Por eso el cálculo es una cadena de multiplicaciones: cada síntoma aporta un factor condicional independiente."
    )
    st.markdown(
        "Si quisieras hacer la inferencia en sentido inverso, para estimar el diagnóstico a partir de los síntomas, usarías Bayes: "
        "esa versión sí requiere un denominador de normalización. En esta subsección, en cambio, se muestra la **verosimilitud** de los síntomas dado el diagnóstico."
    )
    st.latex(r"P(D \mid S_1, \dots, S_n) = \frac{P(D) \prod_{i=1}^{n} P(S_i \mid D)}{\sum_{d \in \mathcal{D}} P(d) \prod_{i=1}^{n} P(S_i \mid d)}")
    st.latex(r"P(S_1, S_2, \dots, S_n \mid D) = \prod_{i=1}^{n} P(S_i \mid D)")
    st.latex(r"\text{Score}(D) = P(S_1 \mid D) \times P(S_2 \mid D) \times \cdots \times P(S_n \mid D)")
    st.caption(f"D corresponde al diagnóstico evaluado: {patologia_es}.")
    st.caption("S_i representa cada síntoma seleccionado como evidencia causal.")
    st.markdown(
        "En términos intuitivos: si un síntoma es muy compatible con el diagnóstico, su factor será alto; "
        "si es poco compatible, el producto total se reduce rápidamente."
    )
    st.markdown(
        "La interpretación práctica es: cada factor condicional responde a la pregunta **'si el paciente tuviera este diagnóstico, "
        "qué tan probable es observar este síntoma'**. Al multiplicar todos esos factores, obtenemos una medida global de compatibilidad."
    )

    pasos = desglose_causalidad["pasos"]
    if not pasos:
        st.info("No hay pasos matemáticos disponibles para mostrar.")
        return

    st.markdown("**Desarrollo numérico:**")
    st.markdown(
        "La tabla siguiente muestra el acumulado después de cada multiplicación. "
        "Así se puede seguir el cálculo desde el primer síntoma hasta el score final."
    )
    acumulado_prev = 1.0
    for indice, paso in enumerate(pasos, start=1):
        # Se acumula la probabilidad generativa síntoma por síntoma.
        st.markdown(f"- Factor {indice}: `{paso['factor']}`")
        st.latex(
            rf"\text{{Paso {indice}:}}\quad {acumulado_prev:.12f} \times {paso['probabilidad']:.6f} = {paso['acumulado']:.12f}"
        )
        st.caption(
            f"Este término sale de la CPT del síntoma **{paso['detalle']}** dado el diagnóstico {patologia_es}; "
            f"el valor se toma directamente de la columna correspondiente del nodo diagnóstico."
        )
        acumulado_prev = paso["acumulado"]

    st.latex(rf"\text{{Score final}} = {desglose_causalidad['puntaje']:.12f}")
