import streamlit as st
import time
from modulos.motor_bayesiano import MotorDiagnostico
from modulos.traducciones import DICCIONARIO_ENFERMEDADES, DICCIONARIO_SINTOMAS
from modulos.ui_utils import mostrar_resultados_top

st.set_page_config(page_title="Agente Médico Bayesiano", page_icon=":material/health_and_safety:", layout="wide")

# ==========================================
# 1. BARRA LATERAL: FICHA TÉCNICA ACADÉMICA
# ==========================================
with st.sidebar:
    col_img = st.columns([1, 6, 1])[1]
    col_img.image("./imagenes/logo-unjbg.png", use_container_width=True)
    
    st.markdown("<h3 style='text-align: center;'>Universidad Nacional<br>Jorge Basadre Grohmann</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888;'>Ingeniería en Informática y Sistemas<br>Quinto Año</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("#### :material/info: Ficha del Proyecto")
    st.markdown("**Curso:** Inteligencia Artificial / Sistemas Expertos")
    st.markdown("**Unidad:** Unidad III")
    st.markdown("**Tema:** Motor de Inferencia Médica usando Redes Bayesianas")
    st.markdown("**Docente:** Ing. Reynaldo Alfonte Zapana")
    st.divider()
    
    st.markdown("#### :material/group: Equipo de Desarrollo")
    st.markdown("""
    * **Marco Luis Samuel Valdivia Ticona** (`2022-119036`)
    * **Alex Javier Flores Ticona** (`2022-119072`)
    * **Jhon Alexander Mamani Arrazola** (`2022-119050`)
    * **Snais Ladinson Mamani Astete** (`2022-119011`)
    """)
    st.divider()
    st.caption(":material/copyright: 2026 - Desarrollo de Software Académico")

# ==========================================
# 2. MOTOR LÓGICO Y CONTROLADOR
# ==========================================
@st.cache_resource
def iniciar_servicio():
    # Simulamos una carga de trabajo inicial para que el usuario perciba el proceso
    time.sleep(3) 
    motor = MotorDiagnostico(ruta_datos='datos/dataset_enfermedades.csv')
    cols = motor.entrenar_modelo()
    return motor, cols

with st.spinner("Entrenando red bayesiana con dataset clínico... (3s)"):
    motor, columnas = iniciar_servicio()
mapa_sintomas = {DICCIONARIO_SINTOMAS.get(s, s.replace('_', ' ').title()): s for s in columnas}
opciones_sintomas = list(mapa_sintomas.keys())
# ==========================================
# 3. INTERFAZ PRINCIPAL
# ==========================================
st.title(":material/vital_signs: Sistema Experto de Diagnóstico Bayesiano")
st.divider()

# Diagnóstico Estándar
st.subheader(":material/list_alt: Cuadro Clínico del Paciente")
seleccionados = st.multiselect("Síntomas observados:", options=opciones_sintomas, max_selections=15)
modo_auditoria = st.toggle("Modo Auditoría")

if st.button("Ejecutar Diagnóstico", type="primary"):
    if not seleccionados:
        st.warning("Seleccione síntomas.")
    else:
        # --- CORRECCIÓN: Definimos la evidencia aquí ---
        evidencia = {mapa_sintomas[s]: 1 for s in seleccionados}
        
        if modo_auditoria:
            with st.status("Auditando cálculo matemático...", expanded=True) as status:
                # Usamos 'evidencia' en lugar de 'datos_usuario'
                pasos, resultado_final = motor.calcular_con_auditoria(evidencia)
                
                for paso in pasos:
                    st.write(f"⚙️ {paso}")
                    time.sleep(1.5)
                    
                st.success("Cálculo completado: Aplicando Normalización Bayesiana.")
                # Aquí formateamos el resultado que viene de pgmpy
                resultados_ordenados = motor.formatear_resultados(resultado_final) 
                status.update(label="Inferencia explicada con éxito.", state="complete")
        else:
            resultados_ordenados = motor.calcular_probabilidades(evidencia)
            
        mostrar_resultados_top(resultados_ordenados, DICCIONARIO_ENFERMEDADES)

# Análisis de Inferencia Diagnóstica (Consulta Personalizada)
st.divider()
st.subheader(":material/calculate: Análisis de Inferencia Diagnóstica")
with st.expander("Configurar consulta personalizada"):
    target = st.selectbox("Patología:", options=list(DICCIONARIO_ENFERMEDADES.values()))
    fixed = st.multiselect("Evidencia:", options=opciones_sintomas)
    
    if st.button("Calcular Precisión"):
        evidencia_fija = {mapa_sintomas[s]: 1 for s in fixed}
        mapa_inv = {v.strip(): k for k, v in DICCIONARIO_ENFERMEDADES.items()}
        target_en = mapa_inv.get(target.strip())
        
        try:
            res = motor.inferencia.query(variables=['prognosis'], evidence=evidencia_fija)
            estados = [s.strip() for s in res.state_names['prognosis']]
            prob = res.values[estados.index(target_en.strip())]
            st.metric(f"Probabilidad de {target}", f"{prob*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")

# Análisis de Causalidad Clínica
st.divider()
st.subheader(":material/psychology: Análisis de Causalidad Clínica")
with st.expander("Calcular probabilidad de síntomas dado un Diagnóstico"):
    enf_causal_es = st.selectbox("Si el paciente tuviera:", options=sorted(list(DICCIONARIO_ENFERMEDADES.values())))
    sintomas_causal_es = st.multiselect("¿Probabilidad de presentar estos síntomas?", options=opciones_sintomas)
    
    if st.button("Calcular Probabilidad Generativa"):
        mapa_inv_enf = {v.strip(): k for k, v in DICCIONARIO_ENFERMEDADES.items()}
        enf_en = mapa_inv_enf.get(enf_causal_es.strip())
        sintomas_en = [mapa_sintomas[s] for s in sintomas_causal_es]
        
        try:
            prob = motor.calcular_probabilidad_causal(enf_en, sintomas_en)
            st.metric(f"Probabilidad de los síntomas dado {enf_causal_es}", f"{prob * 100:.6f}%")
        except Exception as e:
            st.error(f"Error: {e}")