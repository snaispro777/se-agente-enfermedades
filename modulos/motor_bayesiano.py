# modulos/motor_bayesiano.py
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator

class MotorDiagnostico:
    def __init__(self, ruta_datos):
        self.ruta_datos = ruta_datos
        self.modelo = None
        self.inferencia = None
        self.columnas_sintomas = []
        
    def entrenar_modelo(self):
        # Carga y limpieza básica del dataset clínico.
        df = pd.read_csv(self.ruta_datos)
        df.columns = df.columns.str.strip()
        
        columna_objetivo = 'prognosis' 
        
        # ==========================================
        # FILTRO DE LIMPIEZA DE DATOS (NUEVO)
        # ==========================================
        # Identificamos columnas que tengan un solo valor único (ej. puros 0s)
        columnas_basura = [col for col in df.columns if df[col].nunique() <= 1]  # Variables sin variación.
        
        # Se eliminan porque no aportan información al modelo.
        df = df.drop(columns=columnas_basura)
        # ==========================================
        
        self.columnas_sintomas = [col for col in df.columns if col != columna_objetivo and not col.startswith('Unnamed')]
        
        enlaces = [(columna_objetivo, sintoma) for sintoma in self.columnas_sintomas]  # Naive Bayes: diagnóstico -> síntomas.
        self.modelo = DiscreteBayesianNetwork(enlaces)
        
        # Estimación Bayesiana de las CPT con suavizado.
        estimador = BayesianEstimator(self.modelo, df)
        tablas_probabilidad = estimador.get_parameters(prior_type="BDeu", equivalent_sample_size=10)
        self.modelo.add_cpds(*tablas_probabilidad)
        
        self.inferencia = VariableElimination(self.modelo)
        return self.columnas_sintomas

    def calcular_probabilidades(self, evidencia_usuario):
        """Ejecuta la inferencia exacta y devuelve los diagnósticos ordenados."""
        if not self.inferencia:
            raise ValueError("El modelo matemático no ha sido inicializado.")
            
        # Consulta exacta sobre el nodo diagnóstico.
        resultado = self.inferencia.query(variables=['prognosis'], evidence=evidencia_usuario)
        
        estados_diagnostico = resultado.state_names['prognosis']
        probabilidades = resultado.values
        
        # Emparejamos cada enfermedad con su probabilidad y ordenamos de mayor a menor
        resultados_ordenados = sorted(
            zip(estados_diagnostico, probabilidades), 
            key=lambda x: x[1], 
            reverse=True
        )
        return resultados_ordenados
    
    # modulos/motor_bayesiano.py (Añade este método a tu clase MotorDiagnostico)

    def calcular_probabilidad_causal(self, nombre_enfermedad, lista_sintomas_ingles):
        """Calcula P(Sintomas | Enfermedad) = Producto de P(S_i | Enfermedad)"""
        probabilidad_total = 1.0
        
        # Limpiamos el nombre recibido
        nombre_limpio = nombre_enfermedad.strip()
        
        # Obtenemos los estados del modelo y los limpiamos también
        estados_enfermedad = [s.strip() for s in self.modelo.states['prognosis']]  # Lista de patologías del nodo raíz.
        
        if nombre_limpio not in estados_enfermedad:
            raise ValueError(f"La enfermedad '{nombre_limpio}' no fue encontrada en los estados del modelo. "
                             f"Estados detectados: {estados_enfermedad[:5]}...")

        idx_enfermedad = estados_enfermedad.index(nombre_limpio)
        
        for sintoma in lista_sintomas_ingles:
            cpd = self.modelo.get_cpds(sintoma)
            # Tomamos la probabilidad de presencia del síntoma dado el diagnóstico.
            prob = cpd.get_values()[1, idx_enfermedad]
            probabilidad_total *= prob
            
        return probabilidad_total

    def obtener_desglose_inferencia(self, evidencia_usuario):
        """Devuelve el detalle multiplicativo de P(prognosis | evidencia)."""
        if not self.modelo:
            raise ValueError("El modelo matemático no ha sido inicializado.")

        # Priors del nodo diagnóstico.
        cpd_prognosis = self.modelo.get_cpds('prognosis')
        estados_enfermedad = [s.strip() for s in cpd_prognosis.state_names['prognosis']]
        prioridades = cpd_prognosis.values

        desglose = []
        for idx_enfermedad, nombre_enfermedad in enumerate(estados_enfermedad):
            # El acumulado empieza en el prior de cada patología.
            acumulado = float(prioridades[idx_enfermedad])
            pasos = [{
                'factor': 'P(prognosis)',
                'detalle': nombre_enfermedad,
                'probabilidad': float(prioridades[idx_enfermedad]),
                'acumulado': acumulado,
            }]

            for sintoma, valor in evidencia_usuario.items():
                # Cada síntoma aporta una probabilidad condicional distinta.
                cpd_sintoma = self.modelo.get_cpds(sintoma)
                probabilidad = float(cpd_sintoma.get_values()[int(valor), idx_enfermedad])
                acumulado *= probabilidad
                pasos.append({
                    'factor': f"P({sintoma}={valor} | prognosis={nombre_enfermedad})",
                    'detalle': sintoma,
                    'probabilidad': probabilidad,
                    'acumulado': acumulado,
                })

            desglose.append({
                'enfermedad': nombre_enfermedad,
                'puntaje': acumulado,
                'pasos': pasos,
            })

        total = sum(item['puntaje'] for item in desglose)  # Normalización final sobre todas las patologías.
        for item in desglose:
            item['posterior'] = item['puntaje'] / total if total else 0.0

        desglose.sort(key=lambda x: x['posterior'], reverse=True)
        return desglose

    def obtener_desglose_causalidad(self, nombre_enfermedad, lista_sintomas_ingles):
        """Devuelve el detalle multiplicativo de P(sintomas | enfermedad)."""
        if not self.modelo:
            raise ValueError("El modelo matemático no ha sido inicializado.")

        # Se valida y limpia el diagnóstico elegido.
        nombre_limpio = nombre_enfermedad.strip()
        estados_enfermedad = [s.strip() for s in self.modelo.states['prognosis']]

        if nombre_limpio not in estados_enfermedad:
            raise ValueError(
                f"La enfermedad '{nombre_limpio}' no fue encontrada en los estados del modelo. "
                f"Estados detectados: {estados_enfermedad[:5]}..."
            )

        idx_enfermedad = estados_enfermedad.index(nombre_limpio)
        acumulado = 1.0
        pasos = []

        for sintoma in lista_sintomas_ingles:
            # Causalidad directa: síntoma condicionado al diagnóstico.
            cpd_sintoma = self.modelo.get_cpds(sintoma)
            probabilidad = float(cpd_sintoma.get_values()[1, idx_enfermedad])
            acumulado *= probabilidad
            pasos.append({
                'factor': f"P({sintoma}=1 | prognosis={nombre_limpio})",
                'detalle': sintoma,
                'probabilidad': probabilidad,
                'acumulado': acumulado,
            })

        return {
            'enfermedad': nombre_limpio,
            'puntaje': acumulado,
            'pasos': pasos,
        }
    def calcular_con_auditoria(self, evidencia):
        """Ejecuta la inferencia paso a paso."""
        bitacora = []
        
        # 1. Definimos la consulta
        query = self.inferencia.query(variables=['prognosis'], evidence=evidencia, show_progress=False)
        
        # PGMpy no expone los factores internos, así que registramos una bitácora simplificada.
        bitacora.append("Factor inicial: P(prognosis)")
        for sintoma, valor in evidencia.items():
            bitacora.append(f"Multiplicando factor: P({sintoma} | prognosis) con evidencia={valor}")
            
        return bitacora, query
    def formatear_resultados(self, resultado_query):
        """
        Convierte el objeto DiscreteFactor de pgmpy en una lista de tuplas 
        (enfermedad, probabilidad) ordenada de mayor a menor.
        """
        # Convierte el resultado de pgmpy a una estructura simple para la UI.
        estados = resultado_query.state_names['prognosis']
        probabilidades = resultado_query.values
        
        # Combinamos ambos en una lista de tuplas
        lista_resultados = list(zip(estados, probabilidades))
        
        # Ordenamos de mayor a menor probabilidad para mostrar el ranking.
        lista_resultados.sort(key=lambda x: x[1], reverse=True)
        
        return lista_resultados

    # Convierte la CPT de un síntoma en tabla legible para Streamlit.
    def obtener_cpt_dataframe(self, sintoma_ingles, diccionario_enfermedades):
        """
        Extrae la CPT de un síntoma desde pgmpy y la convierte en un DataFrame 
        con los nombres de las enfermedades traducidos al español.
        """
        cpd = self.modelo.get_cpds(sintoma_ingles)
        valores = cpd.get_values()  # Matriz con ausencia/presencia del síntoma.
        estados_enfermedad = self.modelo.states['prognosis']
        
        # Traducción de columnas para la interfaz en español.
        columnas_espanol = [
            diccionario_enfermedades.get(enf.strip(), enf.strip()) 
            for enf in estados_enfermedad
        ]
        
        # Dos filas: ausencia y presencia.
        df_cpt = pd.DataFrame(
            valores,
            index=["Probabilidad de AUSENCIA (0)", "Probabilidad de PRESENCIA (1)"],
            columns=columnas_espanol
        )
        
        return df_cpt