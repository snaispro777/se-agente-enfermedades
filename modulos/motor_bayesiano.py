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
        df = pd.read_csv(self.ruta_datos)
        df.columns = df.columns.str.strip()
        
        columna_objetivo = 'prognosis' 
        
        # ==========================================
        # FILTRO DE LIMPIEZA DE DATOS (NUEVO)
        # ==========================================
        # Identificamos columnas que tengan un solo valor único (ej. puros 0s)
        columnas_basura = [col for col in df.columns if df[col].nunique() <= 1]
        
        # Eliminamos esas variables del dataset antes de entrenar la red
        df = df.drop(columns=columnas_basura)
        # ==========================================
        
        self.columnas_sintomas = [col for col in df.columns if col != columna_objetivo and not col.startswith('Unnamed')]
        
        enlaces = [(columna_objetivo, sintoma) for sintoma in self.columnas_sintomas]
        self.modelo = DiscreteBayesianNetwork(enlaces)
        
        estimador = BayesianEstimator(self.modelo, df)
        tablas_probabilidad = estimador.get_parameters(prior_type="BDeu", equivalent_sample_size=10)
        self.modelo.add_cpds(*tablas_probabilidad)
        
        self.inferencia = VariableElimination(self.modelo)
        return self.columnas_sintomas

    def calcular_probabilidades(self, evidencia_usuario):
        """Ejecuta la inferencia exacta y devuelve los diagnósticos ordenados."""
        if not self.inferencia:
            raise ValueError("El modelo matemático no ha sido inicializado.")
            
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
        estados_enfermedad = [s.strip() for s in self.modelo.states['prognosis']]
        
        if nombre_limpio not in estados_enfermedad:
            raise ValueError(f"La enfermedad '{nombre_limpio}' no fue encontrada en los estados del modelo. "
                             f"Estados detectados: {estados_enfermedad[:5]}...")

        idx_enfermedad = estados_enfermedad.index(nombre_limpio)
        
        for sintoma in lista_sintomas_ingles:
            cpd = self.modelo.get_cpds(sintoma)
            # Obtenemos la probabilidad P(Sintoma=1 | Enfermedad)
            prob = cpd.get_values()[1, idx_enfermedad]
            probabilidad_total *= prob
            
        return probabilidad_total
    def calcular_con_auditoria(self, evidencia):
        """Ejecuta la inferencia paso a paso."""
        bitacora = []
        
        # 1. Definimos la consulta
        query = self.inferencia.query(variables=['prognosis'], evidence=evidencia, show_progress=False)
        
        # Nota: PGMpy oculta las multiplicaciones internas. 
        # Para exponerlas, simulamos la explicación del Teorema de Bayes
        bitacora.append("Factor inicial: P(prognosis)")
        for sintoma, valor in evidencia.items():
            bitacora.append(f"Multiplicando factor: P({sintoma} | prognosis) con evidencia={valor}")
            
        return bitacora, query
    def formatear_resultados(self, resultado_query):
        """
        Convierte el objeto DiscreteFactor de pgmpy en una lista de tuplas 
        (enfermedad, probabilidad) ordenada de mayor a menor.
        """
        # Obtenemos los nombres de las enfermedades (estados) y sus valores
        estados = resultado_query.state_names['prognosis']
        probabilidades = resultado_query.values
        
        # Combinamos ambos en una lista de tuplas
        lista_resultados = list(zip(estados, probabilidades))
        
        # Ordenamos de mayor a menor probabilidad
        lista_resultados.sort(key=lambda x: x[1], reverse=True)
        
        return lista_resultados