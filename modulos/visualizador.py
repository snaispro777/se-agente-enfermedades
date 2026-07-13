# modulos/visualizador.py
import networkx as nx
import matplotlib.pyplot as plt


def _formatear_etiqueta_nodo(base, probabilidad=None, acumulado=None):
    # Combina nombre y métricas en una sola etiqueta legible.
    partes = [base]
    if probabilidad is not None:
        partes.append(f"P={probabilidad * 100:.2f}%")
    if acumulado is not None:
        partes.append(f"Acum={acumulado * 100:.2f}%")
    return "\n".join(partes)

def generar_grafo_bayesiano(columnas_sintomas):
    """Crea una imagen del grafo Naive Bayes."""
    G = nx.DiGraph()
    
    # Nodo raíz del modelo.
    nodo_padre = 'Diagnóstico'
    G.add_node(nodo_padre)
    
    # Cada síntoma cuelga del diagnóstico.
    for sintoma in columnas_sintomas:
        # Se formatea el nombre para mostrarlo de forma más humana.
        nombre_legible = sintoma.replace('_', ' ').title()
        G.add_edge(nodo_padre, nombre_legible)
    
    # Disposición circular simple para una vista limpia.
    plt.figure(figsize=(12, 12))
    pos = nx.circular_layout(G)
    
    # Estilo visual básico del grafo.
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#0b57d0')
    nx.draw_networkx_edges(G, pos, edge_color='#747775', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    
    plt.title("Estructura de la Red Bayesiana (Naive Bayes)")
    plt.axis('off')
    return plt


def generar_grafo_consulta(
    columnas_sintomas,
    sintomas_resaltados=None,
    nodo_consulta='prognosis',
    diccionario_sintomas=None,
    etiqueta_consulta='Diagnóstico',
    detalle_nodos=None,
):
    """Crea una imagen legible del fragmento consultado de la red bayesiana."""
    # Si no hay resaltados, se muestran todos los síntomas visibles.
    sintomas_resaltados = list(dict.fromkeys(sintomas_resaltados or []))
    diccionario_sintomas = diccionario_sintomas or {}
    detalle_nodos = detalle_nodos or []
    # Mapa rápido para acceder al detalle matemático por síntoma.
    detalle_por_sintoma = {item.get('detalle'): item for item in detalle_nodos if item.get('detalle')}

    G = nx.DiGraph()
    G.add_node(nodo_consulta)

    sintomas_visibles = [s for s in columnas_sintomas if s in sintomas_resaltados]
    if not sintomas_visibles:
        sintomas_visibles = list(columnas_sintomas)

    for sintoma in sintomas_visibles:
        G.add_edge(nodo_consulta, sintoma)

    plt.figure(figsize=(12, max(4, 1.25 * len(sintomas_visibles))))

    # Posición manual para evitar cruces y dejar el nodo consulta al centro.
    pos = {nodo_consulta: (0.0, 0.0)}
    if sintomas_visibles:
        espacio = max(1.0, len(sintomas_visibles) - 1)
        inicio = espacio / 2.0
        for indice, sintoma in enumerate(sintomas_visibles):
            pos[sintoma] = (2.5, inicio - indice)

    # Colores y tamaños separados para distinguir consulta y síntomas.
    colores_nodos = []
    tamanios_nodos = []
    for nodo in G.nodes():
        if nodo == nodo_consulta:
            colores_nodos.append('#0b57d0')
            tamanios_nodos.append(2200)
        else:
            colores_nodos.append('#fbbc04')
            tamanios_nodos.append(1200)

    etiquetas = {nodo_consulta: etiqueta_consulta}
    for sintoma in sintomas_visibles:
        nombre_legible = diccionario_sintomas.get(sintoma, sintoma.replace('_', ' ').title())
        detalle = detalle_por_sintoma.get(sintoma)
        if detalle:
            etiquetas[sintoma] = _formatear_etiqueta_nodo(
                nombre_legible,
                probabilidad=detalle.get('probabilidad'),
                acumulado=detalle.get('acumulado'),
            )
        else:
            etiquetas[sintoma] = nombre_legible

    nx.draw_networkx_nodes(G, pos, node_size=tamanios_nodos, node_color=colores_nodos)
    nx.draw_networkx_edges(G, pos, edge_color='#747775', arrows=True, arrowsize=18, width=1.8)
    nx.draw_networkx_labels(G, pos, labels=etiquetas, font_size=10, font_color='black')

    plt.title("Fragmento de la red usado en la consulta")
    plt.axis('off')
    plt.tight_layout()
    return plt


def generar_grafo_top3(resultados_top3, diccionario_enfermedades=None, etiqueta_consulta='Diagnóstico', conteo_evidencia=None):
    """Crea un grafo simple con el diagnóstico central y los tres resultados más probables."""
    diccionario_enfermedades = diccionario_enfermedades or {}

    # Nodo central y tres ramas principales.
    G = nx.DiGraph()
    nodo_consulta = 'prognosis'
    G.add_node(nodo_consulta)

    resultados_visibles = list(resultados_top3[:3])
    for enfermedad_raw, prob in resultados_visibles:
        enfermedad_limpia = enfermedad_raw.strip()
        enfermedad_es = diccionario_enfermedades.get(enfermedad_limpia, enfermedad_limpia)
        nodo_resultado = f"{enfermedad_es}\n{prob * 100:.2f}%"
        G.add_edge(nodo_consulta, nodo_resultado)

    plt.figure(figsize=(13, 6))

    # Posiciones fijas para que el Top 3 quede alineado.
    pos = {nodo_consulta: (0.0, 0.0)}
    separacion = 2.6
    inicio_x = -separacion
    for indice, (enfermedad_raw, prob) in enumerate(resultados_visibles):
        enfermedad_limpia = enfermedad_raw.strip()
        enfermedad_es = diccionario_enfermedades.get(enfermedad_limpia, enfermedad_limpia)
        nodo_resultado = f"{enfermedad_es}\n{prob * 100:.2f}%"
        pos[nodo_resultado] = (inicio_x + indice * separacion, -1.8)

    colores_nodos = []
    tamanios_nodos = []
    for nodo in G.nodes():
        if nodo == nodo_consulta:
            colores_nodos.append('#0b57d0')
            tamanios_nodos.append(2600)
        else:
            colores_nodos.append('#34a853')
            tamanios_nodos.append(1900)

    if conteo_evidencia is not None:
        etiquetas = {nodo_consulta: f"{etiqueta_consulta}\nEvidencia: {conteo_evidencia} síntomas"}
    else:
        etiquetas = {nodo_consulta: etiqueta_consulta}
    for enfermedad_raw, prob in resultados_visibles:
        enfermedad_limpia = enfermedad_raw.strip()
        enfermedad_es = diccionario_enfermedades.get(enfermedad_limpia, enfermedad_limpia)
        nodo_resultado = f"{enfermedad_es}\n{prob * 100:.2f}%"
        etiquetas[nodo_resultado] = _formatear_etiqueta_nodo(enfermedad_es, probabilidad=prob)

    nx.draw_networkx_nodes(G, pos, node_size=tamanios_nodos, node_color=colores_nodos)
    nx.draw_networkx_edges(G, pos, edge_color='#747775', arrows=True, arrowsize=18, width=2.0)
    nx.draw_networkx_labels(G, pos, labels=etiquetas, font_size=10, font_color='black')

    plt.title("Top 3 del diagnóstico sobre la red bayesiana")
    plt.axis('off')
    plt.tight_layout()
    return plt