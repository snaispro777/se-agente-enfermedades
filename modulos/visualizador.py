# modulos/visualizador.py
import networkx as nx
import matplotlib.pyplot as plt

def generar_grafo_bayesiano(columnas_sintomas):
    """Crea una imagen del grafo Naive Bayes."""
    G = nx.DiGraph()
    
    # Nodo padre
    nodo_padre = 'Diagnóstico'
    G.add_node(nodo_padre)
    
    # Nodos hijos
    for sintoma in columnas_sintomas:
        # Limpiamos el nombre para que se vea bien en el grafo
        nombre_legible = sintoma.replace('_', ' ').title()
        G.add_edge(nodo_padre, nombre_legible)
    
    # Configuración del gráfico
    plt.figure(figsize=(12, 12))
    pos = nx.circular_layout(G)
    
    # Dibujar nodos y aristas
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#0b57d0')
    nx.draw_networkx_edges(G, pos, edge_color='#747775', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    
    plt.title("Estructura de la Red Bayesiana (Naive Bayes)")
    plt.axis('off')
    return plt