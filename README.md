# Agente de enfermedades - Curso: Sistemas Expertos 😄

Agente experto para diagnóstico medico basado en una red bayesiana.

## Requisitos

- Python 3.10 o superior
- pip

## Instalación

1. Crear y activar un entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Instalar las dependencias:

```bash
pip install streamlit pandas pgmpy networkx matplotlib
```

## Ejecutar

```bash
streamlit run aplicacion.py
```

## Datos

El archivo principal de datos está en `datos/dataset_enfermedades.csv`.

Si algo no carga, revisa que la carpeta `imagenes/` y el dataset estén en la ruta correcta :)
