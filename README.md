# WIDS Datathon 2024 - Modelo Modular de Clasificación

## Descripción del Proyecto

Este repositorio contiene un proyecto de Machine Learning con **arquitectura modular** para abordar el desafío del **WiDS Datathon 2024 Challenge 1** de Kaggle. El objetivo es predecir si un paciente recibirá un diagnóstico de cáncer de mama en menos de 90 días (`DiagPeriod90`).

Se han aplicado buenas prácticas de desarrollo, separando la lógica de datos, modelado y orquestación en módulos dedicados. La experimentación se gestiona utilizando **MLflow** para el tracking de métricas y parámetros.

## Estructura del Repositorio

wids-datathon-2024/
├── data/
│   ├── training.csv         # (Debes colocar aquí tu archivo descargado)
│   └── test.csv             # (Debes colocar aquí tu archivo descargado)
├── mlruns/                  # (Se generará automáticamente por MLflow)
├── notebooks/
│   └── Actividad_1_analisis.ipynb  # (Tu notebook original)
├── src/
│   ├── __init__.py          # (Archivo vacío para reconocer la carpeta como paquete)
│   ├── module_data.py       # Lógica de procesamiento de datos
│   └── module_ml.py         # Lógica de modelado y MLflow
├── main.py                  # Script principal de ejecución
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación (basada en la que subiste)

