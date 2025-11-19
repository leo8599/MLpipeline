# WIDS Datathon 2024 - Modelo Modular de Clasificación

## Descripción del Proyecto

Este repositorio contiene un proyecto de Machine Learning con **arquitectura modular** para abordar el desafío del **WiDS Datathon 2024 Challenge 1** de Kaggle. El objetivo es predecir si un paciente recibirá un diagnóstico de cáncer de mama en menos de 90 días (`DiagPeriod90`).

Se han aplicado buenas prácticas de desarrollo, separando la lógica de datos, modelado y orquestación en módulos dedicados. La experimentación se gestiona utilizando **MLflow** para el tracking de métricas y parámetros.

## Estructura del Repositorio

```
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
```

## Instalación y Configuración

### Descargar los Datos

1. Instala la API de Kaggle:
```bash
pip install kaggle
```

2. Configura tus credenciales de Kaggle siguiendo las [instrucciones oficiales](https://github.com/Kaggle/kaggle-api#api-credentials)

3. Descarga los datos del concurso:
```bash
kaggle competitions download -c widsdatathon2024-challenge1
```

4. Extrae los archivos en la carpeta `data/`:
```bash
mkdir -p data
unzip widsdatathon2024-challenge1.zip -d data/
```

