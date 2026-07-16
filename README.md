# AutoValor

AutoValor es una aplicacion web educativa para analizar un dataset de autos usados y estimar precios orientativos mediante regresion lineal multivariable.

El proyecto se construye por fases. En esta version ya existe la estructura base, un dataset sintetico reproducible, el procesamiento exploratorio con Pandas y el flujo web de entrenamiento y prediccion con regresion lineal multivariable.

## Uso educativo

Los datos incluidos son ficticios y fueron generados con una semilla fija. Las predicciones futuras del modelo seran solo orientativas y no deben interpretarse como una tasacion comercial real.

## Clases aplicadas

- Analisis de datos con Pandas.
- Regresion lineal multivariable.

## Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- pickle, de la biblioteca estandar de Python

## Estructura inicial

```text
.
+-- app.py
+-- PROJECT_CONTEXT.md
+-- README.md
+-- requirements.txt
+-- data/
|   +-- autos_usados.csv
+-- artifacts/
|   +-- .gitkeep
+-- scripts/
|   +-- generar_dataset.py
+-- src/
|   +-- __init__.py
|   +-- data_processing.py
|   +-- exceptions.py
|   +-- model_service.py
+-- tests/
    +-- test_data_processing.py
    +-- test_model_service.py
```

## Instalacion

```bash
python -m venv .venv
pip install -r requirements.txt
```

En Windows PowerShell, la activacion del entorno virtual es:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Ejecucion local

```bash
python -m streamlit run app.py
```

## Dataset

El CSV de ejemplo se encuentra en `data/autos_usados.csv`. Para regenerarlo:

```bash
python scripts/generar_dataset.py
```

Columnas requeridas:

| Columna | Descripcion |
|---|---|
| `marca` | Marca del vehiculo |
| `modelo` | Modelo del vehiculo |
| `anio` | Anio de fabricacion |
| `kilometraje` | Kilometros recorridos |
| `combustible` | Tipo de combustible |
| `transmision` | Tipo de transmision |
| `precio_usd` | Precio objetivo en dolares |

## Estado actual

- Fase 1: estructura minima del proyecto.
- Fase 2: dataset educativo sintetico y reproducible.
- Fase 3: lectura, validacion, limpieza, metricas, filtros, agregaciones y graficos con Pandas.
- Fase 4: pipeline de regresion lineal multivariable, evaluacion, prediccion y persistencia con pickle.
- Fase 5: integracion web de carga, analisis, entrenamiento, metricas, persistencia y formulario de prediccion.

## Limitaciones conocidas

Faltan pruebas finales de calidad, documentacion completa de entrega y verificacion de despliegue.

## Link a la aplicacion desplegada

https://tp-final-python-n5soq7svzvzjkrur8fkblc.streamlit.app/

## Autor

Maxi Borachoc
