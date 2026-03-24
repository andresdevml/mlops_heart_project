# 🫀 Heart Disease Prediction - End-to-End MLOps Architecture

Este proyecto implementa una arquitectura completa de **Machine Learning Operations (MLOps)** diseñada para predecir la probabilidad de enfermedad cardíaca en pacientes. El sistema abarca todo el ciclo de vida del modelo: desde el almacenamiento y preprocesamiento de datos, hasta el entrenamiento orquestado, el registro de modelos y el despliegue en una API asíncrona de alto rendimiento.

Toda la infraestructura está 100% contenedorizada, garantizando portabilidad absoluta y recuperación ante desastres (Disaster Recovery) en cuestión de minutos utilizando una imagen de Docker personalizada alojada en Docker Hub.

## 🏗️ Arquitectura y Tecnologías

El ecosistema está compuesto por microservicios orquestados mediante **Docker Compose**, integrando las siguientes herramientas estándar de la industria:

* **Apache Airflow**: Encargado de la orquestación. Se utiliza para ejecutar el Pipeline de ML (DAG) que dispara el primer entrenamiento del modelo de forma automatizada.
* **PostgreSQL**: Actúa como la *Single Source of Truth*. Almacena los datos de entrenamiento crudos y registra de forma persistente todos los *logs* de las predicciones de la API para auditoría y futuro reentrenamiento.
* **MLflow**: Funciona como el *Model Registry* y sistema de *Tracking*. Guarda las distintas versiones de los modelos generados (`.keras`), sus métricas de rendimiento (accuracy, loss) y sus hiperparámetros.
* **FastAPI**: Capa de servicio (*Serving Layer*). Proporciona un *endpoint* REST asíncrono y de baja latencia para la inferencia en tiempo real, con validación estricta de datos de entrada mediante Pydantic.
* **Docker & Docker Hub**: Todo el sistema se despliega a partir de contenedores, utilizando una imagen maestra remota (`huachitech/mlops-heart-api:v1.0`).

## 📂 Estructura del Proyecto

```text
mlops_heart_project/
├── app/                  # Lógica de la API (FastAPI) y esquemas de validación
│   └── main.py
├── dags/                 # Definición de grafos acíclicos dirigidos para Airflow
│   └── mlops_pipeline.py
├── scripts/              # Lógica de entrenamiento y evaluación del modelo
│   └── train.py
├── docker-compose.yml    # Orquestador de la infraestructura (Postgres, Airflow, MLflow, API)
├── Dockerfile            # Receta de construcción de la imagen maestra
├── init.sql              # Script de inicialización de la base de datos (tablas y esquemas)
├── requirements.txt      # Dependencias de Python
└── test_api.py           # Pruebas de integración, exactitud, robustez y concurrencia
```

## 🚀 Despliegue Rápido (Disaster Recovery)

Para levantar esta infraestructura desde cero en cualquier servidor Ubuntu/Linux, solo necesitas clonar este repositorio y ejecutar Docker Compose:

```bash
git clone [https://github.com/andresdevml/mlops_heart_project.git](https://github.com/andresdevml/mlops_heart_project.git)
cd mlops_heart_project
docker compose up -d
```
*Los contenedores descargarán la imagen pre-compilada desde Docker Hub y el sistema estará operativo en pocos minutos.*

## 🧪 Pruebas de la API (Testing)

La API fue diseñada para ser asíncrona, aislando el cálculo del modelo (`asyncio.to_thread`) para no bloquear el servidor. Se realizaron pruebas exhaustivas de exactitud frente a la base de datos, robustez (validación de datos anómalos que retornan `Status 422`) y estrés (concurrencia).

A continuación, los resultados de ejecución del script `test_api.py` desde una red remota hacia el VPS en producción:

```text
--- FASE 1: Prueba de Exactitud Remota ---
Status: 200 | Target Real: 0 | Pred: {'probability_of_heart_disease': 0.019256984815001488}
Status: 200 | Target Real: 1 | Pred: {'probability_of_heart_disease': 0.5188168287277222}
Status: 200 | Target Real: 0 | Pred: {'probability_of_heart_disease': 0.23880812525749207}
Status: 200 | Target Real: 0 | Pred: {'probability_of_heart_disease': 0.13055285811424255}
Status: 200 | Target Real: 0 | Pred: {'probability_of_heart_disease': 0.019010480493307114}

--- FASE 2: Prueba de Robustez Remota ---
Status esperado (422), Status obtenido: 422
Mensaje obtenido: {'detail': 'data de ingreso no esta bien estructruada'}

--- FASE 3: Prueba de Concurrencia Remota (100 peticiones) ---
Completado en 2.57 segundos.
Peticiones exitosas: 100/100
```
*(El test de concurrencia completó 100 peticiones de inferencia simultáneas en 2.57 segundos, incluyendo la latencia de red).*
