from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictFloat, StrictInt, StrictStr
import tensorflow as tf
import keras
import asyncio
import asyncpg
import json
import os

app = FastAPI(title="MLOps Heart Disease API")

# Sobrescribir el manejador de errores de validación de FastAPI
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "data de ingreso no esta bien estructruada"},
    )

# Modelo estricto de Pydantic
class HeartDiseaseInput(BaseModel):
    age: StrictFloat | StrictInt
    sex: StrictInt
    cp: StrictInt
    trestbps: StrictFloat | StrictInt
    chol: StrictFloat | StrictInt
    fbs: StrictInt
    restecg: StrictInt
    thalach: StrictFloat | StrictInt
    exang: StrictInt
    oldpeak: StrictFloat | StrictInt
    slope: StrictFloat | StrictInt
    ca: StrictInt
    thal: StrictStr

# Variables globales
model = None
DB_DSN = "postgresql://postgres:postgres@postgres:5432/mlops_db"

@app.on_event("startup")
async def load_model():
    global model
    model_path = "/app/mlruns/heart_disease_model.keras"
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print("Modelo cargado exitosamente.")
    else:
        print("Advertencia: Modelo no encontrado. Entrene el modelo primero.")

async def log_prediction(payload: dict, prediction: float):
    conn = await asyncpg.connect(DB_DSN)
    await conn.execute(
        "INSERT INTO api_logs (request_payload, prediction) VALUES ($1, $2)",
        json.dumps(payload), prediction
    )
    await conn.close()

def run_inference(data_dict: dict):
    # Convertir a tensores como requiere FeatureSpace
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in data_dict.items()}
    # feed-forward directo, sin usar predict()
    pred = model(input_dict, training=False)
    return float(pred[0][0])

@app.post("/predict")
async def predict(data: HeartDiseaseInput):
    if model is None:
        return JSONResponse(status_code=503, content={"detail": "Modelo no disponible"})
    
    data_dict = data.dict()
    
    # Ejecutar inferencia en un hilo separado para no bloquear el Event Loop
    prediction = await asyncio.to_thread(run_inference, data_dict)
    
    # Loggear en la base de datos de manera asíncrona
    asyncio.create_task(log_prediction(data_dict, prediction))
    
    return {"probability_of_heart_disease": prediction}
