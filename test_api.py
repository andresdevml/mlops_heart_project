import asyncio
import aiohttp
import pandas as pd
from sqlalchemy import create_engine
import time

API_URL = "http://localhost:8000/predict"
DB_URI = "postgresql+psycopg2://postgres:postgres@localhost:5432/mlops_db"

async def make_request(session, payload):
    async with session.post(API_URL, json=payload) as response:
        return response.status, await response.json()

async def phase_1_accuracy():
    print("\n--- FASE 1: Prueba de Exactitud ---")
    engine = create_engine(DB_URI)
    try:
        df = pd.read_sql('SELECT * FROM test_data LIMIT 5', engine)
    except Exception as e:
        print("Error al leer DB (Asegúrate de haber entrenado el modelo):", e)
        return

    async with aiohttp.ClientSession() as session:
        for _, row in df.iterrows():
            payload = row.drop("target").to_dict()
            status, res = await make_request(session, payload)
            print(f"Status: {status} | Target Real: {row['target']} | Pred: {res}")

async def phase_2_robustness():
    print("\n--- FASE 2: Prueba de Robustez ---")
    bad_payload = {
        "age": "sesenta",  # Error intencional: string en vez de float/int
        "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0, "thal": "fixed"
    }
    async with aiohttp.ClientSession() as session:
        status, res = await make_request(session, bad_payload)
        print(f"Status esperado (422), Status obtenido: {status}")
        print(f"Mensaje obtenido: {res}")

async def phase_3_concurrency():
    print("\n--- FASE 3: Prueba de Concurrencia (100 peticiones) ---")
    payload = {
        "age": 60, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0, "thal": "fixed"
    }
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, payload) for _ in range(100)]
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    successes = sum(1 for status, _ in results if status == 200)
    print(f"Completado en {end_time - start_time:.2f} segundos.")
    print(f"Peticiones exitosas: {successes}/100")

async def main():
    await phase_1_accuracy()
    await phase_2_robustness()
    await phase_3_concurrency()

if __name__ == "__main__":
    asyncio.run(main())
