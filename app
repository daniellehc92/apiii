from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import re

from sqlalchemy import create_engine
import boto3
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv


load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "default_pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "default_db")
DB_TABLE_NAME = "datos_meteorologicos"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "aemets3")
S3_MODELS_FOLDER = "models"
S3_CLIENT = boto3.client("s3")

TIMESTEPS = 64
PREDICTION_FEATURES = ["tmed", "tmin", "tmax"]


LOCATION_MAP = {
    "A CORUÑA": {"file_prefix": "ACoruna", "indicativo": "1387"},
    "ALBACETE": {"file_prefix": "Albacete", "indicativo": "8178D"},
    "ALICANTE": {"file_prefix": "Alicante", "indicativo": "8025"},
    "ALMERIA": {"file_prefix": "Almeria", "indicativo": "6325O"},
    "ARABA": {"file_prefix": "Araba", "indicativo": "9091R"},
    "ASTURIAS": {"file_prefix": "Asturias", "indicativo": "1249X"},
    "AVILA": {"file_prefix": "Avila", "indicativo": "2444"},
    "BADAJOZ": {"file_prefix": "Badajoz", "indicativo": "4478X"},
    "BARCELONA": {"file_prefix": "Barcelona", "indicativo": "0201D"},
    "BIZKAIA": {"file_prefix": "Bizkaia", "indicativo": "1082"},
    "BURGOS": {"file_prefix": "Burgos", "indicativo": "2331"},
    "CACERES": {"file_prefix": "Caceres", "indicativo": "3469A"},
    "CADIZ": {"file_prefix": "Cadiz", "indicativo": "5973"},
    "CANTABRIA": {"file_prefix": "Cantabria", "indicativo": "1111X"},
    "CASTELLON": {"file_prefix": "Castellon", "indicativo": "8500A"},
    "CEUTA": {"file_prefix": "Ceuta", "indicativo": "5000C"},
    "CIUDAD REAL": {"file_prefix": "CiudadReal", "indicativo": "4121"},
    "CORDOBA": {"file_prefix": "Cordoba", "indicativo": "5402"},
    "CUENCA": {"file_prefix": "Cuenca", "indicativo": "8096"},
    "GIPUZKOA": {"file_prefix": "Gipuzkoa", "indicativo": "1024E"},
    "GIRONA": {"file_prefix": "Girona", "indicativo": "0370E"},
    "GRANADA": {"file_prefix": "Granada", "indicativo": "5530E"},
    "GUADALAJARA": {"file_prefix": "Guadalajara", "indicativo": "3168D"},
    "HUELVA": {"file_prefix": "Huelva", "indicativo": "4642E"},
    "HUESCA": {"file_prefix": "Huesca", "indicativo": "9901X"},
    "ISLAS BALEARES": {"file_prefix": "IslasBaleares", "indicativo": "B236C"},
    "JAEN": {"file_prefix": "Jaen", "indicativo": "5270B"},
    "LA RIOJA": {"file_prefix": "LaRioja", "indicativo": "9170"},
    "LAS PALMAS": {"file_prefix": "LasPalmas", "indicativo": "C658L"},
    "LEON": {"file_prefix": "Leon", "indicativo": "2661"},
    "LLEIDA": {"file_prefix": "Lleida", "indicativo": "9771C"},
    "LUGO": {"file_prefix": "Lugo", "indicativo": "1518A"},
    "MADRID": {"file_prefix": "Madrid", "indicativo": "3195"},
    "MALAGA": {"file_prefix": "Malaga", "indicativo": "6156X"},
    "MELILLA": {"file_prefix": "Melilla", "indicativo": "6000A"},
    "MURCIA": {"file_prefix": "Murcia", "indicativo": "7178I"},
    "NAVARRA": {"file_prefix": "Navarra", "indicativo": "9263D"},
    "OURENSE": {"file_prefix": "Ourense", "indicativo": "1690A"},
    "PALENCIA": {"file_prefix": "Palencia", "indicativo": "2401X"},
    "PONTEVEDRA": {"file_prefix": "Pontevedra", "indicativo": "1484C"},
    "SALAMANCA": {"file_prefix": "Salamanca", "indicativo": "2870"},
    "SANTA CRUZ TENERIFE": {"file_prefix": "SantaCruzTenerife", "indicativo": "C449C"},
    "SEGOVIA": {"file_prefix": "Segovia", "indicativo": "2465"},
    "SEVILLA": {"file_prefix": "Sevilla", "indicativo": "5783"},
    "SORIA": {"file_prefix": "Soria", "indicativo": "2030"},
    "TARRAGONA": {"file_prefix": "Tarragona", "indicativo": "0042Y"},
    "TERUEL": {"file_prefix": "Teruel", "indicativo": "8368U"},
    "TOLEDO": {"file_prefix": "Toledo", "indicativo": "3260B"},
    "VALENCIA": {"file_prefix": "Valencia", "indicativo": "8416"},
    "VALLADOLID": {"file_prefix": "Valladolid", "indicativo": "2422"},
    "ZAMORA": {"file_prefix": "Zamora", "indicativo": "2614"},
    "ZARAGOZA": {"file_prefix": "Zaragoza", "indicativo": "9434P"}
}


INDICATIVO_LIST = ", ".join([f"'{v['indicativo']}' ({k})" for k, v in LOCATION_MAP.items()])
DEFAULT_INDICATIVO = LOCATION_MAP["A CORUÑA"]["indicativo"]

SCHEMA_DEFINITION = f"""
Tabla: {DB_TABLE_NAME}
Columnas:
- fecha (DATE): Fecha del registro (YYYY-MM-DD).
- indicativo (TEXT): Código de la estación. Indicativos soportados: {INDICATIVO_LIST}.
- provincia (TEXT): Provincia.
- tmed (NUMERIC): Temperatura media diaria (°C).
- tmin (NUMERIC): Temperatura mínima diaria (°C).
- tmax (NUMERIC): Temperatura máxima diaria (°C).

INSTRUCCIONES:
1. Crea una consulta SQL que devuelva la información solicitada.
2. Usa el código 'indicativo' para filtrar por ubicación. Si no se especifica ubicación, usa el indicativo '{DEFAULT_INDICATIVO}' ('A CORUÑA').
3. La consulta DEBE ser solo una línea, SIN explicaciones, y SIEMPRE debe empezar con 'SELECT'.
4. Para la "semana pasada" usa 'CURRENT_DATE - INTERVAL '7 days'' como inicio.
"""

try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY no encontrada.")
        
    client = genai.Client(api_key=gemini_api_key)
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    print(f"Advertencia: El cliente de Gemini no se pudo inicializar. NLQ no funcionará. Error: {e}")
    client = None

app = FastAPI(
    title="AEMET API Meteo",
    description="Servicio de predicciones y consulta con datos Aemet.",
    version="1.0.0"
)


def normalize_location_input(location: str) -> str:
    """Normaliza la cadena de ubicación para buscarla en el mapa."""
    location = location.upper().strip()
    replacements = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
    for k, v in replacements.items():
        location = location.replace(k, v)
    return location

def get_location_assets(location: str) -> Dict[str, str]:
    """Valida la ubicación y devuelve las claves S3 y el indicativo."""
    normalized_location = normalize_location_input(location)
    
    if normalized_location not in LOCATION_MAP:
        supported = [k for k in LOCATION_MAP.keys()]
        raise HTTPException(status_code=400, 
                            detail=f"Ubicación '{location}' no soportada. Soportadas: {', '.join(supported)}")
    
    config = LOCATION_MAP[normalized_location]
    
    return {
        "indicativo": config["indicativo"],
        "model_key": f"{S3_MODELS_FOLDER}/{config['file_prefix']}_model.pkl",
        "scaler_key": f"{S3_MODELS_FOLDER}/{config['file_prefix']}_scaler.pkl",
    }

def get_gemini_sql(question: str) -> str:
    """Usa Gemini para convertir la pregunta en una consulta SQL."""
    if not client:
        raise APIError("El cliente de Gemini no está inicializado.")

    prompt = f"{SCHEMA_DEFINITION}\nPregunta: {question}"
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        sql_query = response.text.strip()
        
        if not sql_query.upper().startswith("SELECT"):
            raise ValueError("Gemini no devolvió una consulta SELECT válida.")
        
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        
        return sql_query
    except Exception as e:
        print(f"Error al obtener SQL: {e}")
        raise HTTPException(status_code=500, detail="Error en la conversión de lenguaje natural a SQL.")

def load_ml_assets(model_s3_key: str, scaler_s3_key: str):
    """Descarga y carga el modelo Keras y el scaler de S3 (con caché)."""
    cache = getattr(load_ml_assets, 'cache', {})
    if model_s3_key in cache:
        return cache[model_s3_key]["model"], cache[model_s3_key]["scaler"]

    try:
        print(f"Descargando activos desde S3: {model_s3_key}, {scaler_s3_key}")
        
        model_obj = S3_CLIENT.get_object(Bucket=S3_BUCKET_NAME, Key=model_s3_key)
        model_bytes = model_obj['Body'].read()
        model = pickle.loads(model_bytes)

        scaler_obj = S3_CLIENT.get_object(Bucket=S3_BUCKET_NAME, Key=scaler_s3_key)
        scaler_bytes = scaler_obj['Body'].read()
        scaler = pickle.loads(scaler_bytes)
        
        cache[model_s3_key] = {"model": model, "scaler": scaler}
        setattr(load_ml_assets, 'cache', cache)
        
        return model, scaler

    except Exception as e:
        print(f"Error al cargar los activos ML para {model_s3_key}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al cargar los activos ML para {model_s3_key}. Error: {e}")

def create_sequences(data: np.ndarray, t: int) -> np.ndarray:
    """Prepara los datos históricos en la secuencia de entrada (últimos T días)."""
    return data[-t:].reshape(1, t, len(PREDICTION_FEATURES))


@app.get("/")
def read_root():
    """Endpoint de bienvenida."""
    return {"message": "Welcome to the AEMET Weather Prediction API. Try /ask or /forecast."}

@app.post("/ask", response_model=Dict[str, Any])
async def ask_question(question: str):
    """
    Recibe una pregunta en lenguaje natural, la convierte a SQL con Gemini, 
    consulta la BD y devuelve la respuesta.
    """
    try:
        sql_query = get_gemini_sql(question)
        
        print(f"Generated SQL: {sql_query}")
        
        with engine.connect() as connection:
            df_result = pd.read_sql(sql_query, connection)

        result = df_result.to_dict(orient='records')

        if not result:
            return {"question": question, "sql_query": sql_query, "result": "No se encontraron datos para la consulta."}

        return {"question": question, "sql_query": sql_query, "result": result}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar la consulta SQL o procesar el resultado: {e}")


@app.get("/forecast", response_model=Dict[str, Any])
async def get_forecast(location: str = "A CORUÑA", days: int = 5):
    """
    Predice la temperatura (Tmed, Tmin, Tmax) para los próximos N días 
    basándose en el modelo LSTM entrenado para la ubicación especificada.
    """
    if not 1 <= days <= 30:
        raise HTTPException(status_code=400, detail="El número de días debe ser entre 1 y 30.")

    location_config = get_location_assets(location)
    indicativo = location_config["indicativo"]
    model_s3_key = location_config["model_key"]
    scaler_s3_key = location_config["scaler_key"]

    model, scaler = load_ml_assets(model_s3_key, scaler_s3_key)

    required_history = TIMESTEPS
    query = f"""
    SELECT fecha, {', '.join(PREDICTION_FEATURES)} 
    FROM {DB_TABLE_NAME} 
    WHERE indicativo = '{indicativo}' 
    AND tmed IS NOT NULL AND tmin IS NOT NULL AND tmax IS NOT NULL
    ORDER BY fecha DESC 
    LIMIT {required_history}
    """
    
    try:
        with engine.connect() as connection:
            df_history = pd.read_sql(query, connection)
            df_history = df_history.sort_values(by='fecha', ascending=True).reset_index(drop=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar datos históricos de la BD: {e}")

    if len(df_history) < required_history:
        raise HTTPException(status_code=404, 
                            detail=f"No hay suficientes datos históricos para '{location}' (necesita {required_history}, encontró {len(df_history)}).")

    historical_data = df_history[PREDICTION_FEATURES].values
    scaled_data = scaler.transform(historical_data)
    last_sequence_scaled = create_sequences(scaled_data, TIMESTEPS)
    
    forecast_scaled = []
    current_sequence = last_sequence_scaled
    last_historical_date = df_history['fecha'].iloc[-1]

    for _ in range(days):
        predicted_day_scaled = model.predict(current_sequence, verbose=0)
        forecast_scaled.append(predicted_day_scaled[0])
        
        new_sequence = np.append(current_sequence[:, 1:, :], predicted_day_scaled.reshape(1, 1, len(PREDICTION_FEATURES)), axis=1)
        current_sequence = new_sequence

    forecast_actual = scaler.inverse_transform(np.array(forecast_scaled))
    
    prediction_list = []
    for i in range(days):
        forecast_date = (last_historical_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
        prediction_list.append({
            "fecha": forecast_date,
            "tmed": round(float(forecast_actual[i, 0]), 2), 
            "tmin": round(float(forecast_actual[i, 1]), 2),
            "tmax": round(float(forecast_actual[i, 2]), 2)
        })

    return {
        "location": location.upper(),
        "indicativo": indicativo,
        "days": days,
        "last_historical_date": last_historical_date.strftime('%Y-%m-%d'),
        "forecast": prediction_list
    }
