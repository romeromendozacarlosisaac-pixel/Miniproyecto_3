# =============================================================================
# TELCO CUSTOMER CHURN — SERVICIO DE INFERENCIA FastAPI
# =============================================================================

import os
# import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Dict

from app.schemas import (
    CustomerInput, PredictionResponse,
    BatchInput, BatchResponse,
    ModelName,
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
ALL_CAT = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]

# =============================================================================
# CONFIGURACIÓN — rutas de los 4 modelos
# =============================================================================

MODELS_DIR = os.getenv("MODELS_DIR", "data/models")

MODEL_FILES: Dict[str, str] = {
    "random_forest": "rf_best.pkl",
    "xgboost": "xgb_best.pkl",
    "catboost": "cb_best.pkl",
    "lightgbm": "lgbm_best.pkl",
}

# Diccionario global que almacena los 4 modelos cargados
MODELS: Dict[str, object] = {}


# =============================================================================
# LIFESPAN — carga todos los modelos al arrancar
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    # global MODELS
    failed = []
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        try:
            MODELS[name] = joblib.load(path)
            print(f"[INFO] Modelo '{name}' cargado desde: {path}")
        except FileNotFoundError:
            print(f"[ERROR] No se encontró: {path}")
            failed.append(name)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar '{name}': {e}")
            failed.append(name)

    if len(failed) == len(MODEL_FILES):
        raise RuntimeError("No se pudo cargar ningún modelo. Verifica data/models/.")
    if failed:
        print(f"[WARN] Modelos no disponibles: {failed}")
    yield
    MODELS.clear()
    print("[INFO] Modelos liberados de memoria.")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Telco Customer Churn API",
    description=(
        "Servicio de inferencia para predicción de churn. "
        "Soporta 4 modelos: random_forest, xgboost, catboost, lightgbm."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# =============================================================================
# UTILIDADES
# =============================================================================

FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def get_risk_label(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


def input_to_dataframe(data: CustomerInput) -> pd.DataFrame:
    row = data.model_dump()
    df = pd.DataFrame([row])[FEATURE_ORDER]
    for col in ALL_CAT:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def batch_to_dataframe(data: BatchInput) -> pd.DataFrame:
    rows = [c.model_dump() for c in data.customers]
    df = pd.DataFrame(rows)[FEATURE_ORDER]
    for col in ALL_CAT:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def resolve_model(model_name: str):
    """Retorna el modelo solicitado o lanza 503/404 según corresponda."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="Ningún modelo disponible.")
    if model_name not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Modelo '{model_name}' no disponible. "
                f"Modelos activos: {list(MODELS.keys())}"
            ),
        )
    return MODELS[model_name]


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "Telco Churn Inference API",
        "version": "2.0.0",
        "modelos_disponibles": list(MODELS.keys()),
    }


@app.get("/health", tags=["Health"])
def health():
    """Verifica que al menos un modelo esté cargado."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="Sin modelos disponibles.")
    return {
        "status": "healthy",
        "modelos_cargados": list(MODELS.keys()),
        "total_modelos": len(MODELS),
    }


@app.get("/models", tags=["Health"])
def list_models():
    """Lista los modelos disponibles para inferencia."""
    return {"modelos_disponibles": list(MODELS.keys())}


@app.post("/predict/{model_name}", response_model=PredictionResponse, tags=["Inference"])
def predict(model_name: ModelName, customer: CustomerInput):
    """
    Predice el churn de un cliente usando el modelo indicado en la URL.

    - **model_name**: `random_forest` | `xgboost` | `catboost` | `lightgbm`
    """
    model = resolve_model(model_name.value)
    try:
        df = input_to_dataframe(customer)
        prob = float(model.predict_proba(df)[0][1])
        return PredictionResponse(
            model_used=model_name.value,
            churn_prediction=int(prob >= 0.5),
            churn_probability=round(prob, 4),
            risk_label=get_risk_label(prob),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en inferencia: {str(e)}")


@app.post(
    "/predict/batch/{model_name}",
    response_model=BatchResponse,
    tags=["Inference"],
)
def predict_batch(model_name: ModelName, batch: BatchInput):
    """
    Predice el churn de hasta 500 clientes usando el modelo indicado.

    - **model_name**: `random_forest` | `xgboost` | `catboost` | `lightgbm`
    """
    if len(batch.customers) > 500:
        raise HTTPException(status_code=400, detail="Máximo 500 clientes por lote.")

    model = resolve_model(model_name.value)
    try:
        df = batch_to_dataframe(batch)
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)

        predictions = [
            PredictionResponse(
                model_used=model_name.value,
                churn_prediction=int(pred),
                churn_probability=round(float(prob), 4),
                risk_label=get_risk_label(float(prob)),
            )
            for pred, prob in zip(preds, probs)
        ]

        return BatchResponse(total=len(predictions), predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en inferencia batch: {str(e)}")
