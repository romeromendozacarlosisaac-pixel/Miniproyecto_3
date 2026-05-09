# =============================================================================
# TELCO CUSTOMER CHURN — SCHEMAS PYDANTIC
# =============================================================================

from enum import Enum
from typing import Literal, List
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUM DE MODELOS DISPONIBLES
# =============================================================================

class ModelName(str, Enum):
    random_forest = "random_forest"
    xgboost       = "xgboost"
    catboost      = "catboost"
    lightgbm      = "lightgbm"


# =============================================================================
# ESQUEMA DE ENTRADA — CLIENTE INDIVIDUAL
# =============================================================================

class CustomerInput(BaseModel):
    """
    Representa los atributos de un cliente para inferencia.
    Los valores admitidos en cada campo categórico replican exactamente
    los valores presentes en el dataset de entrenamiento.
    """

    # ── Demográficas ──────────────────────────────────────────────────────────
    gender:         Literal["Male", "Female"]
    SeniorCitizen:  Literal[0, 1]            = Field(..., description="1 si es adulto mayor, 0 si no.")
    Partner:        Literal["Yes", "No"]
    Dependents:     Literal["Yes", "No"]

    # ── Servicios ─────────────────────────────────────────────────────────────
    tenure:          int   = Field(..., ge=0, le=72, description="Meses de permanencia.")
    PhoneService:    Literal["Yes", "No"]
    MultipleLines:   Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:  Literal["Yes", "No", "No internet service"]
    OnlineBackup:    Literal["Yes", "No", "No internet service"]
    DeviceProtection:Literal["Yes", "No", "No internet service"]
    TechSupport:     Literal["Yes", "No", "No internet service"]
    StreamingTV:     Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    # ── Contrato y facturación ────────────────────────────────────────────────
    Contract:        Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling:Literal["Yes", "No"]
    PaymentMethod:   Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    # ── Cargos ────────────────────────────────────────────────────────────────
    MonthlyCharges: float = Field(..., ge=0.0, description="Cargo mensual en USD.")
    TotalCharges:   float = Field(..., ge=0.0, description="Cargo total acumulado en USD.")

    @field_validator("TotalCharges")
    @classmethod
    def total_gte_monthly(cls, v, info):
        monthly = info.data.get("MonthlyCharges", 0)
        tenure  = info.data.get("tenure", 0)
        if tenure > 0 and v < monthly:
            raise ValueError(
                f"TotalCharges ({v}) no puede ser menor que MonthlyCharges ({monthly}) "
                f"cuando tenure={tenure}."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.50,
                "TotalCharges": 1026.0,
            }
        }
    }


# =============================================================================
# ESQUEMA DE ENTRADA — LOTE
# =============================================================================

class BatchInput(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1, max_length=500)


# =============================================================================
# ESQUEMAS DE SALIDA
# =============================================================================

class PredictionResponse(BaseModel):
    model_used:        str   = Field(..., description="Nombre del modelo usado.")
    churn_prediction:  int   = Field(..., description="0 = No Churn | 1 = Churn")
    churn_probability: float = Field(..., description="Probabilidad de churn [0, 1]")
    risk_label:        str   = Field(..., description="Low / Medium / High")


class BatchResponse(BaseModel):
    total:       int
    predictions: List[PredictionResponse]