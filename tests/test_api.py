# =============================================================================
# TELCO CUSTOMER CHURN — PRUEBAS UNITARIAS DE LA API
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Fixture: cliente válido de ejemplo ────────────────────────────────────────
VALID_CUSTOMER = {
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

LOW_RISK_CUSTOMER = {
    **VALID_CUSTOMER,
    "Contract": "Two year",
    "tenure": 60,
    "MonthlyCharges": 20.0,
    "TotalCharges": 1200.0,
}

MODEL_NAME = "random_forest"


# =============================================================================
# SETUP DEL CLIENTE DE PRUEBA CON MODELO MOCKEADO
# =============================================================================

@pytest.fixture
def mock_model():
    """Crea un modelo falso que retorna probabilidades controladas."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.3, 0.7]])
    return m


@pytest.fixture
def client(mock_model):
    """TestClient con MODELS mockeado para simular modelo cargado."""
    from app.api import app
    mock_models = {MODEL_NAME: mock_model}
    with patch("app.api.MODELS", mock_models):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_no_models():
    """TestClient sin ningún modelo cargado (MODELS vacío)."""
    from app.api import app
    with patch("app.api.MODELS", {}):
        with TestClient(app) as c:
            yield c


# =============================================================================
# PRUEBAS DE HEALTH
# =============================================================================

class TestHealth:

    def test_root_returns_ok(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_model_not_loaded(self, client_no_models):
        """Sin modelo cargado debe retornar 503."""
        response = client_no_models.get("/health")
        assert response.status_code == 503

    def test_list_models(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        assert MODEL_NAME in response.json()["modelos_disponibles"]


# =============================================================================
# PRUEBAS DE /predict/{model_name}
# =============================================================================

class TestPredict:

    def test_predict_valid_customer_returns_200(self, client):
        response = client.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        response = client.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_label" in data
        assert "model_used" in data

    def test_predict_churn_prediction_is_binary(self, client):
        response = client.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
        pred = response.json()["churn_prediction"]
        assert pred in [0, 1]

    def test_predict_probability_in_range(self, client):
        response = client.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_label_high(self, client):
        """Con 70% de probabilidad debe retornar High."""
        response = client.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
        assert response.json()["risk_label"] == "High"

    def test_predict_risk_label_low(self, mock_model, client):
        """Con 20% de probabilidad debe retornar Low."""
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        response = client.post(f"/predict/{MODEL_NAME}", json=LOW_RISK_CUSTOMER)
        assert response.json()["risk_label"] == "Low"

    def test_predict_risk_label_medium(self, mock_model, client):
        """Con 50% de probabilidad debe retornar Medium."""
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        response = client.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
        assert response.json()["risk_label"] == "Medium"

    def test_predict_missing_field_returns_422(self, client):
        """Un campo requerido faltante debe retornar 422."""
        incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        response = client.post(f"/predict/{MODEL_NAME}", json=incomplete)
        assert response.status_code == 422

    def test_predict_invalid_gender_returns_422(self, client):
        bad = {**VALID_CUSTOMER, "gender": "Unknown"}
        response = client.post(f"/predict/{MODEL_NAME}", json=bad)
        assert response.status_code == 422

    def test_predict_negative_tenure_returns_422(self, client):
        bad = {**VALID_CUSTOMER, "tenure": -5}
        response = client.post(f"/predict/{MODEL_NAME}", json=bad)
        assert response.status_code == 422

    def test_predict_model_unavailable_returns_503(self, client_no_models):
        """Sin modelos cargados debe retornar 503."""
        response = client_no_models.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
        assert response.status_code == 503

    def test_predict_unknown_model_returns_422(self, client):
        """Un modelo fuera del enum debe retornar 422."""
        response = client.post("/predict/nonexistent_model", json=VALID_CUSTOMER)
        assert response.status_code == 422


# =============================================================================
# PRUEBAS DE /predict/batch/{model_name}
# =============================================================================

class TestPredictBatch:

    def test_batch_valid_returns_200(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        payload = {"customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]}
        response = client.post(f"/predict/batch/{MODEL_NAME}", json=payload)
        assert response.status_code == 200

    def test_batch_total_matches_input(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        payload = {"customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]}
        data = client.post(f"/predict/batch/{MODEL_NAME}", json=payload).json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_list_returns_422(self, client):
        response = client.post(f"/predict/batch/{MODEL_NAME}", json={"customers": []})
        assert response.status_code == 422

    def test_batch_model_unavailable_returns_503(self, client_no_models):
        """Sin modelos cargados debe retornar 503."""
        payload = {"customers": [VALID_CUSTOMER]}
        response = client_no_models.post(f"/predict/batch/{MODEL_NAME}", json=payload)
        assert response.status_code == 503
