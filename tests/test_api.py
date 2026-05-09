# =============================================================================
# TELCO CUSTOMER CHURN — PRUEBAS UNITARIAS DE LA API
# =============================================================================

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


# =============================================================================
# SETUP DEL CLIENTE DE PRUEBA CON MODELO MOCKEADO
# =============================================================================

@pytest.fixture
def mock_model():
    """Crea un modelo falso que retorna probabilidades controladas."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% churn
    return m


@pytest.fixture
def client(mock_model):
    """TestClient con el modelo inyectado vía mock."""
    with patch("app.api.model", mock_model):
        from app.api import app
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
        assert response.json()["model_loaded"] is True

    def test_health_model_not_loaded(self):
        """Sin modelo cargado debe retornar 503."""
        with patch("app.api.model", None):
            from app.api import app
            with TestClient(app) as c:
                response = c.get("/health")
            assert response.status_code == 503


# =============================================================================
# PRUEBAS DE /predict
# =============================================================================

class TestPredict:

    def test_predict_valid_customer_returns_200(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_label" in data

    def test_predict_churn_prediction_is_binary(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        pred = response.json()["churn_prediction"]
        assert pred in [0, 1]

    def test_predict_probability_in_range(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_label_high(self, client):
        """Con 70% de probabilidad debe retornar High."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.json()["risk_label"] == "High"

    def test_predict_risk_label_low(self, mock_model, client):
        """Con 20% de probabilidad debe retornar Low."""
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        response = client.post("/predict", json=LOW_RISK_CUSTOMER)
        assert response.json()["risk_label"] == "Low"

    def test_predict_risk_label_medium(self, mock_model, client):
        """Con 50% de probabilidad debe retornar Medium."""
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.json()["risk_label"] == "Medium"

    def test_predict_missing_field_returns_422(self, client):
        """Un campo requerido faltante debe retornar 422."""
        incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_invalid_gender_returns_422(self, client):
        bad = {**VALID_CUSTOMER, "gender": "Unknown"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_negative_tenure_returns_422(self, client):
        bad = {**VALID_CUSTOMER, "tenure": -5}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_model_unavailable_returns_503(self):
        with patch("app.api.model", None):
            from app.api import app
            with TestClient(app) as c:
                response = c.post("/predict", json=VALID_CUSTOMER)
            assert response.status_code == 503


# =============================================================================
# PRUEBAS DE /predict/batch
# =============================================================================

class TestPredictBatch:

    def test_batch_valid_returns_200(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        payload = {"customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_total_matches_input(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        payload = {"customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]}
        data = client.post("/predict/batch", json=payload).json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_list_returns_422(self, client):
        response = client.post("/predict/batch", json={"customers": []})
        assert response.status_code == 422

    def test_batch_model_unavailable_returns_503(self):
        with patch("app.api.model", None):
            from app.api import app
            with TestClient(app) as c:
                payload = {"customers": [VALID_CUSTOMER]}
                response = c.post("/predict/batch", json=payload)
            assert response.status_code == 503
