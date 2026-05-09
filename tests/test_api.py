# =============================================================================
# TELCO CUSTOMER CHURN — PRUEBAS UNITARIAS DE LA API
# =============================================================================

import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# ── Datos de prueba ───────────────────────────────────────────────────────────
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
# FIXTURES BASE
# =============================================================================

def make_mock(proba_class1=0.7):
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - proba_class1, proba_class1]])
    return m


@pytest.fixture(scope="module")
def app_with_real_models():
    """
    Levanta la app UNA sola vez con los modelos reales para toda la suite.
    Los tests que necesiten comportamiento controlado parchean MODELS inline.
    """
    from app.api import app
    with TestClient(app) as client:
        yield client


# =============================================================================
# PRUEBAS DE HEALTH
# =============================================================================

class TestHealth:

    def test_root_returns_ok(self, app_with_real_models):
        response = app_with_real_models.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_model_loaded(self, app_with_real_models):
        response = app_with_real_models.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_model_not_loaded(self, app_with_real_models):
        """Simula MODELS vacío parcheando durante la request."""
        import app.api as api_module
        original = dict(api_module.MODELS)
        api_module.MODELS.clear()
        try:
            response = app_with_real_models.get("/health")
            assert response.status_code == 503
        finally:
            api_module.MODELS.update(original)

    def test_list_models(self, app_with_real_models):
        response = app_with_real_models.get("/models")
        assert response.status_code == 200
        assert MODEL_NAME in response.json()["modelos_disponibles"]


# =============================================================================
# PRUEBAS DE /predict/{model_name}
# =============================================================================

class TestPredict:

    def test_predict_valid_customer_returns_200(self, app_with_real_models):
        response = app_with_real_models.post(
            f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, app_with_real_models):
        response = app_with_real_models.post(
            f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER
        )
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_label" in data
        assert "model_used" in data

    def test_predict_churn_prediction_is_binary(self, app_with_real_models):
        response = app_with_real_models.post(
            f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER
        )
        assert response.json()["churn_prediction"] in [0, 1]

    def test_predict_probability_in_range(self, app_with_real_models):
        response = app_with_real_models.post(
            f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER
        )
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_label_high(self, app_with_real_models):
        """Inyecta mock con 80% churn → debe retornar High."""
        import app.api as api_module
        original = dict(api_module.MODELS)
        api_module.MODELS[MODEL_NAME] = make_mock(0.80)
        try:
            response = app_with_real_models.post(
                f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER
            )
            assert response.json()["risk_label"] == "High"
        finally:
            api_module.MODELS.update(original)

    def test_predict_risk_label_low(self, app_with_real_models):
        """Inyecta mock con 20% churn → debe retornar Low."""
        import app.api as api_module
        original = dict(api_module.MODELS)
        api_module.MODELS[MODEL_NAME] = make_mock(0.20)
        try:
            response = app_with_real_models.post(
                f"/predict/{MODEL_NAME}", json=LOW_RISK_CUSTOMER
            )
            assert response.json()["risk_label"] == "Low"
        finally:
            api_module.MODELS.update(original)

    def test_predict_risk_label_medium(self, app_with_real_models):
        """Inyecta mock con 50% churn → debe retornar Medium."""
        import app.api as api_module
        original = dict(api_module.MODELS)
        api_module.MODELS[MODEL_NAME] = make_mock(0.50)
        try:
            response = app_with_real_models.post(
                f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER
            )
            assert response.json()["risk_label"] == "Medium"
        finally:
            api_module.MODELS.update(original)

    def test_predict_missing_field_returns_422(self, app_with_real_models):
        incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        response = app_with_real_models.post(
            f"/predict/{MODEL_NAME}", json=incomplete
        )
        assert response.status_code == 422

    def test_predict_invalid_gender_returns_422(self, app_with_real_models):
        bad = {**VALID_CUSTOMER, "gender": "Unknown"}
        response = app_with_real_models.post(f"/predict/{MODEL_NAME}", json=bad)
        assert response.status_code == 422

    def test_predict_negative_tenure_returns_422(self, app_with_real_models):
        bad = {**VALID_CUSTOMER, "tenure": -5}
        response = app_with_real_models.post(f"/predict/{MODEL_NAME}", json=bad)
        assert response.status_code == 422

    def test_predict_model_unavailable_returns_503(self, app_with_real_models):
        """Vacía MODELS durante la request → debe retornar 503."""
        import app.api as api_module
        original = dict(api_module.MODELS)
        api_module.MODELS.clear()
        try:
            response = app_with_real_models.post(
                f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER
            )
            assert response.status_code == 503
        finally:
            api_module.MODELS.update(original)

    def test_predict_unknown_model_returns_422(self, app_with_real_models):
        response = app_with_real_models.post(
            "/predict/nonexistent_model", json=VALID_CUSTOMER
        )
        assert response.status_code == 422


# =============================================================================
# PRUEBAS DE /predict/batch/{model_name}
# =============================================================================

class TestPredictBatch:

    def test_batch_valid_returns_200(self, app_with_real_models):
        payload = {"customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]}
        response = app_with_real_models.post(
            f"/predict/batch/{MODEL_NAME}", json=payload
        )
        assert response.status_code == 200

    def test_batch_total_matches_input(self, app_with_real_models):
        payload = {"customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]}
        data = app_with_real_models.post(
            f"/predict/batch/{MODEL_NAME}", json=payload
        ).json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_list_returns_422(self, app_with_real_models):
        response = app_with_real_models.post(
            f"/predict/batch/{MODEL_NAME}", json={"customers": []}
        )
        assert response.status_code == 422

    def test_batch_model_unavailable_returns_503(self, app_with_real_models):
        """Vacía MODELS durante la request → debe retornar 503."""
        import app.api as api_module
        original = dict(api_module.MODELS)
        api_module.MODELS.clear()
        try:
            payload = {"customers": [VALID_CUSTOMER]}
            response = app_with_real_models.post(
                f"/predict/batch/{MODEL_NAME}", json=payload
            )
            assert response.status_code == 503
        finally:
            api_module.MODELS.update(original)
