# =============================================================================
# TELCO CUSTOMER CHURN — PRUEBAS UNITARIAS DE LA API
# =============================================================================

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Fixtures de datos ─────────────────────────────────────────────────────────
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
# HELPERS
# =============================================================================

def make_mock_model(proba=None):
    """Crea un modelo falso con probabilidades controladas."""
    if proba is None:
        proba = [0.3, 0.7]
    m = MagicMock()
    m.predict_proba.return_value = np.array([proba])
    return m


def make_client(models_dict):
    """
    Crea un TestClient parcheando joblib.load Y MODELS para que el lifespan
    no toque disco y use únicamente los modelos del dict recibido.
    """
    from app.api import app

    # Parcheamos joblib.load para que no lea .pkl reales
    mock_loader = MagicMock(side_effect=lambda path: models_dict.get(
        next((k for k, v in {
            "random_forest": "rf_best.pkl",
            "xgboost": "xgb_best.pkl",
            "catboost": "cb_best.pkl",
            "lightgbm": "lgbm_best.pkl",
        }.items() if v in path), None),
        None,
    ))

    with patch("app.api.joblib.load", mock_loader):
        with patch("app.api.MODEL_FILES", {k: f"{k}.pkl" for k in models_dict}):
            with TestClient(app) as c:
                yield c


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_model():
    return make_mock_model([0.3, 0.7])


@pytest.fixture
def client(mock_model):
    """Cliente con un modelo mockeado cargado."""
    yield from make_client({MODEL_NAME: mock_model})


@pytest.fixture
def client_no_models():
    """Cliente sin ningún modelo disponible."""
    from app.api import app
    # MODEL_FILES vacío → lifespan no carga nada → lanza RuntimeError
    # Lo capturamos para que el cliente arranque igualmente con MODELS vacío
    with patch("app.api.MODEL_FILES", {}):
        with patch("app.api.MODELS", {}):
            # Evitamos el RuntimeError del lifespan parcheando la condición
            with patch("app.api.MODELS", {}) as mock_models:
                mock_models.clear()
                # Arrancamos sin lifespan ejecutando el app directamente
                with TestClient(app, raise_server_exceptions=False) as c:
                    # Forzamos MODELS vacío después del arranque
                    with patch("app.api.MODELS", {}):
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

    def test_health_model_not_loaded(self):
        """Sin modelo cargado debe retornar 503."""
        from app.api import app
        with patch("app.api.MODEL_FILES", {}):
            with patch("app.api.MODELS", {}):
                with TestClient(app, raise_server_exceptions=False) as c:
                    with patch("app.api.MODELS", {}):
                        response = c.get("/health")
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

    def test_predict_risk_label_low(self):
        """Con 20% de probabilidad debe retornar Low."""
        m = make_mock_model([0.8, 0.2])
        gen = make_client({MODEL_NAME: m})
        c = next(gen)
        response = c.post(f"/predict/{MODEL_NAME}", json=LOW_RISK_CUSTOMER)
        assert response.json()["risk_label"] == "Low"

    def test_predict_risk_label_medium(self):
        """Con 50% de probabilidad debe retornar Medium."""
        m = make_mock_model([0.5, 0.5])
        gen = make_client({MODEL_NAME: m})
        c = next(gen)
        response = c.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
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

    def test_predict_model_unavailable_returns_503(self):
        """Sin modelos cargados debe retornar 503."""
        from app.api import app
        with patch("app.api.MODEL_FILES", {}):
            with TestClient(app, raise_server_exceptions=False) as c:
                with patch("app.api.MODELS", {}):
                    response = c.post(f"/predict/{MODEL_NAME}", json=VALID_CUSTOMER)
        assert response.status_code == 503

    def test_predict_unknown_model_returns_422(self, client):
        """Un modelo fuera del enum debe retornar 422."""
        response = client.post("/predict/nonexistent_model", json=VALID_CUSTOMER)
        assert response.status_code == 422


# =============================================================================
# PRUEBAS DE /predict/batch/{model_name}
# =============================================================================

class TestPredictBatch:

    def test_batch_valid_returns_200(self):
        m = make_mock_model([0.3, 0.7])
        m.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        gen = make_client({MODEL_NAME: m})
        c = next(gen)
        payload = {"customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]}
        response = c.post(f"/predict/batch/{MODEL_NAME}", json=payload)
        assert response.status_code == 200

    def test_batch_total_matches_input(self):
        m = make_mock_model([0.3, 0.7])
        m.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        gen = make_client({MODEL_NAME: m})
        c = next(gen)
        payload = {"customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]}
        data = c.post(f"/predict/batch/{MODEL_NAME}", json=payload).json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_list_returns_422(self, client):
        response = client.post(f"/predict/batch/{MODEL_NAME}", json={"customers": []})
        assert response.status_code == 422

    def test_batch_model_unavailable_returns_503(self):
        """Sin modelos cargados debe retornar 503."""
        from app.api import app
        with patch("app.api.MODEL_FILES", {}):
            with TestClient(app, raise_server_exceptions=False) as c:
                with patch("app.api.MODELS", {}):
                    payload = {"customers": [VALID_CUSTOMER]}
                    response = c.post(f"/predict/batch/{MODEL_NAME}", json=payload)
        assert response.status_code == 503
