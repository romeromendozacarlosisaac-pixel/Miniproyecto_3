# =============================================================================
# TELCO CUSTOMER CHURN — VALIDACIÓN DEL MODELO
# =============================================================================
# Estas pruebas verifican que el modelo cargado desde disco cumpla
# los contratos mínimos de calidad antes de pasar a producción.
# =============================================================================

import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("app/model.joblib")

# Umbrales mínimos de calidad aceptables para producción
MIN_ROC_AUC = 0.78
MIN_F1      = 0.55

# Un cliente de ejemplo para pruebas de forma
SAMPLE_INPUT = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50, "TotalCharges": 1026.0,
}])


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Modelo no encontrado en {MODEL_PATH} — saltar pruebas de modelo.")
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_data():
    """
    Carga una porción del dataset para validación de métricas.
    Si no existe el CSV, las pruebas de métricas se omiten.
    """
    csv_path = Path("data/telco_churn.csv")
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
    )
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.drop(columns=["customerID"], inplace=True)

    y = (df.pop("Churn") == "Yes").astype(int)
    X = df.copy()
    return X, y


# =============================================================================
# PRUEBAS DE CONTRATO DEL MODELO
# =============================================================================

class TestModelContract:

    def test_model_loads(self, model):
        """El modelo debe cargarse sin errores."""
        assert model is not None

    def test_model_has_predict(self, model):
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, "predict_proba")

    def test_predict_returns_binary(self, model):
        preds = model.predict(SAMPLE_INPUT)
        assert set(preds).issubset({0, 1}), "Las predicciones deben ser 0 o 1."

    def test_predict_proba_shape(self, model):
        proba = model.predict_proba(SAMPLE_INPUT)
        assert proba.shape == (1, 2), "predict_proba debe retornar (n_samples, 2)."

    def test_predict_proba_sums_to_one(self, model):
        proba = model.predict_proba(SAMPLE_INPUT)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, model):
        proba = model.predict_proba(SAMPLE_INPUT)
        assert (proba >= 0).all() and (proba <= 1).all()


# =============================================================================
# PRUEBAS DE MÉTRICAS MÍNIMAS DE PRODUCCIÓN
# =============================================================================

class TestModelMetrics:

    def test_roc_auc_above_threshold(self, model, test_data):
        from sklearn.metrics import roc_auc_score
        X, y = test_data
        if X is None:
            pytest.skip("Dataset no disponible.")

        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        assert auc >= MIN_ROC_AUC, (
            f"ROC-AUC ({auc:.4f}) por debajo del umbral mínimo ({MIN_ROC_AUC})."
        )

    def test_f1_above_threshold(self, model, test_data):
        from sklearn.metrics import f1_score
        X, y = test_data
        if X is None:
            pytest.skip("Dataset no disponible.")

        preds = model.predict(X)
        f1 = f1_score(y, preds)
        assert f1 >= MIN_F1, (
            f"F1-Score ({f1:.4f}) por debajo del umbral mínimo ({MIN_F1})."
        )

    def test_no_constant_predictions(self, model, test_data):
        """El modelo no debe predecir siempre la misma clase."""
        X, y = test_data
        if X is None:
            pytest.skip("Dataset no disponible.")

        preds = model.predict(X)
        assert len(set(preds)) > 1, "El modelo predice siempre la misma clase."
        