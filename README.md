# Telco Customer Churn — MLOps Pipeline
## Grupo: Natalia Alvarado, Camilo Mújica, Sergio Rada y Carlos Romero

Servicio de inferencia para predicción de churn en clientes de telecomunicaciones, construido con **FastAPI** y empaquetado en **Docker**, con CI/CD automatizado mediante **GitHub Actions**.

La API expone **4 modelos simultáneos**: Random Forest, XGBoost, CatBoost y LightGBM. Todos se cargan al arrancar el servicio y se seleccionan por URL en cada predicción.

---

## Estructura del proyecto

```
telco-churn-mlops/
├── data/
│   ├── telco_churn.csv              #Dataset original
│   └── models/                      #Modelos entrenados
│       ├── rf_best.pkl
│       ├── xgb_best.pkl
│       ├── cb_best.pkl
│       └── lgbm_best.pkl
├── notebooks/
│   ├── 1_eda_preprocessing.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_interpretability.ipynb
├── app/
│   ├── api.py                       #Endpoints FastAPI
│   └── schemas.py                   #Validación Pydantic
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Endpoints disponibles

| Método | Ruta                          | Descripción                           |
|--------|-------------------------------|---------------------------------------|
| GET    | `/`                           | Estado del servicio y modelos activos |
| GET    | `/health`                     | Cantidad de modelos cargados          |
| GET    | `/models`                     | Lista de modelos disponibles          |
| POST   | `/predict/{model_name}`       | Predicción para un cliente            |
| POST   | `/predict/batch/{model_name}` | Predicción para hasta 500 clientes    |

**Valores válidos para `model_name`:** `random_forest` · `xgboost` · `catboost` · `lightgbm`

### Ejemplo de request

```json
POST /predict/xgboost

{
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
  "TotalCharges": 1026.0
}
```

### Ejemplo de response

```json
{
  "model_used": "xgboost",
  "churn_prediction": 1,
  "churn_probability": 0.7241,
  "risk_label": "High"
}
```

---

## Opción A — Probar descargando la imagen Docker publicada

Esta es la forma más rápida. Solo necesitas tener **Docker Desktop** instalado y corriendo. Luego de eso en Bash:

### Paso 1 — Descargar la imagen

```bash
docker pull ghcr.io/romeromendozacarlosisaac-pixel/telco-churn-api:latest
```

### Paso 2 — Correr el contenedor

```bash
docker run -p 8000:8000 ghcr.io/romeromendozacarlosisaac-pixel/telco-churn-api:latest
```

Al arrancar correctamente verás los 4 modelos cargados:

```
[INFO] Modelo 'random_forest' cargado desde: /app/data/models/rf_best.pkl
[INFO] Modelo 'xgboost'       cargado desde: /app/data/models/xgb_best.pkl
[INFO] Modelo 'catboost'      cargado desde: /app/data/models/cb_best.pkl
[INFO] Modelo 'lightgbm'      cargado desde: /app/data/models/lgbm_best.pkl
INFO:  Application startup complete.
```

### Paso 3 — Probar el servicio

Abre el navegador y ve a:

```
http://localhost:8000/docs
```

Verás la documentación interactiva de Swagger generada automáticamente por FastAPI. Desde ahí puedes seleccionar el modelo, completar los campos del cliente y ejecutar predicciones con un botón, sin necesidad de escribir ningún comando adicional.

Para detener el contenedor: `CTRL + C` en la terminal donde corre, y para reactivarlo nuevamente aplicar el Paso 2.

---

## Opción B — Probar localmente desde el código fuente

Esta opción requiere clonar el repositorio y tener Python 3.11 instalado.

### Requisitos previos

- **Python 3.11** — [python.org/downloads](https://www.python.org/downloads/)
- **Git** — [git-scm.com](https://git-scm.com/)

### Paso 1 — Clonar el repositorio

Utilizando Bash:

```bash
git clone https://github.com/romeromendozacarlosisaac-pixel/Miniproyecto_3.git
cd Miniproyecto_3
```

### Paso 2 — Crear y activar el entorno virtual

```bash
#Crear el entorno
conda create -n miniproyecto3 python=3.11 -y

#Activar entorno
conda activate miniproyecto3
```

Una vez activado, el prompt mostrará `(venv)` al inicio.

### Paso 3 — Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 4 — Levantar el servicio
En otra terminal, ejecutar:
```bash
uvicorn app.api:app --reload --port 8000
```

### Paso 5 — Probar el servicio

Igual que en la Opción A, abre el navegador en:

```
http://localhost:8000/docs
```
Por otro lado, si se desea probar localmente, solo basta con ingresar la entrada en la terminal principal, por ejemplo:

```
Invoke-WebRequest -Uri http://localhost:8000/predict/xgboost `
  -Method POST `
  -ContentType "application/json" `
  -UseBasicParsing `
  -Body '{
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
    "TotalCharges": 1026.0
  }' | Select-Object -ExpandProperty Content
```

La salida esperada es:

```
{"model_used":"xgboost",
"churn_prediction":1,
"churn_probability":0.8417,
"risk_label":"High"}
```

Para detener el servidor: `CTRL + C`.

---

## Pruebas unitarias

```bash
#Pruebas de API (no requieren modelos en disco)
pytest tests/test_api.py -v

#Pruebas de modelo (requieren data/models/*.pkl y data/telco_churn.csv)
pytest tests/test_model.py -v
```

---

## CI/CD

El pipeline de GitHub Actions en `.github/workflows/ci.yml` se activa automáticamente en cada `push` a `main` y ejecuta tres jobs en secuencia:

1. **lint** — Verifica calidad del código con `flake8`
2. **test** — Ejecuta las pruebas unitarias con `pytest`
3. **docker** — Construye y publica la imagen en `ghcr.io` si los dos anteriores pasan

La imagen más reciente siempre está disponible en:

```
ghcr.io/romeromendozacarlosisaac-pixel/telco-churn-api:latest
```
