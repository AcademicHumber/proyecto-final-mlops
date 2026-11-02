# API de inferencia

Servicio FastAPI (`serving/app.py`). Resuelve el modelo siempre por **alias** en el Model Registry de MLflow.

| Endpoint | Uso |
|---|---|
| `POST /predict` | Lista de registros (14 variables) -> predicción y probabilidad de `>50K`, versión del modelo, run y **pod** que respondió |
| `GET /health` | 200 si el modelo está cargado, 503 si no (sonda de Kubernetes) |
| `GET /info` | Modelo, versión, run_id, alias y pod |

Variables de entorno: `MLFLOW_TRACKING_URI` (defecto `sqlite:///mlflow.db`) y `MODEL_URI` (defecto `models:/adult-income-classifier@production`).

## Flujo de una petición

```mermaid
sequenceDiagram
    participant C as Cliente
    participant S as Service K8s
    participant P as Pod (FastAPI)
    participant R as Model Registry MLflow

    Note over P,R: Al arrancar (lifespan)
    P->>R: "resolver alias production"
    R-->>P: "versión 2 + run_id + modelo"

    C->>S: POST /predict
    S->>P: "reenvía a un pod"
    P->>P: "valida con pydantic, None a NaN"
    P->>P: "pipeline.predict_proba"
    P-->>C: "predicción + versión + run_id + nombre del pod"
```

El campo `pod` de la respuesta permite demostrar el balanceo de carga: peticiones sucesivas al Service devuelven nombres de pod distintos.
