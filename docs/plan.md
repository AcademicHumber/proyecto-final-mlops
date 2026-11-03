# Plan de avance — Proyecto final MLOps

Proyecto individual. Dataset: **Adult Income** (censo EE. UU. 1994, clasificación binaria `>50K` / `<=50K`). Clúster: **minikube** sobre Docker Desktop.

## Ruta general

```mermaid
flowchart LR
    P0[Paso 0<br/>Repositorio base] --> P1[Paso 1<br/>Modelo + MLflow]
    P1 --> P2[Paso 2<br/>API de inferencia]
    P2 --> P3[Paso 3<br/>Docker]
    P3 --> P4[Paso 4<br/>Kubernetes]
    P1 --> P5[Paso 5<br/>Drift]
    P4 --> P6[Paso 6<br/>Documentación y entrega]
    P5 --> P6
    P6 -.-> X[Extra: UI Streamlit]
    P5 --> C[Corrección<br/>del drift]
    C --> P6
```

## Arquitectura objetivo

```mermaid
flowchart TB
    subgraph Entrenamiento
        D[(Adult Income)] --> T[train.py<br/>split 60/20/20, semilla fija]
        T -->|5+ runs| M[MLflow Tracking]
        M --> R[Model Registry<br/>adult-income-classifier<br/>alias: production]
    end

    subgraph minikube
        SVC[Service<br/>balanceo de carga] --> P1[Pod 1]
        SVC --> P2[Pod 2]
        SVC --> P3[Pod 3]
    end

    R -->|"carga por alias production"| P1
    R --> P2
    R --> P3

    C[Cliente / tests de drift] --> SVC
```

## Pasos

| Paso | Qué se hace | Entregable |
|---|---|---|
| 0 | Git, estructura de carpetas, `requirements.txt` con versiones fijas | Repositorio con primer commit |
| 1 | Entrenar con 5+ runs en MLflow, registrar 2+ versiones, alias `production` | UI de MLflow con runs comparables |
| 2 | API FastAPI: `/predict` (devuelve el nombre del pod) y `/health`; carga el modelo por alias | API local funcionando |
| 3 | `Dockerfile` propio con dependencias fijas (incluida la versión de MLflow) | Imagen que responde con `docker run` |
| 4 | `Deployment` (3 réplicas) y `Service` en minikube; 4 demostraciones con evidencia | Capturas / logs de terminal |
| 5 | Pruebas de data drift y concept drift con pytest | Tests verde/rojo y gráfica de degradación |
| 6 | Documento de arquitectura, tabla de reparto, ZIP de evidencias | Entrega final |

## Demostraciones de Kubernetes (Paso 4)

```mermaid
flowchart LR
    A[1. Tres pods Running] --> B[2. Peticiones sucesivas<br/>atendidas por pods distintos]
    B --> C[3. Borrar un pod:<br/>se recrea y el servicio responde]
    C --> D[4. kubectl scale:<br/>cambiar réplicas y ver el efecto]
```

## Estructura del repositorio

```
proyecto-final/
├── configs/     # configuración del experimento (YAML)
├── train/       # entrenamiento y registro en MLflow
├── serving/     # API de inferencia
├── docker/      # Dockerfiles y dependencias de las imágenes
├── k8s/         # manifiestos de Kubernetes
├── drift/       # inyección y detección de drift
├── tests/       # pruebas automatizadas
├── ui/          # interfaz Streamlit (extra)
├── data/        # datos versionados con DVC
├── entrega/     # script del ZIP de entrega
├── .github/     # CI con GitHub Actions
└── docs/        # plan, arquitectura y evidencias
```
