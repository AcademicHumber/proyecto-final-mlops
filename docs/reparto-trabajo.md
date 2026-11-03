# Reparto de trabajo

Proyecto **individual**. Todos los componentes los construyó Adrián Fernández.

| Componente | Quién | Dónde |
|---|---|---|
| Elección del problema y caso de negocio (CRISP-DM) | Adrián Fernández | `docs/crisp-dm.md` |
| Entrenamiento, 9 runs y configuración YAML | Adrián Fernández | `train/`, `configs/` |
| MLflow: seguimiento, Registry y alias | Adrián Fernández | `train/register.py`, `docs/seleccion-modelo.md` |
| API de inferencia | Adrián Fernández | `serving/`, `docs/api.md` |
| Docker | Adrián Fernández | `docker/` |
| Kubernetes y demostraciones | Adrián Fernández | `k8s/`, `docs/despliegue.md` |
| Versionado de datos con DVC | Adrián Fernández | `data/*.dvc`, `docs/datos.md` |
| Data drift y concept drift, con pruebas | Adrián Fernández | `drift/`, `tests/`, `docs/drift.md` |
| Corrección del drift y análisis | Adrián Fernández | `train/retrain.py`, `docs/correccion.md` |
| Documentación y entrega | Adrián Fernández | `docs/`, `README.md`, `entrega/` |

## Uso de asistencia de IA

Se usó Claude Code como asistente de programación durante el desarrollo. Los commits que generó llevan la línea `Co-Authored-By` y se pueden ver en el historial (`git log`). Las decisiones de diseño (dataset, caso de negocio, métricas, umbrales, reparto de datos para la corrección y el enfoque de la demo) las confirmó el autor, y están registradas en `docs/desviaciones.md`.

## Evidencia en el historial

El historial de Git (`git log`) muestra el avance por pasos, con checkpoints en etiquetas (`git tag -n1`).
