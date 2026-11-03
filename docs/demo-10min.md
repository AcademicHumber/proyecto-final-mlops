# Agenda de la demostración (10 minutos)

Recorrido por el proyecto en el orden de las instrucciones: MLflow, Docker, Kubernetes, deriva de datos y de concepto, y dos extras (interfaz y servidor MCP). Cada bloque indica qué se muestra y qué requisito cubre; la correspondencia completa está en [cumplimiento.md](cumplimiento.md).

```mermaid
gantt
    title Demostración de 10 minutos
    dateFormat mm:ss
    axisFormat %M:%S
    section Contexto
    Caso y arquitectura        :a1, 00:00, 40s
    section Fase 1
    MLflow y Model Registry    :a2, after a1, 140s
    section Fase 2
    Docker                     :a3, after a2, 25s
    section Fase 3
    Kubernetes, 4 demostraciones :a4, after a3, 180s
    section Drift
    Data drift y concept drift :a5, after a4, 120s
    section Extras
    Interfaz                   :a6, after a5, 35s
    Servidor MCP               :a7, after a6, 35s
    section Cierre
    Trazabilidad               :a8, after a7, 25s
```

| Min | Bloque | Qué se muestra | Cubre |
|---|---|---|---|
| 0:00 a 0:40 | Caso y arquitectura | El caso de negocio y el diagrama de [arquitectura.md](arquitectura.md) | Documento de arquitectura (3.3.2) |
| 0:40 a 3:00 | MLflow | Experimento con 9 runs, orden y filtro por `cv_f1_mean`, comparación de runs y Model Registry con la versión en `production` | 3.3 y 3.4 |
| 3:00 a 3:25 | Docker | `Dockerfile.serving` con dependencias fijadas | Sección 4 |
| 3:25 a 6:25 | Kubernetes | Réplicas, balanceo, autorreparación y escalado | Sección 5.2 |
| 6:25 a 8:25 | Drift | Pruebas en verde y rojo, criterio de reentrenamiento, retraso de etiquetas y qué pasa al reentrenar | 6.1 y 6.2 |
| 8:25 a 9:00 | Interfaz | Aplicación Streamlit sobre la API del clúster | Sección 7 (extra) |
| 9:00 a 9:35 | Servidor MCP | Un asistente predice y consulta el dataset | Extra fuera de la rúbrica |
| 9:35 a 10:00 | Trazabilidad | De una respuesta de la API al pod, la versión, el run y los datos | Cierre |

## Requisitos previos

Clúster de minikube con MLflow y las tres réplicas de la API en `Running`, más los datos de `data/` (`dvc pull`). Para reproducir el recorrido:

| Ventana | Qué hace | Comando |
|---|---|---|
| A | Túnel a la API (queda abierta) | `minikube service adult-api --url` |
| B | Túnel a MLflow (queda abierta) | `kubectl port-forward service/mlflow 5000:5000` |
| C | Comandos de la demo | ver las secciones |
| D | Interfaz | `API_URL=<URL de A> streamlit run ui/app.py` |
| E | Servidor MCP desde Claude Code | `claude`, con `API_URL` apuntando a la URL de A en `.mcp.json` |

El puerto del túnel de la API cambia en cada ejecución. El entrenamiento (`train.train`) y la corrección (`train.retrain`) tardan minutos y no se ejecutan en la demostración; sus resultados están en MLflow y en [correccion.md](correccion.md).

## 1. Caso y arquitectura

Una entidad financiera quiere priorizar a quién contactar para un producto premium. El modelo estima si un prospecto gana más de 50K. Se evalúa con F1 porque solo el 24 % de las personas supera ese umbral y la accuracy engaña. El modelo vive en el Model Registry, la API lo carga por alias y tres réplicas lo sirven. Ver [crisp-dm.md](crisp-dm.md) y [arquitectura.md](arquitectura.md).

## 2. MLflow

En `localhost:5000`:

1. **Experimento `adult-income`**: 9 runs en tres familias de modelos, con y sin `class_weight`; cada run tiene descripción y 5 tags.
2. **Ordenar y filtrar** por la métrica principal, `cv_f1_mean`, y por `params.model_type`. La elección se hace con validación cruzada, sin mirar la prueba reservada.
3. **Comparar** `gb-n200-lr0.05-d5`, `rf-n200-d20-balanced` y `lr-C1`. Gana GradientBoosting en F1, AUC y precisión del 10 % superior, y es el más estable.
4. **Model Registry**: `adult-income-classifier` con v1 (base), v2 con el alias `production` (la desplegada, del run `fdded4d1`) y v3 y v4 (intentos de corregir el drift).

Más detalle en [seleccion-modelo.md](seleccion-modelo.md) y [tags.md](tags.md).

## 3. Docker

`docker/Dockerfile.serving` y `docker/requirements-serving.txt`: Dockerfile propio con las dependencias fijadas, incluida la versión de MLflow. La imagen no contiene el modelo: lo descarga del Registry por alias al arrancar, así que cambiar de versión no obliga a reconstruirla. El CI construye las dos imágenes en cada push. Ver [despliegue.md](despliegue.md).

## 4. Kubernetes

```powershell
kubectl get pods -l app=adult-api                       # 1. tres réplicas Running
python k8s/probe.py $U 12                               # 2. balanceo
kubectl delete pod <uno> --wait=false                   # 3. autorreparación
python k8s/probe.py $U 30
kubectl scale deployment/adult-api --replicas=5         # 4. escalado
python k8s/probe.py $U 25
kubectl scale deployment/adult-api --replicas=3
```

| Demostración | Qué ocurre |
|---|---|
| Réplicas | El Deployment pide 3 y hay 3 pods `1/1 Running` a la vez |
| Balanceo | Cada respuesta incluye el nombre del pod; 12 peticiones sin errores repartidas entre los tres |
| Autorreparación | Con un pod borrado y 30 peticiones en curso, Kubernetes crea otro y el Service solo envía tráfico a pods listos: 0 errores |
| Escalado | Un comando sube a 5 réplicas, que el Service incorpora al estar listas, y se vuelve a 3 |

El túnel de minikube balancea por conexión, no por petición, por lo que con pocas peticiones el reparto puede no ser parejo (ver [despliegue.md](despliegue.md)). Evidencia guardada: `evidencia-autorreparacion.txt` y `evidencia-escalado.txt`. Guía completa: [guia-demo-k8s.md](guia-demo-k8s.md).

## 5. Drift

**Pruebas en verde y rojo**, en unos 3 segundos:

```powershell
python -m pytest tests/test_data_drift.py tests/test_concept_drift.py -v -k "verde or rojo"
```

Datos del mismo origen dan verde y datos derivados dan rojo, tanto para el data drift como para el concept drift.

- **Data drift**: una prueba por tipo de variable (KS en numéricas, PSI en `capital-gain` y `capital-loss`, chi-cuadrado en categóricas), con umbral p < 0,01 dividido entre 12 pruebas (Bonferroni) y PSI ≥ 0,2. Datos limpios: 0 de 14 variables con drift; datos derivados: 5 de 14.
- **Concept drift**: el F1 por lote cae desde el lote 5. El criterio de reentrenamiento es un F1 por debajo del 90 % del base durante tres lotes seguidos; la alerta salta en el lote 9. Gráfica en `docs/img/concept_drift.png`.
- **Retraso de etiquetas**: en la práctica la etiqueta tarda meses, así que el data drift actúa como señal temprana y el F1 como confirmación; mientras tanto se puede evaluar con una muestra etiquetada de forma prioritaria o con indicadores indirectos.
- **Reentrenar no basta**: tres variantes (experimento `adult-income-correccion`, versiones 3 y 4 del Registry) suben el F1 de 0,345 a 0,363 como máximo, lejos del 0,71 de referencia. Por eso el alias sigue en la v2. Las soluciones posibles están en [correccion.md](correccion.md).

Detalle en [drift.md](drift.md). El drift es inyectado: el dataset es de 1994 y no tiene deriva real.

## 6. Interfaz (extra)

Aplicación Streamlit en `localhost:8501` que consume la API del clúster, no un proceso local. Muestra el modelo y el run desplegados, ofrece atajos de perfiles para rellenar el formulario, evalúa un prospecto o una campaña completa (contactando al 10 % mejor puntuado) e indica qué pod respondió cada consulta. Ver [ui.md](ui.md).

## 7. Servidor MCP (extra)

Desde Claude Code con el servidor `adult-income` conectado, dos preguntas de ejemplo:

1. *¿Qué proporción gana más de 50K por nivel educativo en `test`?* Usa `consultar_datos`.
2. *Evalúa a un ejecutivo de 45 años con maestría que trabaja 50 horas.* Usa `predecir_prospecto` y devuelve la probabilidad, la prioridad, la versión del modelo y el pod que respondió.

La predicción no se calcula en local: pasa por la API del clúster. Las consultas al dataset son de solo lectura. Si el asistente no está disponible, `python -m pytest tests/test_mcp.py` ejecuta las 16 pruebas del servidor, incluida una de extremo a extremo. Ver [mcp.md](mcp.md).

## 8. Trazabilidad

```powershell
curl $U/info
```

De una respuesta de la API se llega al pod, a la versión 2, al run `fdded4d1`, a su configuración y a los datos versionados con DVC. Todo está en el repositorio con integración continua, y cada desviación del flujo estándar está registrada en [desviaciones.md](desviaciones.md).
