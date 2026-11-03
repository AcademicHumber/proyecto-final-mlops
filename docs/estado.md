# Estado del proyecto y cómo retomarlo

Última actualización: 2026-11-02, al cerrar la sesión (clúster apagado).

## Dónde quedamos

| Paso | Estado |
|---|---|
| 0. Repositorio base | Hecho |
| 1. Modelo y MLflow (9 runs, Registry, alias `production`) | Hecho |
| 2. API de inferencia | Hecho |
| 3. Docker | Hecho (`docker/Dockerfile.mlflow`, `docker/Dockerfile.serving`) |
| 4. Kubernetes | MLflow y API desplegados con 3 réplicas. Balanceo verificado, autorreparación y escalado demostrados con evidencia en `docs/` |
| 5. Drift (data y concept) | Hecho: `drift/`, 20 pruebas, documentado en `drift.md` |
| 6. Documentación final y entrega | En curso |
| Extra (UI Streamlit) | Opcional, solo con el proyecto base completo |

## Situación al cerrar la sesión

- **Clúster apagado** con `minikube stop`. No se borró nada: el volumen de MLflow persiste. Túneles (`port-forward` y `minikube service`) cerrados.
- **Registry de MLflow:** v1 (`lr-C1-balanced`, run `1b717ed7…`) y v2 (Champion `gb-n200-lr0.05-d5`, run `fdded4d1…`), alias `production` -> v2. Entrenado sobre los datos `data-v1`.
- **API:** 3 réplicas sirviendo la v2.
- **Datos:** v1 y v2 versionadas con DVC (etiquetas `data-v1`, `data-v2`). Almacén en `../dvc-almacen`. `dvc status` al día.
- **Código y docs:** todo commiteado. Checkpoints en etiquetas de Git (ver abajo). 32 pruebas en verde.
- **Sin hacer:** el modelo corregido (v3). Nada pendiente de commitear.

## Cómo retomar mañana

```powershell
minikube start
kubectl wait --for=condition=Ready pod -l app=mlflow --timeout=180s
kubectl get pods                                  # la API tarda un poco más: 3 pods 1/1
kubectl port-forward service/mlflow 5000:5000     # dejar abierta (entrenar, pruebas con modelo)
```

- Al arrancar, la API puede reiniciarse una vez mientras MLflow termina de levantar. Es normal; esperar a que estén `1/1`.
- No hace falta reentrenar: el volumen persiste. Comprobar con `curl http://localhost:5000/health` y que `/api/2.0/mlflow/registered-models/search` liste `adult-income-classifier`.
- Si el Registry apareciera vacío: `PYTHONUTF8=1 MLFLOW_DISABLE_AGENT_HINT=1 .venv/Scripts/python.exe -m train.train`, luego `-m train.register` y `kubectl rollout restart deployment/adult-api`.
- Los datos: `dvc status` debe decir "up to date". Si faltan CSV, `dvc pull`.
- Para la demo de Kubernetes en vivo, ver `guia-demo-k8s.md`.

## Siguiente trabajo

1. **Modelo corregido (v3) con los datos v2.** Propuesta pendiente de confirmar: entrenar con `train.csv` más los lotes 1 a 9 de `produccion_concept_drift.csv` (los etiquetados cuando salta la alarma) y evaluar con los lotes 10 a 12. Registrar como v3 sin mover el alias. Añadir al run el hash DVC de los datos y el drift aplicado.
2. **Demo de corrección repetible:** mover el alias a la v3, `rollout restart`, y poder volver a la v2. Las pruebas de concept drift dependen del alias, así que deben fijar la versión del modelo (o la demo debe terminar devolviendo `production` a la v2). Dejar una guía como la de Kubernetes.
3. **Paso 6:** documento de arquitectura (versión del Registry desplegada y su run), tabla de reparto de trabajo, ZIP de evidencias.

## Decisiones ya tomadas (para no repetirlas)

- Datos versionados con DVC (ver `datos.md`). Almacén local en `../dvc-almacen`, fuera del repositorio.
- Dataset: Adult Income. Trabajo individual. Clúster: minikube sobre Docker Desktop.
- Caso de negocio: entidad financiera que prioriza prospectos. Se mantienen `sex`, `race` y `native-country` (limitación declarada en `crisp-dm.md`).
- Métrica principal: F1. El Champion se elige por F1 en validación cruzada, no por test.
- 5 tags por run; sin `role`, `git_dirty` ni `dataset` (motivos en `tags.md`).
- Toda desviación del flujo estándar se anota en `desviaciones.md` (actualmente 16 entradas).
- Documentar con diagramas Mermaid (se presenta en GitHub). Cuidado con `@` y `:` en etiquetas de flechas.

## Detalles del entorno que conviene recordar

- Usar `PYTHONUTF8=1` al entrenar en Windows, o MLflow falla al imprimir un emoji.
- El minikube usa containerd: las imágenes se construyen con `docker build` y se cargan con `minikube image load`. Si se cambia una imagen, subir la etiqueta.
- `kubectl` es 1.32 y el clúster 1.37 (fuera de la diferencia de versiones soportada; funciona).
- MLflow necesita unos 1.7 GiB en el clúster; el límite está en 2.5 GiB.
- Para demostrar el balanceo, usar el NodePort con `minikube service`, no `port-forward` (este fija un solo pod).

## Checkpoints (etiquetas de Git)

| Etiqueta | Commit | Qué marca |
|---|---|---|
| `checkpoint-1-base-k8s` | `d5b05e8` | Modelo en MLflow, API y despliegue en Kubernetes |
| `checkpoint-2-config-yaml` | `e71faa7` | Configuración del experimento en YAML, tags y descripciones |
| `checkpoint-3-k8s-demos` | `0f09cb6` | Paso 4 completo: autorreparación y escalado con evidencia |
| `checkpoint-4-drift` | `f288a74` | Paso 5: detección de data drift y concept drift con pruebas |
| `checkpoint-5-dvc` | `236751f` | Datos versionados con DVC (igual que `data-v1`) |
| `checkpoint-6-registry-limpio` | `c005383` | Registry reentrenado desde cero, API sirviendo v2 |

Ver la lista: `git tag -n1 --sort=creatordate`. Volver a un punto y dejar los datos coherentes:

```bash
git checkout checkpoint-4-drift    # código y archivos .dvc de ese momento
dvc checkout                        # datos de ese momento (desde checkpoint-5 en adelante)
git checkout main                   # regresar
```

**Qué cubren y qué no:** una etiqueta fija el código, la documentación y (desde `checkpoint-5`) los datos. **No fija el estado de MLflow ni del clúster**: runs y modelos viven en el volumen de minikube. Tras volver a un checkpoint antiguo hay que reentrenar para que el Registry coincida. Las etiquetas son locales (el repositorio no tiene remoto).
