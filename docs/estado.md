# Estado del proyecto y cómo retomarlo

Última actualización: 2026-11-02 (reentrenamiento limpio completado).

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

## Situación actual

Reentrenamiento limpio completado el 2026-11-02. Registry con v1 (`lr-C1-balanced`, run `1b717ed7…`) y v2 (Champion `gb-n200-lr0.05-d5`, run `fdded4d1…`), alias `production` -> v2. La API (3 réplicas) se reinició y `/info` devuelve el run nuevo. Los `run_id` de la documentación ya están actualizados.

## Cómo retomar tras apagar el equipo

```powershell
minikube start
kubectl wait --for=condition=Ready pod -l app=mlflow --timeout=180s
kubectl port-forward service/mlflow 5000:5000   # solo si se va a entrenar o abrir la UI
```

El volumen de MLflow persiste, así que no hace falta reentrenar. Para reentrenar: `PYTHONUTF8=1 MLFLOW_DISABLE_AGENT_HINT=1 .venv/Scripts/python.exe -m train.train` y luego `-m train.register`, seguido de `kubectl rollout restart deployment/adult-api`.

## Siguiente trabajo

1. **Paso 6:** documento de arquitectura (versión del Registry desplegada y su run), tabla de reparto de trabajo, ZIP de evidencias.

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
