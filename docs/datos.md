# Versionado de datos con DVC

**Problema:** Git guarda bien el código, pero no los datos (archivos grandes que cambian). Sin versionar los datos no se puede saber con qué datos se entrenó cada modelo.

**Solución:** DVC. Git guarda un archivo pequeño (`.dvc`) con el hash del dato, y DVC guarda el dato real en otro sitio. Cada commit de Git apunta a una versión exacta de los datos.

```mermaid
flowchart LR
    subgraph Git
        C[Código] --- F["data/train.csv.dvc<br/>hash MD5 del dato"]
    end
    subgraph DVC
        K[(Caché local<br/>.dvc/cache)] --- A[(Almacén<br/>../dvc-almacen)]
    end
    F -.->|apunta a| K
    D["data/train.csv<br/>el CSV real"] -->|dvc add| K
    K -->|dvc push| A
    A -->|dvc pull| K
    K -->|dvc checkout| D
```

## Qué está versionado

| Archivo | Qué es | Hash MD5 |
|---|---|---|
| `data/train.csv` | 60 %: entrenamiento y baseline del drift | `0b6e063f1d79…` |
| `data/test.csv` | 20 %: evaluación | `3506b073e70a…` |
| `data/produccion.csv` | 20 %: tráfico futuro simulado | `9c69d817222e…` |

En Git están `data/*.csv.dvc`, `data/.gitignore` (generado por DVC para no subir los CSV) y `.dvc/config`. Los CSV no van a Git.

## Comandos del día a día

| Qué quiero | Comando |
|---|---|
| Bajar los datos en una máquina nueva | `git clone` y luego `dvc pull` |
| Ver si los datos locales cambiaron | `dvc status` |
| Registrar datos modificados o nuevos | `dvc add data/archivo.csv`, luego `git add data/archivo.csv.dvc` y `git commit` |
| Subir los datos al almacén | `dvc push` |
| Volver a los datos de un commit antiguo | `git checkout <commit>` y luego `dvc checkout` |

## Configuración

- **Almacén (remote):** carpeta local `../dvc-almacen`, hermana del repositorio. Es la opción más simple para un proyecto individual.
- **Telemetría de DVC desactivada** (`core.analytics false`).
- **Versión fijada:** `dvc==3.67.1` en `requirements.txt`.

## Cómo se verifica

- `tests/test_datos.py`: comprueba que cada CSV local tiene el mismo MD5 que su archivo `.dvc`. Si alguien cambia un dato sin registrarlo, la prueba falla y dice cómo restaurarlo.
- Se probó la recuperación: se borraron los CSV y la caché, `dvc pull` los restauró y los hashes fueron idénticos.
- Se comprobó que regenerar los splits desde OpenML (`train.data`) produce **exactamente los mismos bytes** que los CSV versionados. DVC y el código concuerdan.

## Relación con MLflow

Cada run de MLflow tiene el tag `data_hash` (huella del contenido de `train`, calculada con pandas). El hash de DVC es MD5 del archivo y es **otro** valor. Ambos identifican los mismos datos, pero no son comparables entre sí. Para la v3 (modelo corregido) se añadirán al run el hash DVC de los datos y el tipo de drift aplicado, de modo que se pueda volver a los datos exactos con `dvc checkout`.

## Limitaciones

- **El almacén está en el mismo disco que el proyecto.** Protege contra borrar o cambiar los CSV por error, pero no contra perder el disco. En un proyecto real el almacén sería remoto (S3, Google Drive, Azure…) y bastaría con `dvc remote modify`.
- **El dato original sigue en OpenML.** DVC asegura que tenemos una copia exacta de lo que se usó, aunque OpenML cambiara el dataset.
- **Los CSV de `data/` no están en el repositorio.** Quien clone necesita acceso al almacén, o regenerarlos con `python -m train.train` (producen los mismos bytes).
