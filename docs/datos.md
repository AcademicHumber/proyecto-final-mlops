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

Tabla de archivos (la v1 y la v2 se explican abajo, en "Versiones de los datos").

| Archivo | Qué es | Hash MD5 |
|---|---|---|
| `data/train.csv` | 60 %: entrenamiento y baseline del drift | `0b6e063f1d79…` |
| `data/test.csv` | 20 %: evaluación | `3506b073e70a…` |
| `data/produccion.csv` | 20 %: tráfico futuro simulado | `9c69d817222e…` |
| `data/produccion_concept_drift.csv` | v2: producción con concept drift | ver su `.dvc` |
| `data/produccion_data_drift.csv` | v2: producción con data drift | ver su `.dvc` |

En Git están `data/*.csv.dvc`, `data/.gitignore` (generado por DVC para no subir los CSV) y `.dvc/config`. Los CSV no van a Git.

## Versiones de los datos

Cada versión es una etiqueta de Git. `git checkout <etiqueta>` y luego `dvc checkout` dejan en `data/` exactamente esos archivos.

| Etiqueta | Contenido | Para qué |
|---|---|---|
| `data-v1` | `train.csv`, `test.csv`, `produccion.csv` | Entrenar el modelo original y el baseline del drift |
| `data-v2` | v1 **más** `produccion_concept_drift.csv` y `produccion_data_drift.csv` | Simular drift y entrenar el modelo corregido |

La v2 **no reemplaza** nada de la v1: la amplía. Los archivos nuevos salen de `produccion.csv` con `python -m drift.make_data_v2` (semillas fijas, mismo resultado siempre).

| Archivo nuevo | Filas | Qué tiene |
|---|---|---|
| `produccion_concept_drift.csv` | 9.769 | Producción en 12 lotes (columna `lote`, 1 a 12). Lotes 1-4 limpios; desde el 5 cambia la relación entre las X y el ingreso, con intensidad creciente. **Las X son idénticas a las de v1**, solo cambian las etiquetas. El % de `>50K` baja de 0.25 (lote 1) a 0.16 (lote 12). |
| `produccion_data_drift.csv` | 9.769 | Toda producción con las X alteradas (edad +8, −6 horas, más ganancias de capital, más autónomos y mujeres). Las etiquetas no cambian. |

La prueba `test_v2_se_regenera_exactamente_desde_v1` comprueba que estos archivos se reproducen desde la v1: no son datos opacos, y quien dude puede regenerarlos.

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
