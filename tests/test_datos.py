"""Los CSV locales deben coincidir con la versión registrada por DVC (archivos .dvc en Git)."""
import hashlib

import pandas as pd
import pytest
import yaml

from drift.make_data_v2 import build_v2
from train.data import DATA_DIR

V1 = ["train.csv", "test.csv", "produccion.csv", "datos.csv"]
V2 = ["produccion_concept_drift.csv", "produccion_data_drift.csv"]


@pytest.mark.parametrize("nombre", V1 + V2)
def test_datos_coinciden_con_dvc(nombre):
    meta = yaml.safe_load((DATA_DIR / f"{nombre}.dvc").read_text(encoding="utf-8"))["outs"][0]
    md5 = hashlib.md5((DATA_DIR / nombre).read_bytes()).hexdigest()
    assert md5 == meta["md5"], (
        f"{nombre} no coincide con la versión registrada. Restaurar con `dvc checkout` o `dvc pull`."
    )


@pytest.mark.parametrize("nombre", V2)
def test_v2_se_regenera_exactamente_desde_v1(nombre):
    """La v2 no es un archivo opaco: sale de v1 con semillas fijas, y se puede reproducir."""
    regenerado = build_v2(pd.read_csv(DATA_DIR / "produccion.csv"))[nombre]
    pd.testing.assert_frame_equal(regenerado, pd.read_csv(DATA_DIR / nombre))
