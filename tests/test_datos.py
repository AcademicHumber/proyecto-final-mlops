"""Los CSV locales deben coincidir con la versión registrada por DVC (archivos .dvc en Git)."""
import hashlib

import pytest
import yaml

from train.data import DATA_DIR

ARCHIVOS = ["train.csv", "test.csv", "produccion.csv"]


@pytest.mark.parametrize("nombre", ARCHIVOS)
def test_datos_coinciden_con_dvc(nombre):
    meta = yaml.safe_load((DATA_DIR / f"{nombre}.dvc").read_text(encoding="utf-8"))["outs"][0]
    md5 = hashlib.md5((DATA_DIR / nombre).read_bytes()).hexdigest()
    assert md5 == meta["md5"], (
        f"{nombre} no coincide con la versión registrada. Restaurar con `dvc checkout` o `dvc pull`."
    )
