import pandas as pd
import pytest

from app import config
from app.data.dataset import parse_upload
from scripts.augment_dataset import SEP, augment


@pytest.fixture(scope="module")
def nasa():
    return pd.read_csv(config.PROCESSED_FILE, dtype={"battery_id": str})


def test_test_intacto_y_sin_baterias_virtuales_de_test(nasa):
    out = augment(nasa, ["B0006"])
    real = out[~out["battery_id"].str.contains(SEP, regex=False)]
    assert len(real) == len(nasa)
    cols = ["battery_id", "cycle", "capacity_ah", "v_mean"]
    pd.testing.assert_frame_equal(real[real.battery_id == "B0006"][cols].reset_index(drop=True),
                                  nasa[nasa.battery_id == "B0006"][cols].reset_index(drop=True))
    virtual = out.loc[out["battery_id"].str.contains(SEP, regex=False), "battery_id"]
    assert not virtual.str.startswith("B0006").any()
    # 3 baterías de entrenamiento × (3 ruido + 2 interpolación + 2 mixup)
    assert virtual.nunique() == 3 * 7


def test_reproducible_y_subible(nasa):
    a, b = augment(nasa, ["B0006"], seed=1), augment(nasa, ["B0006"], seed=1)
    pd.testing.assert_frame_equal(a, b)
    df, _ = parse_upload(a.to_csv(index=False).encode(), config.NOMINAL_CAPACITY_AH, config.EOL_CAPACITY_AH)
    assert len(df) == len(a)


def test_errores(nasa):
    with pytest.raises(ValueError, match="inexistentes"):
        augment(nasa, ["B9999"])
    with pytest.raises(ValueError, match="dos veces"):
        augment(augment(nasa, ["B0006"]), ["B0006"])
