import numpy as np

from app.data.features import _time_to_threshold, discharge_features
from app.data.preprocessing import add_targets
from app.ml.models import resolve_hyperparameters
from app.errors import AppError
import pandas as pd
import pytest


def test_time_to_threshold_interpola():
    t = np.array([0.0, 10.0, 20.0])
    v = np.array([4.0, 3.9, 3.7])
    assert _time_to_threshold(t, v, 3.8) == pytest.approx(15.0)
    assert np.isnan(_time_to_threshold(t, v, 3.0))


def test_discharge_features_ignora_rebote():
    t = np.arange(0, 1000, 10.0)
    v = np.concatenate([np.linspace(4.2, 2.7, 90), np.linspace(3.0, 3.3, 10)])  # rebote al final
    temp = np.linspace(24, 40, 100)
    f = discharge_features({"Time": t, "Voltage_measured": v, "Temperature_measured": temp})
    assert f["v_mean"] == pytest.approx(np.mean(v[:90]))
    assert f["temp_rise"] == pytest.approx(16.0)


def test_rul_y_censura():
    df = pd.DataFrame({"battery_id": ["A"] * 4 + ["B"] * 2, "cycle": [1, 2, 3, 4, 1, 2],
                       "capacity_ah": [1.6, 1.5, 1.39, 1.41, 1.9, 1.8]})
    out = add_targets(df).set_index(["battery_id", "cycle"])
    assert out.loc[("A", 1), "rul"] == 2 and out.loc[("A", 3), "rul"] == 0 and out.loc[("A", 4), "rul"] == 0
    assert out.loc[("B", 1), "rul"] != out.loc[("B", 1), "rul"]  # NaN: nunca llega al EOL
    assert out.loc[("A", 1), "soh"] == pytest.approx(0.8)


def test_hiperparametros_validacion():
    assert resolve_hyperparameters("ridge", {})["alpha"] == 1.0
    with pytest.raises(AppError):
        resolve_hyperparameters("ridge", {"alpha": -1})
    with pytest.raises(AppError):
        resolve_hyperparameters("ridge", {"gamma": 1})
    with pytest.raises(AppError):
        resolve_hyperparameters("random_forest", {"n_estimators": 10.5})
