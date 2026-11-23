import pandas as pd

from app import config
from scripts.clean_cycles import clean_cycles


def _battery(bid, caps, temps):
    return pd.DataFrame({"battery_id": bid, "cycle": range(1, len(caps) + 1), "capacity_ah": caps,
                         "ambient_temperature": temps, "v_mean": 3.5})


def test_nasa_no_se_toca():
    nasa = pd.read_csv(config.PROCESSED_FILE, dtype={"battery_id": str})
    clean, report = clean_cycles(nasa)
    assert len(clean) == len(nasa) and all(r["status"] == "ok" for r in report)


def test_separa_por_temperatura_y_descarta_tramos_cortos():
    df = _battery("B1", [1.0] * 5 + [1.8 - 0.01 * i for i in range(20)], [24] * 5 + [44] * 20)
    clean, _ = clean_cycles(df)
    assert set(clean["battery_id"]) == {"B1_44C"}
    assert clean["cycle"].min() == 6  # conserva la numeración original


def test_descargas_cortadas_y_caidas_aisladas():
    caps = [0.05] * 5 + [1.8 - 0.01 * i for i in range(20)]
    caps[15] = 0.6  # caída aislada
    clean, report = clean_cycles(_battery("B2", caps, [4] * 25))
    assert len(clean) == 19 and clean["capacity_ah"].min() > 1.0
    assert report[0]["dropouts"] == 5 and report[0]["isolated_drops"] == 1


def test_recuperacion_de_capacidad_se_conserva():
    caps = [1.8 - 0.02 * i for i in range(20)]
    caps[10] += 0.08  # subida tras reposo: real
    clean, _ = clean_cycles(_battery("B3", caps, [24] * 20))
    assert len(clean) == 20
