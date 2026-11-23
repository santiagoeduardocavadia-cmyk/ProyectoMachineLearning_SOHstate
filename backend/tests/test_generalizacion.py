"""v1.2.0: dataset multitemperatura, split por temperatura, celda física, dominio y selección automática."""
import json

import pandas as pd

from app import config
from app.data.dataset import get_dataset
from app.ml.domain import physical_cell

FAST = {"n_estimators": 20}


def _csv_of(battery: str, dataset: str = "nasa_multitemp") -> bytes:
    df = get_dataset(dataset)
    df = df[df["battery_id"] == battery].drop(columns=["soh", "rul", "eol_cycle"])
    return df.to_csv(index=False).encode()


def _train(client, **body):
    base = {"name": "t", "target": "soh", "model_type": "ridge"}
    r = client.post("/api/v1/experiments", json={**base, **body})
    assert r.status_code == 201, r.text
    return r.json()


def test_celda_fisica():
    assert physical_cell("B0005~ruido1") == "B0005"
    assert physical_cell("B0042_04C") == "B0042"
    assert physical_cell("B0043_22C~mix2") == "B0043"
    assert physical_cell("MI_B0005") == "MI_B0005"


def test_dataset_multitemp_incluido(client):
    ds = {d["id"]: d for d in client.get("/api/v1/datasets").json()["datasets"]}
    m = ds["nasa_multitemp"]
    assert m["temperatures"] == [4, 22, 24, 43, 44]
    assert len({physical_cell(b["battery_id"]) for b in m["batteries"]}) >= 30
    assert client.delete("/api/v1/datasets/nasa_multitemp").status_code == 403
    feats = {f["name"] for f in client.get("/api/v1/dataset/features").json()["features"]}
    assert {"ambient_temperature", "discharge_current_a"} <= feats


def test_split_por_temperatura_y_sin_fuga_de_tramos(client):
    exp = _train(client, dataset_id="nasa_multitemp", test_temperatures=[43])
    test = exp["config"]["test_batteries"]
    assert set(test) == {"B0029", "B0030", "B0031", "B0032", "B0038_44C", "B0039_44C", "B0040_44C"}
    assert not set(test) & set(exp["config"]["train_batteries"])
    assert set(exp["metrics"]["per_temperature"]) == {"43", "44"}
    assert exp["domain"]["temperatures"] == [4.0, 22.0, 24.0]

    exp2 = _train(client, dataset_id="nasa_multitemp", test_batteries=["B0042_04C"])
    assert "B0042_22C" not in exp2["config"]["train_batteries"]  # misma celda física que el test
    assert any("misma celda" in w for w in exp2["warnings"])
    assert "celda física" in exp2["metrics"]["cv"]["strategy"]


def test_dataset_aumentado_cv_y_train_solo_reales(client):
    from scripts.augment_dataset import augment

    nasa = pd.read_csv(config.PROCESSED_FILE, dtype={"battery_id": str})
    aug = augment(nasa, ["B0006"], noise_copies=1, interpolation_copies=0, mixup_copies=0)
    r = client.post("/api/v1/datasets", files={"file": ("a.csv", aug.to_csv(index=False).encode(), "text/csv")},
                    data={"name": "aumentado"})
    assert r.status_code == 201, r.text
    exp = _train(client, dataset_id=r.json()["id"], test_batteries=["B0006"])
    assert "3 pliegues" in exp["metrics"]["cv"]["strategy"]  # 3 celdas físicas, no 6 battery_id
    assert exp["metrics"]["train"]["n"] == int((nasa["battery_id"] != "B0006").sum())
    assert any("aumentado" in w for w in exp["warnings"])


def test_dominio_avisa_fuera_de_rango(client):
    exp = _train(client, features=None)  # NASA a 24 °C y 2 A
    eid = exp["id"]
    fuera = client.post(f"/api/v1/experiments/{eid}/predict/csv",
                        files={"file": ("b.csv", _csv_of("B0031"), "text/csv")}).json()
    assert fuera["domain"]["status"] == "fuera"
    assert fuera["domain"]["unseen_temperatures"] == [43.0]
    assert all(p["domain_status"] == "fuera" for p in fuera["predictions"])
    assert any("Temperaturas no vistas" in w for w in fuera["warnings"])

    dentro = client.post(f"/api/v1/experiments/{eid}/predict", json={"battery_id": "B0005"}).json()
    assert dentro["domain"]["status"] == "dentro"


def test_dominio_en_experimento_antiguo_sin_campo_domain(client):
    exp = _train(client)
    path = config.EXPERIMENTS_DIR / exp["id"] / "experiment.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.pop("domain")
    path.write_text(json.dumps(rec), encoding="utf-8")
    r = client.post(f"/api/v1/experiments/{exp['id']}/predict/csv",
                    files={"file": ("b.csv", _csv_of("B0047"), "text/csv")}).json()
    assert r["domain"]["status"] == "fuera"


def test_seleccion_automatica(client):
    solo24 = _train(client, name="solo 24 C")
    multi = _train(client, name="multi", dataset_id="nasa_multitemp", test_batteries=["B0006", "B0031", "B0047"])
    r = client.post("/api/v1/predict/auto", files={"file": ("b.csv", _csv_of("B0031"), "text/csv")},
                    data={"target": "soh"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["selection"]["selected_experiment_id"] == multi["id"]
    ids = [c["experiment_id"] for c in body["selection"]["candidates"]]
    assert ids.index(multi["id"]) < ids.index(solo24["id"])
    assert body["metrics"]["mae"] < 0.05

    none = client.post("/api/v1/predict/auto", files={"file": ("b.csv", _csv_of("B0031"), "text/csv")},
                       data={"target": "rul"})
    assert none.status_code == 422 and none.json()["error"]["code"] == "no_compatible_experiment"


def test_dataset_subido_con_version_anterior_sin_corriente(client, tmp_path):
    df = get_dataset("nasa")
    old = df[df["battery_id"].isin(["B0005", "B0006"])].drop(columns=["discharge_current_a"])
    up = config.UPLOADS_DIR
    up.mkdir(parents=True, exist_ok=True)
    old.to_csv(up / "up_antiguo.csv", index=False)
    (up / "up_antiguo.json").write_text(json.dumps({
        "id": "up_antiguo", "name": "antiguo", "source": "upload", "is_default": False, "created_at": "2026-11-20T00:00:00+00:00",
        "nominal_capacity_ah": 2.0, "eol_capacity_ah": 1.4, "warnings": []}), encoding="utf-8")
    exp = _train(client, dataset_id="up_antiguo", test_batteries=["B0006"])
    assert "discharge_current_a" not in exp["config"]["features"]


def test_celdas_similares_en_seleccion_automatica(client):
    _train(client, name="multi", dataset_id="nasa_multitemp", test_batteries=["B0006", "B0031", "B0047"])
    for csv, temp in (("B0031", 43), ("B0047", 4)):
        body = client.post("/api/v1/predict/auto", files={"file": ("b.csv", _csv_of(csv), "text/csv")},
                           data={"target": "soh"}).json()
        cells = body["similar_cells"]
        assert len(cells) == 5 and cells[0]["distance"] <= cells[-1]["distance"]
        assert csv not in {c["battery_id"] for c in cells}  # la celda de test no está en entrenamiento
        assert all(abs(c["ambient_temperature"] - temp) <= 3 for c in cells[:3])
        assert cells[0]["battery_id"] in body["selection"]["similar_cells_summary"]


def test_vida_restante_en_dias(client):
    from app.ml import life

    _train(client, name="multi", dataset_id="nasa_multitemp", test_batteries=["B0006", "B0031", "B0047"])
    df = get_dataset("nasa_multitemp")
    sana = df[(df["battery_id"] == "B0006") & (df["cycle"] <= 40)].drop(columns=["soh", "rul", "eol_cycle"])
    body = client.post("/api/v1/predict/auto", files={"file": ("b.csv", sana.to_csv(index=False).encode(), "text/csv")},
                       data={"target": "soh"}).json()
    le = body["life_estimate"]
    assert le["eol_soh"] == 0.7 and not le["past_end_of_life"]
    assert le["fade_per_cycle"] == max(le["fade_trend_csv"], le["fade_similar_cells"])
    assert 30 < le["remaining_cycles"] < 200  # real: 69 ciclos (EOL en el ciclo 109)
    dias = {p["key"]: p["days"] for p in le["profiles"]}
    assert dias["dron"] < dias["laptop"] < dias["linterna"]
    assert dias["laptop"] == round(le["remaining_cycles"] / 1.0, 1)
    rc = [p["remaining_cycles"] for p in body["predictions"]]
    assert all(r is not None and r >= 0 for r in rc) and max(rc[:10]) >= rc[-1]

    # Celda que ya pasó el fin de vida: 0 ciclos restantes.
    vieja = df[df["battery_id"] == "B0006"].drop(columns=["soh", "rul", "eol_cycle"])
    le2 = client.post("/api/v1/predict/auto", files={"file": ("b.csv", vieja.to_csv(index=False).encode(), "text/csv")},
                      data={"target": "soh"}).json()["life_estimate"]
    assert le2["past_end_of_life"] and le2["remaining_cycles"] == 0

    # Sin columna cycle: se usa el ritmo de las celdas parecidas.
    est = life.estimate([{"prediction": 0.9}], "soh", [{"battery_id": "X", "soh_initial": 0.95, "soh_final": 0.75,
                                                         "n_cycles": 101}], nominal=2.0, eol_capacity=1.4)
    assert est["fade_trend_csv"] is None and abs(est["remaining_cycles"] - 100) < 1e-6
