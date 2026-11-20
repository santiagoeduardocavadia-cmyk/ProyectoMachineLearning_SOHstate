import io

import pandas as pd

from app import config


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _nasa_subset(batteries=("B0005", "B0006", "B0018"), cols=None):
    df = pd.read_csv(config.PROCESSED_FILE)
    df = df[df["battery_id"].isin(batteries)]
    cols = cols or ["battery_id", "cycle", "capacity_ah", "time_to_3_5v", "time_to_temp_max", "v_mean", "temp_max"]
    # renombrar baterías para simular un dataset propio
    return df[cols].assign(battery_id=lambda d: "MI_" + d["battery_id"])


def _upload(client, df, name="mis baterías", **form):
    files = {"file": ("mis_datos.csv", _csv(df), "text/csv")}
    return client.post("/api/v1/datasets", files=files, data={"name": name, **form})


def test_subir_entrenar_y_borrar_dataset(client):
    r = _upload(client, _nasa_subset())
    assert r.status_code == 201, r.text
    d = r.json()
    did = d["id"]
    assert d["n_batteries"] == 3 and d["source"] == "upload" and d["eol_capacity_ah"] == 1.4
    assert any("ausentes" in w for w in d["warnings"])

    ids = [x["id"] for x in client.get("/api/v1/datasets").json()["datasets"]]
    assert ids[0] == "nasa" and did in ids
    assert client.get(f"/api/v1/dataset/summary?dataset_id={did}").json()["n_rows"] == d["n_rows"]
    deg = client.get(f"/api/v1/dataset/batteries/MI_B0006/degradation?dataset_id={did}").json()
    assert deg["eol_cycle"] == 109

    body = {"name": "propio", "dataset_id": did, "test_batteries": ["MI_B0006"],
            "hyperparameters": {"n_estimators": 20}}
    exp = client.post("/api/v1/experiments", json=body)
    assert exp.status_code == 201, exp.text
    exp = exp.json()
    assert exp["config"]["dataset_id"] == did
    assert exp["config"]["features"] == ["cycle", "v_mean", "time_to_3_5v", "temp_max", "time_to_temp_max"]
    assert exp["metrics"]["test"]["mae"] < exp["metrics"]["baseline_test"]["mae"]

    pr = client.post(f"/api/v1/experiments/{exp['id']}/predict", json={"battery_id": "MI_B0006", "cycles": [10, 20]})
    assert pr.status_code == 200 and pr.json()["metrics"]["n"] == 2

    # pedir una característica que el dataset no tiene
    bad = client.post("/api/v1/experiments", json={**body, "features": ["re_ohm"]})
    assert bad.json()["error"]["code"] == "invalid_features"

    assert client.delete(f"/api/v1/datasets/{did}").status_code == 204
    assert client.get(f"/api/v1/datasets/{did}").json()["error"]["code"] == "dataset_not_found"
    assert client.delete("/api/v1/datasets/nasa").json()["error"]["code"] == "dataset_protected"


def test_validaciones_de_subida(client):
    def code(r):
        return r.status_code, r.json()["error"]["code"]

    df = _nasa_subset()
    assert code(_upload(client, df.drop(columns=["capacity_ah"]))) == (422, "invalid_dataset")
    assert code(_upload(client, df[["battery_id", "cycle", "capacity_ah"]])) == (422, "invalid_dataset")
    assert code(_upload(client, df[df["battery_id"] == "MI_B0005"])) == (422, "invalid_dataset")  # 1 batería
    dup = pd.concat([df, df.head(1)])
    assert code(_upload(client, dup)) == (422, "invalid_dataset")
    txt = df.astype({"v_mean": str}).copy()
    txt.loc[txt.index[0], "v_mean"] = "abc"
    r = _upload(client, txt)
    assert code(r) == (422, "invalid_dataset") and r.json()["error"]["details"]["columns"] == {"v_mean": 1}
    assert code(_upload(client, df, nominal_capacity_ah=2.0, eol_capacity_ah=3.0)) == (422, "invalid_dataset")
    files = {"file": ("x.txt", b"a,b", "text/plain")}
    assert code(client.post("/api/v1/datasets", files=files, data={"name": "x"})) == (415, "invalid_file")
    assert client.post("/api/v1/datasets", files={"file": ("x.csv", _csv(df), "text/csv")}).status_code == 422


def test_plantilla_y_metricas_csv(client, fast_experiment):
    t = client.get("/api/v1/datasets/template")
    assert t.status_code == 200 and t.text.startswith("battery_id,cycle,capacity_ah")

    df = pd.read_csv(config.PROCESSED_FILE)
    sub = df[df["battery_id"] == "B0006"].head(20)
    files = {"file": ("b6.csv", sub.to_csv(index=False).encode(), "text/csv")}
    r = client.post(f"/api/v1/experiments/{fast_experiment['id']}/predict/csv", files=files).json()
    assert r["metrics"]["n"] == 20 and r["predictions"][0]["actual"] is not None

    # sin columna soh pero con capacity_ah: el SOH real se deriva de la capacidad
    files = {"file": ("b6.csv", sub.drop(columns=["soh"]).to_csv(index=False).encode(), "text/csv")}
    r2 = client.post(f"/api/v1/experiments/{fast_experiment['id']}/predict/csv", files=files).json()
    assert r2["metrics"]["n"] == 20 and abs(r2["metrics"]["mae"] - r["metrics"]["mae"]) < 1e-3

    files = {"file": ("b6.csv", sub.drop(columns=["soh", "capacity_ah"]).to_csv(index=False).encode(), "text/csv")}
    r = client.post(f"/api/v1/experiments/{fast_experiment['id']}/predict/csv", files=files).json()
    assert r["metrics"] is None and any("no se calculan" in w for w in r["warnings"])
