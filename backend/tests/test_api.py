def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["dataset"]["loaded"] is True


def test_dataset_endpoints(client):
    s = client.get("/api/v1/dataset/summary").json()
    assert s["n_batteries"] == 4 and s["n_rows"] > 600
    d = client.get("/api/v1/dataset/batteries/B0005/degradation").json()
    assert d["eol_cycle"] is not None and len(d["points"]) == 168
    assert client.get("/api/v1/dataset/batteries/B9999/degradation").json()["error"]["code"] == "battery_not_found"
    f = client.get("/api/v1/dataset/features").json()
    assert any(x["name"] == "time_to_3_5v" for x in f["features"])
    assert client.get("/api/v1/dataset/cycles?battery_id=B0018&limit=5").json()["total"] == 132


def test_models(client):
    m = client.get("/api/v1/models").json()
    assert {"random_forest", "ridge"} <= {x["key"] for x in m["models"]}


def test_experimento_punta_a_punta(client, fast_experiment):
    exp = fast_experiment
    assert exp["config"]["test_batteries"] == ["B0006"]
    assert "B0006" not in exp["config"]["train_batteries"]
    t = exp["metrics"]["test"]
    # El modelo debe superar claramente a la línea base que predice la media.
    assert t["mae"] < exp["metrics"]["baseline_test"]["mae"]
    assert t["r2"] > 0.5

    eid = exp["id"]
    assert client.get("/api/v1/experiments").json()["experiments"][0]["id"] == eid
    assert client.get(f"/api/v1/experiments/{eid}").json()["id"] == eid
    p = client.get(f"/api/v1/experiments/{eid}/predictions?split=test").json()
    assert len(p["points"]) == 168 and {"y_true", "y_pred", "error"} <= set(p["points"][0])
    assert len(client.get(f"/api/v1/experiments/{eid}/feature-importance").json()["importances"]) == len(exp["config"]["features"])
    assert client.get(f"/api/v1/experiments/{eid}/model").status_code == 200

    pr = client.post(f"/api/v1/experiments/{eid}/predict", json={"battery_id": "B0006", "cycles": [1, 50]}).json()
    assert len(pr["predictions"]) == 2 and pr["predictions"][0]["health_status"] in {"bueno", "degradado", "fin_de_vida"}

    sample = {f: None for f in exp["config"]["features"]}
    sample["cycle"] = 10
    pr2 = client.post(f"/api/v1/experiments/{eid}/predict", json={"samples": [sample]}).json()
    assert 0 < pr2["predictions"][0]["prediction"] < 1.2 and pr2["warnings"]

    assert client.delete(f"/api/v1/experiments/{eid}").status_code == 204
    assert client.get(f"/api/v1/experiments/{eid}").status_code == 404


def test_rul(client):
    r = client.post("/api/v1/experiments", json={"name": "rul", "target": "rul", "model_type": "ridge",
                                                 "test_batteries": ["B0006"]})
    assert r.status_code == 201
    exp = r.json()
    assert "B0007" not in exp["config"]["train_batteries"] and exp["warnings"]


def test_errores_de_entrada(client, fast_experiment):
    def code(r):
        return r.status_code, r.json()["error"]["code"]

    assert code(client.post("/api/v1/experiments", json={"target": "soh"})) == (422, "validation_error")
    assert code(client.post("/api/v1/experiments", json={"name": "x", "model_type": "xgb"})) == (422, "unknown_model")
    assert code(client.post("/api/v1/experiments", json={"name": "x", "test_batteries": ["B0005", "B0006", "B0007", "B0018"]})) == (422, "invalid_split")
    assert code(client.post("/api/v1/experiments", json={"name": "x", "target": "rul", "test_batteries": ["B0007"]})) == (422, "invalid_split")
    assert code(client.post("/api/v1/experiments", json={"name": "x", "features": ["inventada"]})) == (422, "invalid_features")
    assert code(client.post("/api/v1/experiments", json={"name": "x", "campo_extra": 1})) == (422, "validation_error")
    assert code(client.get("/api/v1/experiments/no-existe")) == (404, "experiment_not_found")
    assert code(client.get("/api/v1/experiments/../../etc")) [0] == 404

    eid = fast_experiment["id"]
    assert code(client.post(f"/api/v1/experiments/{eid}/predict", json={})) == (422, "validation_error")
    assert code(client.post(f"/api/v1/experiments/{eid}/predict", json={"samples": [{"cycle": 1}]})) == (422, "invalid_samples")
    assert code(client.post(f"/api/v1/experiments/{eid}/predict", json={"battery_id": "B0006", "cycles": [999]})) == (404, "cycle_not_found")
    files = {"file": ("x.csv", b"cycle,v_mean\n1,abc\n", "text/csv")}
    assert client.post(f"/api/v1/experiments/{eid}/predict/csv", files=files).status_code == 422
    files = {"file": ("x.txt", b"hola", "text/plain")}
    assert code(client.post(f"/api/v1/experiments/{eid}/predict/csv", files=files)) == (415, "invalid_file")


def test_cors(client):
    r = client.options("/api/v1/experiments", headers={"Origin": "http://localhost:5173",
                                                        "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") in {"*", "http://localhost:5173"}
