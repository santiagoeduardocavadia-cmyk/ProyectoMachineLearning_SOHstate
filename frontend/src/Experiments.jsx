import React, { useState } from "react";
import { base, request, useApi } from "./api";
import { Inference } from "./Inference";
import {
  ErrorBox,
  Heading,
  Load,
  Plot,
  Table,
  Warnings,
  blue,
  green,
  measure,
  num,
  ochre,
  quantities,
  scale,
  series,
  targetName,
  unit,
} from "./shared";

const experimentLink = (id) => "#experimento/" + encodeURIComponent(id);
const errorUnit = (target) => (target === "soh" ? "pp" : unit[target]);
const errorValue = (v, target) =>
  v == null
    ? "—"
    : target === "soh"
      ? num(v, 5) + " (" + num(v * 100, 3) + " pp)"
      : measure(v, target);

export function History() {
  const [target, setTarget] = useState("");
  const history = useApi("/experiments" + (target ? "?target=" + target : ""));
  return (
    <>
      <Heading
        title="Experimentos"
        text="Compara el rendimiento de los modelos con baterías de prueba."
        action={
          <a className="primary" href="#entrenar">
            Entrenar modelo ↗
          </a>
        }
      />
      <section className="panel">
        <div className="panel-head">
          <h2>Historial</h2>
          <label>
            Objetivo
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Todos</option>
              {Object.entries(targetName).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </div>
        <Load query={history}>
          {(h) => (
            <Table
              label="Comparación de experimentos"
              headers={[
                "Experimento",
                "Objetivo",
                "Modelo",
                "Prueba",
                "MAE",
                "RMSE",
                "R²",
                "MAPE (%)",
              ]}
              rows={h.experiments.map((e) => [
                <a href={experimentLink(e.id)}>{e.name}</a>,
                targetName[e.target],
                e.model_type,
                e.test_batteries.join(", "),
                errorValue(e.metrics_test.mae, e.target),
                errorValue(e.metrics_test.rmse, e.target),
                num(e.metrics_test.r2, 4),
                num(e.metrics_test.mape_pct),
              ])}
            />
          )}
        </Load>
      </section>
    </>
  );
}
function Metrics({ experiment: e }) {
  const t = e.config.target,
    m = e.metrics,
    cv = m.cv;
  const scores = [
    ["MAE", "mae"],
    ["RMSE", "rmse"],
    ["R²", "r2"],
    ["MAPE (%)", "mape_pct"],
  ];
  const score = (k, v) =>
    ["mae", "rmse"].includes(k) ? errorValue(v, t) : num(v, 4);
  return (
    <>
      <section className="panel">
        <h2>Evaluación del modelo</h2>
        <p>
          La línea base predice la media del objetivo en el entrenamiento. Ambas
          columnas de prueba usan los mismos ciclos.
        </p>
        <Table
          label="Modelo frente a línea base"
          headers={[
            "Métrica",
            "Modelo · prueba",
            "Línea base · prueba",
            "Entrenamiento",
          ]}
          rows={scores.map(([label, k]) => [
            label,
            score(k, m.test[k]),
            score(k, m.baseline_test[k]),
            score(k, m.train[k]),
          ])}
        />
        {m.per_temperature && Object.keys(m.per_temperature).length > 0 && (
          <>
            <h3>Métricas por temperatura de prueba</h3>
            <Table
              label="Métricas por temperatura de prueba"
              headers={["Temperatura (°C)", "MAE", "RMSE", "R²"]}
              rows={Object.entries(m.per_temperature)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([temperature, values]) => [
                  num(Number(temperature)),
                  score("mae", values.mae),
                  score("rmse", values.rmse),
                  score("r2", values.r2),
                ])}
            />
          </>
        )}
        <h3>Validación cruzada</h3>
        {cv ? (
          <>
            <p>{cv.strategy}</p>
            <Table
              label="Resultados de validación cruzada"
              headers={["Métrica", "Media", "Desviación estándar"]}
              rows={scores
                .slice(0, 3)
                .map(([label, k]) => [
                  label,
                  score(k, cv[k + "_mean"]),
                  score(k, cv[k + "_std"]),
                ])}
            />
          </>
        ) : (
          <p>
            Se necesitan al menos dos baterías de entrenamiento para validar por
            batería.
          </p>
        )}
      </section>
      <section className="panel">
        <h2>Configuración guardada</h2>
        <dl className="specs">
          {[
            ["Modelo", e.config.model_type],
            ["Objetivo", targetName[t]],
            ...(e.domain
              ? [
                  [
                    "Temperaturas de entrenamiento",
                    quantities(e.domain.temperatures, "°C"),
                  ],
                  [
                    "Corrientes de entrenamiento",
                    quantities(e.domain.currents, "A"),
                  ],
                ]
              : []),
            ...(e.config.test_temperatures?.length
              ? [
                  [
                    "Temperaturas reservadas",
                    quantities(e.config.test_temperatures, "°C"),
                  ],
                ]
              : []),
            [
              "Entrenamiento",
              e.config.train_batteries.join(", ") +
                " · " +
                e.n_train +
                " ciclos",
            ],
            [
              "Prueba",
              e.config.test_batteries.join(", ") + " · " + e.n_test + " ciclos",
            ],
            ["Semilla", e.config.random_state],
            ["Características", e.config.features.join(", ")],
            [
              "Hiperparámetros",
              Object.entries(e.config.hyperparameters)
                .map(([k, v]) => k + " = " + (v ?? "sin límite"))
                .join(", "),
            ],
          ].map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      </section>
    </>
  );
}
function Charts({ experiment: e, path }) {
  const [split, setSplit] = useState("test");
  const cells =
    split === "test" ? e.config.test_batteries : e.config.train_batteries;
  const [chosen, choose] = useState("");
  const cell = cells.includes(chosen) ? chosen : cells[0];
  const points = useApi(
    path + "/predictions?" + new URLSearchParams({ split, battery_id: cell }),
  );
  const importance = useApi(path + "/feature-importance");
  const target = e.config.target;
  const axis = targetName[target] + " (" + unit[target] + ")";
  return (
    <>
      <section className="panel">
        <h2>Predicciones por ciclo</h2>
        <div className="fields">
          <label>
            Conjunto
            <select value={split} onChange={(ev) => setSplit(ev.target.value)}>
              <option value="test">Prueba</option>
              <option value="train">Entrenamiento</option>
            </select>
          </label>
          <label>
            Batería
            <select value={cell} onChange={(ev) => choose(ev.target.value)}>
              {cells.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
        </div>
        <Load query={points}>
          {(d) => {
            const values = d.points
              .filter((p) => p.y_true != null && p.y_pred != null)
              .sort((a, b) => a.cycle - b.cycle);
            if (!values.length)
              return <p>Sin predicciones para esta selección.</p>;
            const byCycle = (key) =>
              values.map((p) => ({ x: p.cycle, y: scale(p[key], target) }));
            const pairs = values.map((p) => ({
              x: scale(p.y_true, target),
              y: scale(p.y_pred, target),
            }));
            const lower = pairs.reduce(
              (n, p) => Math.min(n, p.x, p.y),
              Infinity,
            );
            const upper = pairs.reduce(
              (n, p) => Math.max(n, p.x, p.y),
              -Infinity,
            );
            return (
              <>
                <Plot
                  title="Predicción y valor real por ciclo"
                  x="Ciclo"
                  y={axis}
                  datasets={[
                    series("Real", byCycle("y_true")),
                    series("Predicho", byCycle("y_pred"), ochre),
                  ]}
                />
                <div className="chart-grid">
                  <div>
                    <h3>Predicho frente a real</h3>
                    <Plot
                      title="Dispersión de predicciones"
                      x={"Real (" + unit[target] + ")"}
                      y={"Predicho (" + unit[target] + ")"}
                      type="scatter"
                      datasets={[
                        series("Ciclos", pairs, green, { pointRadius: 2.5 }),
                        series(
                          "Predicción ideal",
                          [lower, upper].map((v) => ({ x: v, y: v })),
                          ochre,
                          { showLine: true, borderDash: [4, 4] },
                        ),
                      ]}
                    />
                  </div>
                  <div>
                    <h3>Residuales por ciclo</h3>
                    <Plot
                      title="Error de predicción por ciclo"
                      x="Ciclo"
                      y={"Predicho − real (" + errorUnit(target) + ")"}
                      datasets={[series("Residual", byCycle("error"), blue)]}
                    />
                  </div>
                </div>
              </>
            );
          }}
        </Load>
      </section>
      <section className="panel">
        <h2>Importancia de las variables</h2>
        <Load query={importance}>
          {(d) => (
            <>
              <p>{d.method}</p>
              {d.importances.length ? (
                <Plot
                  title="Importancia por permutación"
                  x="Cambio en MAE (unidad original)"
                  y=""
                  horizontal
                  type="bar"
                  labels={d.importances.map((f) => f.feature)}
                  datasets={[
                    {
                      label: "Cambio en MAE",
                      data: d.importances.map((f) => f.importance),
                      backgroundColor: green,
                    },
                  ]}
                />
              ) : (
                <p>Sin importancias disponibles.</p>
              )}
            </>
          )}
        </Load>
      </section>
    </>
  );
}
export function Detail({ id, supportsDomain }) {
  const path = "/experiments/" + encodeURIComponent(id);
  const detail = useApi(path);
  const [tab, setTab] = useState("Métricas");
  const [deleting, setDeleting] = useState(false),
    [error, setError] = useState(null);
  async function remove(name) {
    if (!window.confirm("¿Borrar «" + name + "» y sus resultados?")) return;
    setDeleting(true);
    try {
      await request(path, { method: "DELETE" });
      if (location.hash === experimentLink(id)) location.hash = "experimentos";
    } catch (err) {
      setError(err);
      setDeleting(false);
    }
  }
  return (
    <>
      <a className="back" href="#experimentos">
        ← Experimentos
      </a>
      <Load query={detail}>
        {(e) => (
          <>
            <Heading
              title={e.name}
              text={
                e.description ||
                targetName[e.config.target] + " · " + e.config.model_type
              }
            />
            <div className="actions">
              <a className="button" download href={base + path + "/model"}>
                Descargar modelo
              </a>
              <button disabled={deleting} onClick={() => remove(e.name)}>
                Borrar experimento
              </button>
            </div>
            <ErrorBox error={error} />
            <Warnings items={e.warnings} />
            <div className="tabs">
              {["Métricas", "Gráficos", "Inferencia"].map((v) => (
                <button
                  key={v}
                  aria-pressed={tab === v}
                  onClick={() => setTab(v)}
                >
                  {v}
                </button>
              ))}
            </div>
            {tab === "Métricas" && <Metrics experiment={e} />}
            {tab === "Gráficos" && <Charts experiment={e} path={path} />}
            {tab === "Inferencia" && (
              <Inference
                experiment={e}
                path={path}
                supportsDomain={supportsDomain}
              />
            )}
          </>
        )}
      </Load>
    </>
  );
}
