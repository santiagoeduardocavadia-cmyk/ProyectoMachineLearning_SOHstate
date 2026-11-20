import React, { useState } from "react";
import { post } from "./api";
import {
  ErrorBox,
  Load,
  Table,
  Warnings,
  measure,
  num,
  targetName,
} from "./shared";

function Predictions({ result }) {
  const [page, setPage] = useState(0);
  const { predictions, target } = result;
  const start = page * 25;
  const headers = ["Muestra", "Predicción", "Real"];
  if (target === "soh") headers.push("Capacidad estimada (Ah)");
  if (target === "capacity") headers.push("SOH estimado (%)");
  if (target !== "rul") headers.push("Estado");
  const rows = predictions.slice(start, start + 25).map((p, i) => {
    const row = [
      p.battery_id
        ? [p.battery_id, p.cycle].filter((v) => v != null).join(" · ")
        : start + i + 1,
      measure(p.prediction, target),
      measure(p.actual, target),
    ];
    if (target === "soh") row.push(num(p.capacity_ah_estimated, 3));
    if (target === "capacity")
      row.push(p.soh_estimated == null ? "—" : num(p.soh_estimated * 100));
    if (target !== "rul")
      row.push(
        <span className={"health " + p.health_status}>
          {p.health_status?.replaceAll("_", " ") || "—"}
        </span>,
      );
    return row;
  });
  return (
    <>
      <Warnings items={result.warnings} />
      <h3>Resultados · {predictions.length}</h3>
      <Table headers={headers} rows={rows} label="Resultados de inferencia" />
      {predictions.length > 25 && (
        <div className="pager">
          <span>
            Página {page + 1} de {Math.ceil(predictions.length / 25)}
          </span>
          <button onClick={() => setPage(page - 1)} disabled={!page}>
            Anterior
          </button>
          <button
            onClick={() => setPage(page + 1)}
            disabled={start + 25 >= predictions.length}
          >
            Siguiente
          </button>
        </div>
      )}
    </>
  );
}
function PredictForm({ mode, experiment, path, catalog, features }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const names = experiment.config.features;
  async function predict(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    setResult(null);
    setError(null);
    try {
      let body,
        endpoint = path + "/predict";
      if (mode === "Archivo CSV") {
        if (form.get("file").size > 5 * 1024 * 1024)
          throw new Error("El CSV debe pesar como máximo 5 MB.");
        body = form;
        endpoint += "/csv";
      } else if (mode === "Valores manuales") {
        body = {
          samples: [
            Object.fromEntries(
              names.map((n) => [
                n,
                form.get(n) === "" ? null : Number(form.get(n)),
              ]),
            ),
          ],
        };
      } else {
        body = { battery_id: form.get("battery") };
        const cycles = form.get("cycles").trim();
        if (cycles) {
          const parts = cycles.split(",").map((s) => s.trim());
          if (parts.some((p) => !/^\d+$/.test(p) || Number(p) < 1))
            throw new Error(
              "Escribe ciclos enteros positivos separados por comas.",
            );
          body.cycles = parts.map(Number);
        }
      }
      setResult(await post(endpoint, body));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <form onSubmit={predict}>
        <fieldset className="flat" disabled={busy}>
          {mode === "Por batería" && (
            <div className="fields">
              <label>
                Batería
                <select name="battery">
                  {catalog.batteries.map((b) => (
                    <option key={b}>{b}</option>
                  ))}
                </select>
              </label>
              <label>
                Ciclos opcionales
                <input name="cycles" placeholder="Ej. 1, 25, 100" />
                <small>
                  Deja el campo vacío para predecir todos los ciclos.
                </small>
              </label>
            </div>
          )}
          {mode === "Valores manuales" && (
            <>
              <p>Los campos vacíos usan la mediana del entrenamiento.</p>
              <div className="fields">
                {names.map((n) => {
                  const f = features.find((v) => v.name === n);
                  return (
                    <label key={n}>
                      {n} {f?.unit && "(" + f.unit + ")"}
                      <input
                        name={n}
                        type="number"
                        step="any"
                        placeholder={String(
                          experiment.train_feature_medians?.[n] ?? "",
                        )}
                      />
                      <small>
                        Rango observado {num(f?.min, 3)} a {num(f?.max, 3)}
                      </small>
                    </label>
                  );
                })}
              </div>
            </>
          )}
          {mode === "Archivo CSV" && (
            <>
              <label>
                Archivo CSV (máx. 5 MB)
                <input
                  type="file"
                  name="file"
                  accept=".csv,text/csv"
                  required
                />
              </label>
              <p className="muted">Columnas del modelo: {names.join(", ")}.</p>
            </>
          )}
          <button className="primary" disabled={busy}>
            {busy ? "Calculando…" : "Calcular predicción"}
          </button>
        </fieldset>
      </form>
      <ErrorBox error={error} />
      {result && <Predictions result={result} />}
    </>
  );
}
export function Inference({ experiment, path, models, features }) {
  const [mode, setMode] = useState("Por batería");
  return (
    <section className="panel">
      <h2>Inferencia</h2>
      <p>
        Predice {targetName[experiment.config.target].toLowerCase()} con este
        modelo.
      </p>
      <div className="tabs">
        {["Por batería", "Valores manuales", "Archivo CSV"].map((m) => (
          <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>
            {m}
          </button>
        ))}
      </div>
      <Load query={models}>
        {(catalog) => (
          <Load query={features}>
            {(f) => (
              <PredictForm
                key={mode}
                mode={mode}
                experiment={experiment}
                path={path}
                catalog={catalog}
                features={f.features}
              />
            )}
          </Load>
        )}
      </Load>
    </section>
  );
}
