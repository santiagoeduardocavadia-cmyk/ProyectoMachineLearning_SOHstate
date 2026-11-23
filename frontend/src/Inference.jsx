import React, { useState } from "react";
import { datasetPath, post, useApi } from "./api";
import {
  ErrorBox,
  Load,
  Table,
  Warnings,
  measure,
  num,
  quantities,
  scale,
  targetName,
  unit,
} from "./shared";

const autoMode = "Selección automática (el sistema elige el modelo)";
const domainColors = {
  dentro: "domain-inside",
  borde: "domain-edge",
  fuera: "domain-outside",
};

export function DomainNotice({ domain }) {
  if (!domain) return null;
  return (
    <div
      className={"domain-notice " + (domainColors[domain.status] || "")}
      role="status"
    >
      <strong>Fiabilidad: {domain.status || "sin clasificar"}</strong>
      <p>{domain.message}</p>
      {domain.unseen_temperatures?.length > 0 && (
        <p>
          Temperaturas no vistas: {quantities(domain.unseen_temperatures, "°C")}
        </p>
      )}
      {domain.unseen_currents?.length > 0 && (
        <p>Corrientes no vistas: {quantities(domain.unseen_currents, "A")}</p>
      )}
    </div>
  );
}

export function Candidates({ items, target, showHeading = true }) {
  return (
    <>
      {showHeading && <h3>Modelos candidatos</h3>}
      <Table
        label="Modelos candidatos"
        headers={[
          "Nombre",
          "Dataset",
          "Estado de dominio",
          "Distancia relativa mediana",
          "MAE de prueba (" +
            (target === "soh" ? "fracción" : unit[target]) +
            ")",
          "Motivo de exclusión",
        ]}
        rows={items.map((c) => [
          c.name ?? c.experiment_id,
          c.dataset_id,
          c.domain_status,
          num(c.distance_ratio_median, 3),
          num(c.test_mae, 5),
          c.eligible === false ? c.reason : null,
        ])}
      />
    </>
  );
}

export function PredictionTable({ result, children, showReliability = false }) {
  const [page, setPage] = useState(0);
  const { predictions, target, domain } = result;
  const showDomain =
    showReliability ||
    domain != null ||
    predictions.some((p) => p.domain_status != null);
  const start = page * 25;
  const headers = ["Muestra", "Predicción", "Real"];
  if (target === "soh") headers.push("Capacidad estimada (Ah)");
  if (target === "capacity") headers.push("SOH estimado (%)");
  if (target !== "rul") headers.push("Estado");
  if (showDomain) headers.push("Fiabilidad");
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
    if (showDomain) row.push(p.domain_status);
    return row;
  });
  return (
    <>
      <h3>Resultados · {predictions.length}</h3>
      {children}
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
function Predictions({ result }) {
  const { target, metrics, domain, selection } = result;
  const errorUnit = target === "soh" ? "pp" : unit[target];
  return (
    <>
      <Warnings items={result.warnings?.filter((w) => w !== domain?.message)} />
      {selection && (
        <>
          <h3>Modelo elegido</h3>
          <a
            href={
              "#experimento/" +
              encodeURIComponent(selection.selected_experiment_id)
            }
          >
            {selection.selected_experiment_id}
          </a>
          <p>{selection.reason}</p>
          <Candidates items={selection.candidates} target={target} />
        </>
      )}
      {metrics != null && (
        <>
          <h3>Métricas de inferencia</h3>
          <Table
            label="Métricas de inferencia"
            headers={[
              "MAE (" + errorUnit + ")",
              "RMSE (" + errorUnit + ")",
              "R²",
              "MAPE (%)",
              "Muestras",
            ]}
            rows={[
              [
                num(scale(metrics.mae, target), 4),
                num(scale(metrics.rmse, target), 4),
                num(metrics.r2, 4),
                num(metrics.mape_pct, 3),
                num(metrics.n),
              ],
            ]}
          />
        </>
      )}
      <PredictionTable result={result}>
        <DomainNotice domain={domain} />
      </PredictionTable>
    </>
  );
}
export function PredictForm({
  mode = autoMode,
  experiment,
  path,
  batteries = [],
  features = [],
  ResultView = Predictions,
  submitLabel = "Calcular predicción",
  formClassName,
}) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [autoTarget, setAutoTarget] = useState("soh");
  const automatic = mode === autoMode;
  const csvMode = automatic || mode === "Archivo CSV";
  const csvTarget = automatic ? autoTarget : experiment.config.target;
  const names = experiment?.config.features ?? [];
  async function predict(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    setResult(null);
    setError(null);
    try {
      let body,
        endpoint = automatic ? "/predict/auto" : path + "/predict";
      if (csvMode) {
        if (form.get("file").size > 5 * 1024 * 1024)
          throw new Error("El CSV debe pesar como máximo 5 MB.");
        body = form;
        if (!automatic) endpoint += "/csv";
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
      <form onSubmit={predict} className={formClassName}>
        <fieldset className="flat" disabled={busy}>
          {mode === "Por batería" && (
            <div className="fields">
              <label>
                Batería
                <select name="battery">
                  {batteries.map((b) => (
                    <option key={b.battery_id}>{b.battery_id}</option>
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
                      {f && (
                        <small>
                          Rango observado {num(f?.min, 3)} a {num(f?.max, 3)}
                        </small>
                      )}
                    </label>
                  );
                })}
              </div>
            </>
          )}
          {csvMode && (
            <>
              {automatic && (
                <label>
                  Objetivo de la selección automática
                  <select
                    name="target"
                    value={autoTarget}
                    onChange={(e) => {
                      setAutoTarget(e.target.value);
                      setResult(null);
                      setError(null);
                    }}
                  >
                    {Object.entries(targetName).map(([key, label]) => (
                      <option key={key} value={key}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label>
                Archivo CSV (máx. 5 MB)
                <input
                  type="file"
                  name="file"
                  accept=".csv,text/csv"
                  required
                />
              </label>
              {!automatic && (
                <p className="muted">
                  Columnas del modelo: {names.join(", ")}.
                </p>
              )}
              <p className="muted">
                Para obtener métricas, incluye la columna real{" "}
                {csvTarget === "capacity" ? "capacity_ah" : csvTarget}
                {!experiment && csvTarget === "soh" ? " o capacity_ah" : ""}.
              </p>
            </>
          )}
          <button className="primary" disabled={busy}>
            {busy ? "Calculando…" : submitLabel}
          </button>
        </fieldset>
      </form>
      <ErrorBox error={error} />
      {automatic &&
        error?.code === "no_compatible_experiment" &&
        Array.isArray(error.details?.candidates) && (
          <Candidates items={error.details.candidates} target={autoTarget} />
        )}
      {result && <ResultView result={result} />}
    </>
  );
}
export function Inference({ experiment, path, supportsDomain }) {
  const [mode, setMode] = useState("Por batería");
  const datasetId = experiment.config.dataset_id || "nasa";
  const summary = useApi(
    mode === "Por batería" ? datasetPath("/dataset/summary", datasetId) : null,
  );
  const features = useApi(
    mode === "Valores manuales"
      ? datasetPath("/dataset/features", datasetId)
      : null,
  );
  return (
    <section className="panel">
      <h2>Inferencia</h2>
      <p>
        {mode === autoMode
          ? "Carga un CSV para comparar los experimentos guardados del objetivo elegido."
          : "Predice " +
            targetName[experiment.config.target].toLowerCase() +
            " con este modelo."}
      </p>
      <div className="tabs">
        {[
          "Por batería",
          "Valores manuales",
          "Archivo CSV",
          ...(supportsDomain ? [autoMode] : []),
        ].map((m) => (
          <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>
            {m}
          </button>
        ))}
      </div>
      {mode === "Por batería" ? (
        <Load query={summary}>
          {(s) => (
            <PredictForm
              key={mode}
              mode={mode}
              experiment={experiment}
              path={path}
              batteries={s.batteries}
            />
          )}
        </Load>
      ) : (
        <PredictForm
          key={mode}
          mode={mode}
          experiment={experiment}
          path={path}
          features={features.data?.features}
        />
      )}
    </section>
  );
}
