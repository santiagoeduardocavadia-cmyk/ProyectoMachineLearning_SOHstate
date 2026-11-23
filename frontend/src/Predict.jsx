import React from "react";
import {
  Candidates,
  DomainNotice,
  PredictForm,
  PredictionTable,
} from "./Inference";
import {
  Heading,
  Plot,
  Table,
  Warnings,
  green,
  measure,
  num,
  ochre,
  scale,
  series,
  targetName,
  unit,
} from "./shared";

function Summary({ result }) {
  const { selection, domain, metrics, target } = result;
  const id = selection.selected_experiment_id;
  const chosen = selection.candidates.find((c) => c.experiment_id === id);
  const errorUnit = target === "soh" ? "pp" : unit[target];
  const reference = {
    soh: "capacity_ah ni de soh",
    capacity: "capacity_ah",
    rul: "rul",
  };
  return (
    <div className="prediction-summary">
      <section className="panel">
        <h2>Modelo elegido</h2>
        <a href={"#experimento/" + encodeURIComponent(id)}>
          {chosen?.name || id}
        </a>
        <p>{selection.reason}</p>
      </section>
      <section className="panel">
        <h2>Fiabilidad</h2>
        {domain ? (
          <DomainNotice domain={domain} />
        ) : (
          <p>El modelo no dispone de una evaluación de dominio.</p>
        )}
      </section>
      <section className="panel">
        <h2>Métricas</h2>
        {metrics ? (
          <dl>
            {[
              ["MAE (" + errorUnit + ")", num(scale(metrics.mae, target), 4)],
              ["RMSE (" + errorUnit + ")", num(scale(metrics.rmse, target), 4)],
              ["R²", num(metrics.r2, 4)],
            ].map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p>
            Sin métricas. El CSV no trae valores reales utilizables de{" "}
            {reference[target]} para calcular el error.
          </p>
        )}
      </section>
    </div>
  );
}

function SimilarCells({ result }) {
  const cells = result.similar_cells ?? [];
  return (
    <section className="panel">
      <h2>Tu batería se parece a…</h2>
      {result.selection.similar_cells_summary && (
        <p>{result.selection.similar_cells_summary}</p>
      )}
      {cells.length ? (
        <Table
          label="Celdas de entrenamiento similares"
          headers={[
            "Batería",
            "Temperatura (°C)",
            "Corriente (A)",
            "Ciclos",
            "SOH inicial → final",
            "Distancia (menor = más parecida)",
          ]}
          rows={cells.map((c) => [
            c.battery_id,
            num(c.ambient_temperature),
            num(c.discharge_current_a),
            num(c.n_cycles),
            measure(c.soh_initial, "soh") + " → " + measure(c.soh_final, "soh"),
            num(c.distance, 4),
          ])}
        />
      ) : (
        <p>El modelo no devolvió celdas de entrenamiento comparables.</p>
      )}
      <small>
        Una celda nueva no es ninguna de las entrenadas; esto indica a cuáles se
        parece
      </small>
    </section>
  );
}

function Comparison({ predictions, target }) {
  const actualCount = predictions.filter((p) =>
    Number.isFinite(p.actual),
  ).length;
  if (!actualCount) return null;
  const useCycles = predictions.every((p) => Number.isFinite(p.cycle));
  const points = predictions
    .map((p, i) => ({
      x: useCycles ? p.cycle : i + 1,
      actual: scale(p.actual, target),
      prediction: scale(p.prediction, target),
    }))
    .sort((a, b) => a.x - b.x);
  const datasets = [
    series(
      "Real",
      points.map((p) => ({ x: p.x, y: p.actual })),
      green,
      { pointRadius: actualCount < 40 ? 3 : 1 },
    ),
    series(
      "Predicho",
      points.map((p) => ({ x: p.x, y: p.prediction })),
      ochre,
      { pointRadius: points.length < 40 ? 2 : 0 },
    ),
  ];
  return (
    <section className="panel">
      <h2>Predicho vs. real</h2>
      <Plot
        title="Predicción y valor real"
        datasets={datasets}
        x={useCycles ? "Ciclo" : "Fila del CSV"}
        y={targetName[target] + " (" + unit[target] + ")"}
      />
    </section>
  );
}

function Results({ result }) {
  return (
    <>
      <Summary result={result} />
      <Warnings
        items={result.warnings?.filter((w) => w !== result.domain?.message)}
      />
      <SimilarCells result={result} />
      <Comparison predictions={result.predictions} target={result.target} />
      <section className="panel">
        <PredictionTable result={result} showReliability />
      </section>
      <details className="panel prediction-candidates">
        <summary>Modelos considerados</summary>
        <Candidates
          items={result.selection.candidates}
          target={result.target}
          showHeading={false}
        />
      </details>
    </>
  );
}

export function Predict() {
  return (
    <>
      <Heading
        title="Predecir"
        text="Sube el CSV de una batería. El sistema elige el modelo entrenado que mejor conoce esas condiciones, te dice a qué celdas se parece y, si el CSV trae la capacidad o el SOH real, calcula el error."
      />
      <PredictForm
        formClassName="panel prediction-upload"
        ResultView={Results}
        submitLabel="Predecir"
      />
    </>
  );
}
