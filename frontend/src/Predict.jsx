import React, { useState } from "react";
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

export function lifeText(days) {
  if (days == null || !Number.isFinite(days)) return "—";
  if (days <= 0) return "0 días";
  if (days < 60) return num(days, 0) + " días";
  if (days < 730) return num(days / 30.4, 1) + " meses";
  return num(days / 365, 1) + " años";
}

export function hoursText(hours) {
  if (hours == null || !Number.isFinite(hours)) return "—";
  const total = Math.round(hours * 60);
  if (total < 60) return total + " min";
  const h = Math.floor(total / 60);
  const m = total % 60;
  return m ? `${h} h ${m} min` : `${h} h`;
}

function ContinuousUse({ life }) {
  const c = life.continuous_use;
  if (!c?.cycle_hours) return null;
  const days = life.profiles.find((p) => p.key === "continuo")?.days;
  return (
    <p className="life-continuous">
      <strong>Un ciclo completo a uso constante dura ≈ {hoursText(c.cycle_hours)}</strong>
      : {hoursText(c.discharge_hours)} de descarga a {num(c.current_a, 1)} A
      más {hoursText(c.charge_hours)} de recarga (medidos en tu CSV). Usándola
      sin parar son ≈ {num(24 / c.cycle_hours, 1)} ciclos al día
      {!life.past_end_of_life && days != null && (
        <> y le quedarían ≈ {lifeText(days)}</>
      )}
      .
    </p>
  );
}

function LifeEstimate({ life, profileKey, setProfileKey }) {
  if (!life) return null;
  const profile =
    life.profiles.find((p) => p.key === profileKey) || life.profiles[0];
  return (
    <section className="panel life-panel">
      <h2>Vida restante</h2>
      <div className="life-grid">
        <div>
          <label>
            Uso de la batería
            <select
              value={profile.key}
              onChange={(e) => setProfileKey(e.target.value)}
            >
              {life.profiles.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.name} · {num(p.cycles_per_day)} ciclos/día
                </option>
              ))}
            </select>
          </label>
          {life.past_end_of_life ? (
            <p className="life-big life-dead">
              Ya pasó el fin de vida
              <small>
                SOH actual {measure(life.soh_current, "soh")}, por debajo del{" "}
                {num(life.eol_soh * 100, 0)} %. Se recomienda reemplazarla.
              </small>
            </p>
          ) : (
            <p className="life-big">
              ≈ {lifeText(profile.days)}
              <small>
                {num(life.remaining_cycles, 0)} ciclos restantes ÷{" "}
                {num(profile.cycles_per_day)} ciclos/día = {num(profile.days, 0)}{" "}
                días
              </small>
              {profile.hours_per_charge != null && (
                <small>
                  Cada carga dura ≈ {hoursText(profile.hours_per_charge)} (
                  {num(profile.current_a, 1)} A por celda), unas{" "}
                  {hoursText(profile.use_hours_per_day)} de uso al día.
                </small>
              )}
            </p>
          )}
          <dl>
            {life.soh_current != null && (
              <div>
                <dt>SOH actual (predicho)</dt>
                <dd>{measure(life.soh_current, "soh")}</dd>
              </div>
            )}
            {life.capacity_current_ah != null && (
              <div>
                <dt>Capacidad actual</dt>
                <dd>{num(life.capacity_current_ah, 2)} Ah</dd>
              </div>
            )}
            {life.fade_per_cycle != null && (
              <div>
                <dt>Pérdida de SOH por ciclo</dt>
                <dd>{num(life.fade_per_cycle * 100, 3)} %</dd>
              </div>
            )}
            <div>
              <dt>Fin de vida</dt>
              <dd>SOH {num(life.eol_soh * 100, 0)} %</dd>
            </div>
          </dl>
        </div>
        <Table
          label="Vida restante por tipo de uso"
          headers={[
            "Uso",
            "Corriente",
            "Una carga dura",
            "Uso al día",
            "Ciclos por día",
            "Vida restante",
          ]}
          rows={life.profiles.map((p) => [
            p.key === profile.key ? <strong>{p.name}</strong> : p.name,
            num(p.current_a, 1) + " A",
            hoursText(p.hours_per_charge),
            hoursText(p.use_hours_per_day),
            num(p.cycles_per_day),
            lifeText(p.days),
          ])}
        />
      </div>
      <ContinuousUse life={life} />
      <p className="muted">
        Cómo se calcula: ciclos restantes = (SOH actual − 70 %) ÷ pérdida por
        ciclo, usando {life.method}.
      </p>
      <small>{life.assumptions}</small>
    </section>
  );
}

function Results({ result }) {
  const [profileKey, setProfileKey] = useState("laptop");
  const life = result.life_estimate;
  const profile =
    life?.profiles.find((p) => p.key === profileKey) || life?.profiles[0];
  const extraColumns =
    life && profile
      ? [
          {
            header: "Ciclos restantes",
            cell: (p) => num(p.remaining_cycles, 0),
          },
          {
            header: "Días de vida (" + profile.name.split(" (")[0] + ")",
            cell: (p) =>
              p.remaining_cycles == null
                ? "—"
                : num(p.remaining_cycles / profile.cycles_per_day, 0),
          },
        ]
      : [];
  return (
    <>
      <Summary result={result} />
      <Warnings
        items={result.warnings?.filter((w) => w !== result.domain?.message)}
      />
      <LifeEstimate
        life={life}
        profileKey={profileKey}
        setProfileKey={setProfileKey}
      />
      <SimilarCells result={result} />
      <Comparison predictions={result.predictions} target={result.target} />
      <section className="panel">
        <PredictionTable
          result={result}
          showReliability
          extraColumns={extraColumns}
        >
          <p className="muted">
            Cada fila es un ciclo de carga y descarga de la batería, en orden de
            tiempo: al avanzar de página la celda envejece y su estado pasa de
            bueno (SOH ≥ 85 %) a degradado y a fin de vida (SOH &lt; 70 %).
            "Días de vida" usa el perfil elegido en Vida restante.
          </p>
        </PredictionTable>
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
