import React, { useState } from "react";
import { datasetPath, useApi } from "./api";
import {
  Heading,
  Load,
  Plot,
  Table,
  green,
  num,
  ochre,
  series,
} from "./shared";

function Degradation({ cell, nominal, datasetId }) {
  const query = useApi(
    datasetPath(
      "/dataset/batteries/" +
        encodeURIComponent(cell.battery_id) +
        "/degradation",
      datasetId,
    ),
  );
  const remaining = (100 * cell.final_capacity_ah) / nominal;
  return (
    <div className="analysis-grid">
      <section className="panel">
        <h2>Curva de degradación</h2>
        <Load query={query}>
          {({ points, eol_capacity_ah, eol_cycle }) => {
            if (!points.length)
              return <p>Esta celda aún no tiene ciclos registrados.</p>;
            const values = points.map((p) => ({
              x: p.cycle,
              y: p.capacity_ah,
            }));
            const datasets = [
              series("Capacidad medida", values),
              series(
                "Umbral de fin de vida",
                [values[0], values.at(-1)].map((p) => ({
                  x: p.x,
                  y: eol_capacity_ah,
                })),
                ochre,
                { borderDash: [6, 4] },
              ),
            ];
            if (eol_cycle != null)
              datasets.push(
                series(
                  "Ciclo de fin de vida",
                  [
                    {
                      x: eol_cycle,
                      y: values.reduce(
                        (low, p) => Math.min(low, p.y),
                        eol_capacity_ah,
                      ),
                    },
                    {
                      x: eol_cycle,
                      y: values.reduce(
                        (high, p) => Math.max(high, p.y),
                        eol_capacity_ah,
                      ),
                    },
                  ],
                  "#85978e",
                  { borderDash: [2, 4] },
                ),
              );
            return (
              <Plot
                title={"Capacidad por ciclo de " + cell.battery_id}
                x="Ciclo de descarga"
                y="Capacidad (Ah)"
                datasets={datasets}
              />
            );
          }}
        </Load>
      </section>
      <aside className="cell-card">
        <span className="eyebrow">Celda seleccionada</span>
        <h2>{cell.battery_id}</h2>
        <div className="battery" aria-hidden="true">
          <i style={{ width: Math.min(100, Math.max(0, remaining)) + "%" }} />
        </div>
        <strong>{num(remaining, 1)} %</strong>
        <span>Salud en el último ciclo</span>
        <dl>
          <div>
            <dt>Capacidad final</dt>
            <dd>{num(cell.final_capacity_ah, 3)} Ah</dd>
          </div>
          <div>
            <dt>Fin de vida</dt>
            <dd>
              {cell.eol_cycle == null
                ? "Sin observar"
                : "Ciclo " + cell.eol_cycle}
            </dd>
          </div>
        </dl>
      </aside>
    </div>
  );
}
function Cycles({ battery, datasetId }) {
  const [page, setPage] = useState(0);
  const query = useApi(
    datasetPath("/dataset/cycles", datasetId, {
      battery_id: battery,
      limit: 20,
      offset: page * 20,
    }),
  );
  return (
    <section className="panel">
      <h2>Ciclos de {battery}</h2>
      <Load query={query}>
        {(data) => (
          <>
            <Table
              label="Ciclos de descarga"
              headers={[
                "Ciclo",
                "Capacidad (Ah)",
                "SOH (%)",
                "RUL (ciclos)",
                "Voltaje medio (V)",
                "Temperatura máx. (°C)",
              ]}
              rows={data.rows.map((r) => [
                r.cycle,
                num(r.capacity_ah, 3),
                num(r.soh == null ? null : 100 * r.soh),
                num(r.rul),
                num(r.v_mean, 3),
                num(r.temp_max),
              ])}
            />
            <div className="pager">
              <span>
                {data.total ? data.offset + 1 : 0}–
                {data.offset + data.rows.length} de {data.total}
              </span>
              <button disabled={!page} onClick={() => setPage(page - 1)}>
                Anterior
              </button>
              <button
                disabled={data.offset + data.rows.length >= data.total}
                onClick={() => setPage(page + 1)}
              >
                Siguiente
              </button>
            </div>
          </>
        )}
      </Load>
    </section>
  );
}
export function Data({ dataset, selector }) {
  const summary = useApi(datasetPath("/dataset/summary", dataset.id));
  const features = useApi(datasetPath("/dataset/features", dataset.id));
  const [selected, select] = useState("");
  const [tab, setTab] = useState("Baterías");
  return (
    <>
      <Heading
        title="Datos de baterías"
        text="Capacidad y estado de salud a lo largo de los ciclos de descarga."
        action={
          <a className="primary" href="#entrenar">
            Entrenar modelo ↗
          </a>
        }
      />
      {selector}
      <Load query={summary}>
        {(s) => {
          const cell =
            s.batteries.find((b) => b.battery_id === selected) ||
            s.batteries[0];
          return (
            <>
              <div className="stats">
                {[
                  ["Ciclos registrados", num(s.n_rows)],
                  ["Baterías", num(s.n_batteries)],
                  ["Capacidad nominal", num(s.nominal_capacity_ah) + " Ah"],
                  ["Umbral de fin de vida", num(s.eol_capacity_ah) + " Ah"],
                ].map(([label, v]) => (
                  <div key={label}>
                    <span>{label}</span>
                    <strong>{v}</strong>
                  </div>
                ))}
              </div>
              {cell && (
                <>
                  <div className="section-line">
                    <span className="eyebrow">Explorar celda</span>
                    <div className="segmented">
                      {s.batteries.map((b) => (
                        <button
                          aria-pressed={b.battery_id === cell.battery_id}
                          key={b.battery_id}
                          onClick={() => select(b.battery_id)}
                        >
                          {b.battery_id}
                        </button>
                      ))}
                    </div>
                  </div>
                  <Degradation
                    cell={cell}
                    nominal={s.nominal_capacity_ah}
                    datasetId={dataset.id}
                  />
                </>
              )}
              <div className="tabs">
                {["Baterías", "Ciclos", "Características"].map((t) => (
                  <button
                    key={t}
                    aria-pressed={t === tab}
                    onClick={() => setTab(t)}
                  >
                    {t}
                  </button>
                ))}
              </div>
              {tab === "Baterías" && (
                <section className="panel">
                  <h2>Resumen por batería</h2>
                  <Table
                    label="Resumen por batería"
                    headers={[
                      "Batería",
                      "Ciclos",
                      "Inicial (Ah)",
                      "Final (Ah)",
                      "Ciclo de fin de vida",
                    ]}
                    rows={s.batteries.map((b) => [
                      b.battery_id,
                      b.n_cycles,
                      num(b.initial_capacity_ah, 3),
                      num(b.final_capacity_ah, 3),
                      b.eol_cycle ?? "Sin observar",
                    ])}
                  />
                  <p className="source">Fuente · {s.source}</p>
                </section>
              )}
              {tab === "Ciclos" && cell && (
                <Cycles
                  key={cell.battery_id}
                  battery={cell.battery_id}
                  datasetId={dataset.id}
                />
              )}
              {tab === "Características" && (
                <section className="panel">
                  <h2>Características de los ciclos</h2>
                  <Load query={features}>
                    {(d) => (
                      <Table
                        label="Características"
                        headers={[
                          "Variable",
                          "Unidad",
                          "Descripción",
                          "Mín.",
                          "Máx.",
                        ]}
                        rows={d.features.map((f) => [
                          f.name,
                          f.unit,
                          f.description,
                          num(f.min, 3),
                          num(f.max, 3),
                        ])}
                      />
                    )}
                  </Load>
                </section>
              )}
            </>
          );
        }}
      </Load>
    </>
  );
}
