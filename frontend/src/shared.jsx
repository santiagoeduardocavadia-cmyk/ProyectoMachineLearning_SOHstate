import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

export const num = (value, digits = 2) =>
  Number.isFinite(value)
    ? value.toLocaleString("es-CO", { maximumFractionDigits: digits })
    : "—";
export const targetName = {
  soh: "Estado de salud",
  rul: "Vida útil restante",
  capacity: "Capacidad",
};
export const unit = { soh: "%", rul: "ciclos", capacity: "Ah" };
export const scale = (value, target) =>
  value == null ? null : target === "soh" ? value * 100 : value;
export const measure = (v, target) =>
  v == null ? "—" : num(scale(v, target), 3) + " " + unit[target];
export const green = "#187769",
  ochre = "#ad702e",
  blue = "#4b72aa";

export function ErrorBox({ error, retry }) {
  if (!error) return null;
  return (
    <div role="alert" className="error">
      {error.message}
      {retry && (
        <button type="button" onClick={retry}>
          Reintentar
        </button>
      )}
    </div>
  );
}
export function Load({ query, children }) {
  if (query.error) return <ErrorBox error={query.error} retry={query.retry} />;
  if (query.data === undefined)
    return (
      <p className="muted" role="status">
        Cargando datos…
      </p>
    );
  return children(query.data);
}
export function Table({ headers, rows, label }) {
  return (
    <div className="table-wrap" tabIndex="0" role="region" aria-label={label}>
      <table aria-label={label}>
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j}>{v ?? "—"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {!rows.length && (
        <p className="empty">Sin registros para esta selección.</p>
      )}
    </div>
  );
}
export function Heading({ title, text, action }) {
  return (
    <header className="heading">
      <div>
        <h1>{title}</h1>
        <p>{text}</p>
      </div>
      {action}
    </header>
  );
}
export function Warnings({ items }) {
  return items?.length ? (
    <ul className="warnings">
      {items.map((w) => (
        <li key={w}>{w}</li>
      ))}
    </ul>
  ) : null;
}
export function series(label, data, color = green, extra = {}) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color,
    borderWidth: 2,
    pointRadius: 0,
    pointHitRadius: 8,
    ...extra,
  };
}
export function Plot({
  title,
  datasets,
  x,
  y,
  type = "line",
  horizontal = false,
  labels,
}) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const chart = new Chart(ref.current, {
      type,
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        locale: "es-CO",
        indexAxis: horizontal ? "y" : "x",
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: x },
            grid: { color: "#e7ede9" },
            ticks: { maxTicksLimit: 8 },
          },
          y: {
            type: horizontal ? "category" : "linear",
            title: { display: Boolean(y), text: y },
            grid: { color: "#e7ede9" },
            ticks: { autoSkip: !horizontal },
          },
        },
        plugins: {
          legend: {
            display: datasets.length > 1,
            position: "bottom",
            labels: { usePointStyle: true, boxHeight: 7, padding: 15 },
          },
        },
      },
    });
    return () => chart.destroy();
  }, [type, datasets, x, y, horizontal, labels]);
  return (
    <div className={"plot" + (horizontal ? " tall" : "")}>
      <canvas ref={ref} role="img" aria-label={title}>
        {title}
      </canvas>
    </div>
  );
}
