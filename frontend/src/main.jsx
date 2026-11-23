import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Data } from "./Data";
import { Datasets, DatasetSelector } from "./Datasets";
import { Train } from "./Train";
import { Detail, History } from "./Experiments";
import { Heading, Load } from "./shared";
import { datasetPath, useApi } from "./api";
import "./style.css";

const pages = [
  ["datos", "Datos"],
  ["cargar-datos", "Cargar datos"],
  ["entrenar", "Entrenamiento"],
  ["experimentos", "Experimentos"],
  ["metodologia", "Metodología"],
];
function Method() {
  return (
    <>
      <Heading
        title="Metodología"
        text="Cómo se evalúan las predicciones de salud, capacidad y vida útil."
      />
      <div className="method">
        {[
          [
            "Separación por batería",
            "Cada batería pertenece por completo al entrenamiento o a la prueba. Así se mide si el modelo generaliza a una celda nueva y se evita repartir ciclos consecutivos casi idénticos entre conjuntos.",
          ],
          [
            "Comparación con la media",
            "La línea base predice siempre la media del entrenamiento. Su MAE y RMSE indican cuánto mejora el modelo sobre esa predicción sencilla en las mismas baterías de prueba.",
          ],
          [
            "Validación cruzada",
            "Dentro del entrenamiento se reserva una batería por turno. La media y la desviación de las métricas resumen cuánto cambia el resultado según la celda reservada.",
          ],
          [
            "Selección de características",
            "La duración total de la descarga permite reconstruir la capacidad a corriente constante. Se excluye para evitar que la respuesta entre en el modelo.",
          ],
          [
            "Fin de vida observado",
            "El umbral del conjunto NASA es 1,4 Ah. B0007 terminó el ensayo por encima del umbral, así que se desconoce cuántos ciclos le quedaban. Se excluye al entrenar y evaluar RUL.",
          ],
        ].map(([title, text]) => (
          <section className="panel" key={title}>
            <h2>{title}</h2>
            <p>{text}</p>
          </section>
        ))}
      </div>
    </>
  );
}
function App() {
  const [hash, setHash] = useState(() => location.hash.slice(1) || "datos");
  useEffect(() => {
    const sync = () => {
      setHash(location.hash.slice(1) || "datos");
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  const [page, id] = hash.split("/");
  const active = page === "experimento" ? "experimentos" : page;
  const health = useApi("/health");
  const version = health.data?.api_version;
  const incompatible = version && version.split(".")[0] !== "1";
  const legacy = version?.startsWith("1.0.");
  const supportsDomain = !incompatible && Number(version?.split(".")[1]) >= 2;
  const [selectedId, setSelectedId] = useState("nasa");
  const catalog = useApi(
    version && !incompatible
      ? legacy
        ? datasetPath("/dataset/summary", "nasa")
        : "/datasets"
      : null,
  );
  const online = health.data?.status === "ok";
  return (
    <>
      <a
        className="skip"
        href="#content"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("content").focus();
        }}
      >
        Ir al contenido
      </a>
      <header className="mast">
        <div className="mast-inner">
          <a className="brand" href="#datos">
            <i className="mark" aria-hidden="true">
              <b />
              <b />
              <b />
            </i>
            Laboratorio de baterías
          </a>
          <span
            className={
              "connection " + (online && !incompatible ? "online" : "")
            }
          >
            {health.error
              ? "API sin conexión"
              : incompatible
                ? "API incompatible"
                : online
                  ? "API conectada"
                  : "Conectando…"}
            {health.error && <button onClick={health.retry}>Reintentar</button>}
          </span>
        </div>
      </header>
      <nav aria-label="Principal">
        <div className="nav-inner">
          {pages.map(([key, label]) => (
            <a
              key={key}
              href={"#" + key}
              aria-current={active === key ? "page" : undefined}
            >
              {label}
            </a>
          ))}
        </div>
      </nav>
      <main id="content" tabIndex="-1">
        {incompatible ? (
          <div className="error" role="alert">
            Esta interfaz requiere la API v1. El servidor indica {version}.
          </div>
        ) : page === "cargar-datos" ? (
          legacy ? (
            <p className="panel">
              La carga de datasets requiere la API v1.1.0 o posterior.
            </p>
          ) : (
            <Load query={health}>
              {() => (
                <Datasets
                  catalog={catalog}
                  onChange={(id) => {
                    if (id) setSelectedId(id);
                    catalog.retry();
                  }}
                />
              )}
            </Load>
          )
        ) : page === "datos" || page === "entrenar" ? (
          <Load query={health.error ? health : catalog}>
            {(data) => {
              const datasets = legacy
                ? [{ ...data, id: "nasa", name: "NASA" }]
                : data.datasets;
              const dataset =
                datasets.find((d) => d.id === selectedId) ||
                datasets.find((d) => d.id === "nasa") ||
                datasets[0];
              if (!dataset)
                return (
                  <p className="panel">Todavía no hay datasets disponibles.</p>
                );
              const selector = (
                <DatasetSelector
                  datasets={datasets}
                  selected={dataset.id}
                  onSelect={setSelectedId}
                />
              );
              return page === "datos" ? (
                <Data key={dataset.id} dataset={dataset} selector={selector} />
              ) : (
                <Train
                  key={dataset.id}
                  dataset={dataset}
                  selector={selector}
                  legacy={legacy}
                  supportsDomain={supportsDomain}
                  multitemp={datasets.find((d) => d.id === "nasa_multitemp")}
                  onSelectDataset={setSelectedId}
                />
              );
            }}
          </Load>
        ) : page === "experimentos" ? (
          <History />
        ) : page === "experimento" && id ? (
          <Detail key={id} id={id} supportsDomain={supportsDomain} />
        ) : page === "metodologia" ? (
          <Method />
        ) : (
          <section className="panel">
            <h1>Vista desconocida</h1>
            <a href="#datos">Volver a los datos</a>
          </section>
        )}
      </main>
      <footer>
        Universidad del Magdalena{" "}
        <span>
          Ingeniería Electrónica · Fundamentos de IA y Machine Learning
        </span>
      </footer>
    </>
  );
}
createRoot(document.getElementById("root")).render(<App />);
