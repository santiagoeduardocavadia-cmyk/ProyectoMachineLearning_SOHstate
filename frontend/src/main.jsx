import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Data } from "./Data";
import { Datasets, DatasetSelector } from "./Datasets";
import { Train } from "./Train";
import { Detail, History } from "./Experiments";
import { Predict } from "./Predict";
import { Heading, Load, Table } from "./shared";
import { datasetPath, useApi } from "./api";
import "./style.css";

const pages = [
  ["datos", "Datos"],
  ["cargar-datos", "Cargar datos"],
  ["entrenar", "Entrenamiento"],
  ["experimentos", "Experimentos"],
  ["predecir", "Predecir"],
  ["metodologia", "Metodología"],
];
// Resultados de python -m scripts.compare_models (backend/docs/SELECCION_MODELO.md).
const MODEL_CV = [
  ["Random Forest", "2,80 ± 1,13", "—", "Elegido", "2,01"],
  ["SVR (kernel RBF)", "3,07 ± 2,13", "1,00", "Empatado", "1,48"],
  ["Gradient Boosting", "3,33 ± 1,60", "0,28", "Peor", "3,22"],
  ["Regresión lineal (Ridge)", "3,95 ± 1,92", "0,13", "Peor", "2,10"],
  ["K vecinos más cercanos", "4,31 ± 2,49", "0,002", "Peor (significativo)", "3,03"],
];

function ModelComparison() {
  return (
    <section className="panel">
      <h2>Comparación de modelos: SOH multitemperatura</h2>
      <p>
        Error medio (MAE) en puntos de SOH sobre celdas que el modelo no vio,
        con validación cruzada de 10 pliegues por celda física (29 celdas). La
        prueba de Wilcoxon compara cada modelo con el mejor pliegue a pliegue:
        con p ≥ 0,05 la diferencia puede deberse al azar. La columna de prueba
        (B0006, B0031 y B0047) solo confirma; no se usó para elegir. Predecir
        siempre la media daría 12,9 puntos.
      </p>
      <Table
        label="Comparación de modelos por validación cruzada"
        headers={[
          "Modelo",
          "MAE validación cruzada",
          "p (vs. el mejor)",
          "Resultado",
          "MAE prueba",
        ]}
        rows={MODEL_CV}
      />
      <p className="muted">
        Random Forest y SVR empatan: cualquiera de los dos es adecuado. Se usa
        Random Forest porque tiene el menor error medio y es más estable entre
        celdas (desviación 1,13 frente a 2,13). En el dataset NASA original (4
        celdas a 24 °C) gana Ridge, porque los árboles no pueden predecir un SOH
        más bajo que el mínimo visto.
      </p>
    </section>
  );
}

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
          [
            "Otras temperaturas y corrientes",
            "Un modelo entrenado solo a 24 °C falla con celdas a 4 °C o 43 °C. Por eso hay un dataset con 32 celdas a 4, 24 y 43 °C, la temperatura y la corriente entran como características, y cada predicción indica si los datos están dentro, en el borde o fuera de lo que el modelo conoce.",
          ],
          [
            "Data augmentation",
            "Las copias con ruido, interpolación y mezcla de celdas mejoran poco dentro de un mismo dataset y no ayudan a generalizar a otras condiciones: solo recombinan lo que ya existe. Lo que reduce el error es entrenar con celdas reales de esas condiciones.",
          ],
          [
            "¿Qué modelo se usa y por qué?",
            "Se compararon 5 modelos (Random Forest, Gradient Boosting, Ridge, SVR y K vecinos) con validación cruzada por celda física: 10 pliegues que dejan fuera celdas completas, sin tocar las celdas de prueba. Se elige el de menor error en celdas no vistas. En Predecir, si varios experimentos se parecen igual a tus datos, gana el de menor error de validación cruzada.",
          ],
          [
            "Vida restante en días",
            "Ciclos restantes = (SOH actual − 70 %) ÷ pérdida de SOH por ciclo, tomando el ritmo mayor entre la tendencia del CSV y el de las celdas parecidas. Los ciclos se convierten en días con perfiles de uso de celdas Li-ion recargables (laptop, dron, linterna…). Es una estimación orientativa.",
          ],
        ].map(([title, text]) => (
          <section className="panel" key={title}>
            <h2>{title}</h2>
            <p>{text}</p>
          </section>
        ))}
      </div>
      <ModelComparison />
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
    version &&
      !incompatible &&
      ["datos", "entrenar", "cargar-datos"].includes(page)
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
        ) : page === "predecir" ? (
          <Predict />
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
