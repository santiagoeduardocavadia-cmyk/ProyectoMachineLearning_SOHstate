import React, { useState } from "react";
import { base, post, request } from "./api";
import { ErrorBox, Heading, Load, Warnings, num } from "./shared";

export function DatasetSelector({ datasets, selected, onSelect }) {
  return (
    <label className="dataset-selector">
      Dataset
      <select
        value={selected}
        onChange={(event) => onSelect(event.target.value)}
      >
        {datasets.map((d) => (
          <option key={d.id} value={d.id}>
            {d.name}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Datasets({ catalog, onChange }) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState("");

  async function upload(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    setError(null);
    setNotice("");
    setPending(true);
    try {
      if (data.get("file").size > 20 * 1024 * 1024)
        throw new Error("El CSV debe pesar como máximo 20 MB.");
      data.set("name", data.get("name").trim());
      for (const field of ["nominal_capacity_ah", "eol_capacity_ah"]) {
        if (data.get(field) === "") data.delete(field);
      }
      const result = await post("/datasets", data);
      form.reset();
      setNotice("Dataset cargado: " + result.name + ".");
      onChange(result.id);
    } catch (err) {
      setError(err);
    } finally {
      setPending(false);
    }
  }

  async function remove(dataset) {
    if (
      dataset.source !== "upload" ||
      !window.confirm("¿Borrar el dataset «" + dataset.name + "»?")
    )
      return;
    setPending(true);
    setError(null);
    setNotice("");
    try {
      await request("/datasets/" + encodeURIComponent(dataset.id), {
        method: "DELETE",
      });
      setNotice("Dataset borrado: " + dataset.name + ".");
      onChange();
    } catch (err) {
      setError(err);
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <Heading
        title="Cargar datos"
        text="Importa ciclos de descarga para explorar tus baterías y entrenar modelos."
        action={
          <a className="button" href={base + "/datasets/template"} download>
            Descargar plantilla CSV
          </a>
        }
      />
      <form onSubmit={upload}>
        <fieldset className="panel" disabled={pending}>
          <legend>Nuevo dataset</legend>
          <p>
            Incluye battery_id, cycle, capacity_ah y al menos una característica
            adicional. Se requieren 10 ciclos válidos y dos baterías como
            mínimo.
          </p>
          <div className="fields">
            <label>
              Archivo CSV (máx. 20 MB)
              <input type="file" name="file" accept=".csv,text/csv" required />
            </label>
            <label>
              Nombre del dataset
              <input name="name" maxLength="80" pattern=".*\S.*" required />
            </label>
            <label>
              Capacidad nominal (Ah), opcional
              <input
                type="number"
                name="nominal_capacity_ah"
                min="0"
                step="any"
                placeholder="2,0"
              />
            </label>
            <label>
              Umbral de fin de vida (Ah), opcional
              <input
                type="number"
                name="eol_capacity_ah"
                min="0"
                step="any"
                placeholder="70 % de la capacidad nominal"
              />
            </label>
          </div>
          <div className="actions dataset-actions">
            <button className="primary" disabled={pending}>
              Cargar dataset
            </button>
          </div>
        </fieldset>
      </form>
      <ErrorBox error={error} />
      {notice && (
        <p className="dataset-notice" role="status">
          {notice}
        </p>
      )}
      <section className="panel">
        <h2>Datasets disponibles</h2>
        <Load query={catalog}>
          {({ datasets }) =>
            datasets.map((d) => (
              <article key={d.id} className="dataset-entry">
                <div className="panel-head">
                  <div>
                    <h3>{d.name}</h3>
                    <p className="muted">
                      {num(d.n_rows)} ciclos · {num(d.n_batteries)} baterías
                    </p>
                  </div>
                  {d.source === "upload" && (
                    <button
                      disabled={pending}
                      aria-label={"Borrar " + d.name}
                      onClick={() => remove(d)}
                    >
                      Borrar
                    </button>
                  )}
                </div>
                <Warnings items={d.warnings} />
              </article>
            ))
          }
        </Load>
      </section>
    </>
  );
}
