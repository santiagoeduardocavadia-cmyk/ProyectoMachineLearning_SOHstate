import React, { useState } from "react";
import { datasetPath, post, useApi } from "./api";
import { ErrorBox, Heading, Load, targetName } from "./shared";

function TrainingForm({ catalog, features, dataset, legacy }) {
  const d = catalog.defaults;
  const [target, setTarget] = useState(d.target);
  const [modelKey, setModel] = useState(d.model_type);
  const [holdout, setHoldout] = useState(() => {
    const ids = dataset.batteries.map((b) => b.battery_id);
    const defaults = d.test_batteries.filter((id) => ids.includes(id));
    return defaults.length ? defaults : ids.slice(0, 1);
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const model = catalog.models.find((m) => m.key === modelKey);
  const eligible = dataset.batteries
    .filter((b) => target !== "rul" || b.reaches_eol)
    .map((b) => b.battery_id);
  const excluded = dataset.batteries
    .filter((b) => !b.reaches_eol)
    .map((b) => b.battery_id);
  async function train(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const test = holdout.filter((b) => eligible.includes(b));
    const inputs = data.getAll("features");
    setError(null);
    if (!test.length || test.length >= eligible.length || !inputs.length) {
      setError(
        new Error(
          "Selecciona alguna característica y reserva al menos una batería para cada conjunto, entrenamiento y prueba.",
        ),
      );
      return;
    }
    const hyperparameters = Object.fromEntries(
      model.hyperparameters.map((p) => {
        const value = data.get("param-" + p.name);
        return [p.name, value === "" ? null : Number(value)];
      }),
    );
    setBusy(true);
    const form = event.currentTarget;
    try {
      const result = await post("/experiments", {
        ...(legacy ? {} : { dataset_id: dataset.id }),
        name: data.get("name").trim(),
        description: data.get("description").trim() || null,
        target,
        model_type: modelKey,
        hyperparameters,
        features: inputs.length === features.length ? null : inputs,
        test_batteries: test,
        random_state: Number(data.get("seed")),
      });
      if (form.isConnected)
        location.hash = "experimento/" + encodeURIComponent(result.id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <form className="form-stack" onSubmit={train}>
      <fieldset className="panel" disabled={busy}>
        <legend>Configuración del experimento</legend>
        <div className="fields">
          <label>
            Nombre
            <input
              name="name"
              required
              maxLength="80"
              pattern=".*\S.*"
              placeholder="Ej. SOH con Ridge"
            />
          </label>
          <label>
            Objetivo
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              {catalog.targets.map((t) => (
                <option key={t.key} value={t.key}>
                  {targetName[t.key] || t.key}
                </option>
              ))}
            </select>
          </label>
          <label>
            Modelo
            <select value={modelKey} onChange={(e) => setModel(e.target.value)}>
              {catalog.models.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Semilla aleatoria
            <input
              name="seed"
              type="number"
              min="0"
              max="2147483647"
              required
              defaultValue={d.random_state}
            />
          </label>
        </div>
        <label>
          Descripción opcional
          <textarea name="description" maxLength="500" rows="2" />
        </label>
      </fieldset>
      <fieldset className="panel" disabled={busy}>
        <legend>Hiperparámetros</legend>
        <p>{model.description}</p>
        <div className="fields" key={model.key}>
          {model.hyperparameters.map((p) => (
            <label key={p.name}>
              {p.name}
              <input
                type="number"
                name={"param-" + p.name}
                step={p.type === "int" ? 1 : "any"}
                min={p.min ?? undefined}
                max={p.max ?? undefined}
                required={!p.nullable}
                defaultValue={p.default ?? ""}
                placeholder={p.nullable ? "Sin límite" : ""}
              />
              <small>{p.description}</small>
            </label>
          ))}
        </div>
      </fieldset>
      <fieldset className="panel" disabled={busy}>
        <legend>Baterías de prueba</legend>
        <p>
          El entrenamiento utiliza las baterías restantes. Todos los ciclos de
          una celda pertenecen al mismo conjunto.
        </p>
        {target === "rul" && excluded.length > 0 && (
          <p className="warnings">
            Baterías excluidas por falta de fin de vida observado:{" "}
            {excluded.join(", ")}.
          </p>
        )}
        <div className="checks">
          {eligible.map((b) => (
            <label key={b}>
              <input
                type="checkbox"
                checked={holdout.includes(b)}
                onChange={(e) =>
                  setHoldout(
                    e.target.checked
                      ? [...holdout, b]
                      : holdout.filter((v) => v !== b),
                  )
                }
              />
              {b}
            </label>
          ))}
        </div>
      </fieldset>
      <fieldset className="panel" disabled={busy}>
        <legend>Características de entrada</legend>
        <div className="checks">
          {features.map((f) => (
            <label key={f.name} title={f.description}>
              <input
                type="checkbox"
                name="features"
                value={f.name}
                defaultChecked
              />
              {f.name}
            </label>
          ))}
        </div>
      </fieldset>
      <ErrorBox error={error} />
      <button className="primary" disabled={busy}>
        {busy ? "Entrenando…" : "Entrenar experimento"}
      </button>
    </form>
  );
}
export function Train({ dataset, selector, legacy }) {
  const models = useApi("/models");
  const features = useApi(datasetPath("/dataset/features", dataset.id));
  return (
    <>
      <Heading
        title="Entrenamiento"
        text="Configura un modelo y evalúalo con baterías reservadas para prueba."
      />
      {selector}
      <Load query={models}>
        {(catalog) => (
          <Load query={features}>
            {(f) => (
              <TrainingForm
                catalog={catalog}
                dataset={dataset}
                legacy={legacy}
                features={f.features.filter((feature) => feature.min != null)}
              />
            )}
          </Load>
        )}
      </Load>
    </>
  );
}
