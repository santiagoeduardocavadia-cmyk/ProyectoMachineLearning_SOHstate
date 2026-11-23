import { useEffect, useState } from "react";

export const base = (
  import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1"
).replace(/\/+$/, "");

export function datasetPath(path, datasetId, filters = {}) {
  return (
    path + "?" + new URLSearchParams({ dataset_id: datasetId, ...filters })
  );
}

export async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(base + path, options);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error(
      "No se pudo conectar con la API en " +
        base +
        ". Comprueba que el backend esté en ejecución.",
    );
  }
  if (response.status === 204) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const issue = body?.error;
    const fields = Array.isArray(issue?.details)
      ? issue.details.map((d) =>
          [d.field, d.message].filter(Boolean).join(": "),
        )
      : [];
    if (
      issue?.code === "invalid_dataset" &&
      issue.details?.columns &&
      !Array.isArray(issue.details.columns) &&
      typeof issue.details.columns === "object"
    ) {
      for (const [name, count] of Object.entries(issue.details.columns)) {
        fields.push(name + ": " + count + " valores inválidos");
      }
    }
    const error = new Error(
      [
        issue?.message || "La API devolvió un error HTTP " + response.status,
        ...fields,
      ].join("\n"),
    );
    error.code = issue?.code;
    error.details = issue?.details;
    throw error;
  }
  if (body == null)
    throw new Error("La API devolvió una respuesta vacía o ilegible.");
  return body;
}

export function post(path, body) {
  if (body instanceof FormData) return request(path, { method: "POST", body });
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function useApi(path) {
  const [state, setState] = useState({});
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    const update = (result) => {
      if (!controller.signal.aborted) setState({ path, attempt, ...result });
    };
    request(path, { signal: controller.signal }).then(
      (data) => update({ data }),
      (error) => update({ error }),
    );
    return () => controller.abort();
  }, [path, attempt]);
  const current = state.path === path && state.attempt === attempt ? state : {};
  return { ...current, retry: () => setAttempt((n) => n + 1) };
}
