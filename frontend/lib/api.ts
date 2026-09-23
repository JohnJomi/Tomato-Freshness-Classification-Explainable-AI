import type {
  Dataset,
  ExplainInput,
  FeatureImportance,
  Features,
  Health,
  LimeExplanation,
  ModelsResponse,
  Prediction,
  Sample,
  SampleRef,
  ShapGlobal,
  ShapLocal,
} from "./types";

// Inlined at build time; set NEXT_PUBLIC_API_URL if the backend isn't on :8000.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
  }
}

export const OFFLINE_MESSAGE =
  "Unable to connect to the ML service. Please make sure the backend is running.";

function detailMessage(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  // FastAPI validation errors: [{ msg, loc }, ...]
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => String((d as { msg?: string }).msg ?? d).replace(/^Value error, /, "")).join("; ");
  }
  return `Request failed (${status}).`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(OFFLINE_MESSAGE);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(detailMessage(body, res.status), res.status);
  return body as T;
}

const post = <T,>(path: string, data: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(data) });

export const api = {
  health: () => request<Health>("/health"),
  dataset: () => request<Dataset>("/dataset"),
  samples: () => request<{ samples: SampleRef[] }>("/dataset/samples"),
  sample: (index: number) => request<Sample>(`/dataset/samples/${index}`),
  models: () => request<ModelsResponse>("/models"),
  featureImportance: () => request<FeatureImportance>("/feature-importance"),
  predict: (features: Features, model?: string) => post<Prediction>("/predict", { features, model }),
  lime: (input: ExplainInput) => post<LimeExplanation>("/explain/lime", input),
  shap: (input: ExplainInput) => post<ShapLocal>("/explain/shap", input),
  shapGlobal: (className: string) =>
    request<ShapGlobal>(`/explain/shap/global?class_name=${encodeURIComponent(className)}`),
};
