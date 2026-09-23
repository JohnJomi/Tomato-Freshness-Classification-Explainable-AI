export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const num = (v: number, digits = 3) => v.toFixed(digits);

/** Compact display for sensor values that span 1e-5 .. 1e2. */
export function sensor(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 0.001) return v.toExponential(2);
  if (a < 1) return v.toFixed(4);
  if (a < 100) return v.toFixed(3);
  return v.toFixed(1);
}

/** Fixed model -> categorical slot mapping; color follows the entity. */
export const MODEL_COLORS: Record<string, string> = {
  "Random Forest": "var(--series-1)",
  SVM: "var(--series-2)",
  XGBoost: "var(--series-3)",
};

export const CLASS_ORDER = ["Pure Fresh", "Good", "Stale to Spoiled"] as const;

/** Feature groups for forms and tables (sensor -> its statistics). */
export const FEATURE_GROUPS: { title: string; features: string[] }[] = [
  ...["MQ2", "MQ3", "MQ9", "MQ135", "MQ136", "MQ138"].map((s) => ({
    title: s,
    features: [`${s}_mean`, `${s}_std`, `${s}_slope`],
  })),
  { title: "Temperature", features: ["Temperature_mean", "Temperature_std", "Temperature_slope"] },
  { title: "Humidity", features: ["Humidity_mean", "Humidity_std", "Humidity_slope"] },
  { title: "Ratios", features: ["MQ3_over_MQ2_end", "MQ136_over_MQ138_end"] },
];

export function cn(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}
