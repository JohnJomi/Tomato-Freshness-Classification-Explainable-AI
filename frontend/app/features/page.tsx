"use client";

import { useState } from "react";
import { Info } from "lucide-react";
import { PageContainer } from "@/components/layout/PageContainer";
import { Card, ExplanationMeta } from "@/components/ui/Card";
import { Segmented } from "@/components/ui/Controls";
import { ErrorState, Loading } from "@/components/ui/States";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";

const METHODS = ["Model importance", "Mean |SHAP|"] as const;
const TOPS = ["Top 10", "Top 15", "All 26"] as const;

export default function FeaturesPage() {
  const fi = useApi(api.featureImportance);
  const shap = useApi(() => api.shapGlobal("Stale to Spoiled"));
  const [method, setMethod] = useState<(typeof METHODS)[number]>("Model importance");
  const [top, setTop] = useState<(typeof TOPS)[number]>("Top 15");
  const n = top === "Top 10" ? 10 : top === "Top 15" ? 15 : 26;

  const error = fi.error ?? shap.error;
  const items =
    method === "Model importance"
      ? fi.data?.features.map((f) => ({ name: f.name, value: f.importance }))
      : shap.data?.importance.map((f) => ({ name: f.feature, value: f.mean_abs_shap }));

  return (
    <PageContainer
      title="Feature importance"
      description="Which sensor statistics the CV-selected model relies on most across all predictions."
    >
      {error ? (
        <ErrorState message={error} onRetry={() => { fi.reload(); shap.reload(); }} />
      ) : !fi.data || !items ? (
        <Loading label="Loading feature importance…" />
      ) : (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <ExplanationMeta
              model={fi.data.model}
              scope="global"
              what={method === "Model importance" ? `Method: ${fi.data.method}` : "Method: mean |SHAP value| over the test set, all classes"}
            />
            <div className="flex flex-wrap gap-2">
              <Segmented label="Importance method" options={METHODS} value={method} onChange={setMethod} />
              <Segmented label="Number of features" options={TOPS} value={top} onChange={setTop} />
            </div>
          </div>

          <FeatureImportanceChart
            title="Top features influencing tomato freshness classification"
            subtitle={method === "Model importance" ? "Impurity-based importance from the fitted model" : "Average magnitude of each feature's SHAP contribution"}
            items={items.slice(0, n)}
            valueLabel={method === "Model importance" ? "Importance" : "Mean |SHAP|"}
          />

          <Card>
            <div className="flex items-start gap-3 text-sm text-ink-2">
              <Info className="mt-0.5 size-4 shrink-0 text-ink-2" aria-hidden />
              <p>
                Feature importance indicates how strongly a feature contributes to model predictions. It should not be
                interpreted as proof of causal influence — a high-ranked gas reading is associated with a class, it does
                not show that the gas causes spoilage. Only the CV-selected model is explained here.
              </p>
            </div>
          </Card>
        </>
      )}
    </PageContainer>
  );
}
