"use client";

import { useState } from "react";
import { Info } from "lucide-react";
import { PageContainer } from "@/components/layout/PageContainer";
import { Card, ExplanationMeta } from "@/components/ui/Card";
import { Segmented } from "@/components/ui/Controls";
import { EmptyState, ErrorState, Loading } from "@/components/ui/States";
import { Beeswarm } from "@/components/charts/Beeswarm";
import { ContributionChart } from "@/components/charts/ContributionChart";
import { ChartFrame, DataTable } from "@/components/charts/ChartFrame";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
import { PredictionSummary } from "@/components/xai/PredictionSummary";
import { useExplainInput } from "@/components/xai/SampleSource";
import { api } from "@/lib/api";
import type { ClassName, ShapLocal } from "@/lib/types";
import { useApi } from "@/lib/useApi";
import { CLASS_ORDER, sensor } from "@/lib/utils";

const TOP = 10;

function LocalWaterfall({ d }: { d: ShapLocal }) {
  const cls = d.prediction;
  const vals = d.shap_values[cls];
  const order = vals.map((_, i) => i).sort((a, b) => Math.abs(vals[b]) - Math.abs(vals[a]));
  const topIdx = order.slice(0, TOP);
  const rest = order.slice(TOP).reduce((s, i) => s + vals[i], 0);
  const items = [
    ...topIdx.map((i) => ({
      label: `${d.feature_names[i]} = ${sensor(d.feature_values[i])}`,
      value: vals[i],
      detail: d.feature_names[i],
    })),
    { label: `${order.length - TOP} other features`, value: rest, detail: "combined" },
  ];
  const base = d.base_values[cls];
  const out = d.output_values[cls];

  return (
    <ChartFrame
      title={`Local SHAP contributions toward “${cls}”`}
      subtitle="Largest 10 contributions for this sample; the rest are summed"
      table={
        <DataTable
          head={["Feature", "Value", "SHAP value"]}
          rows={order.map((i) => [d.feature_names[i], sensor(d.feature_values[i]), vals[i].toFixed(4)])}
        />
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-2 text-sm tabular" aria-label="Base value plus contributions equals prediction">
          <span className="rounded-md border border-line bg-surface-2 px-3 py-1.5">
            <span className="text-xs text-muted">Base prediction </span>
            <strong className="text-ink">{base.toFixed(3)}</strong>
          </span>
          <span className="text-muted">+</span>
          <span className="rounded-md border border-line bg-surface-2 px-3 py-1.5">
            <span className="text-xs text-muted">Feature contributions </span>
            <strong className="text-ink">{(out - base >= 0 ? "+" : "") + (out - base).toFixed(3)}</strong>
          </span>
          <span className="text-muted">=</span>
          <span className="rounded-md border border-line bg-surface-2 px-3 py-1.5">
            <span className="text-xs text-muted">Final prediction </span>
            <strong className="text-ink">{out.toFixed(3)}</strong>
          </span>
        </div>
        <ContributionChart explainedClass={cls} items={items} />
      </div>
    </ChartFrame>
  );
}

export default function ShapPage() {
  const [cls, setCls] = useState<ClassName>("Stale to Spoiled");
  const global = useApi(() => api.shapGlobal(cls), [cls]);
  const { input, picker } = useExplainInput();
  const key = JSON.stringify(input);
  const local = useApi(() => (input ? api.shap(input) : Promise.resolve(null)), [key]);

  return (
    <PageContainer
      title="SHAP explanation"
      description="SHAP values describe how individual features move a prediction away from the model's baseline for the selected class."
    >
      {/* Global */}
      <section className="space-y-4" aria-labelledby="shap-global">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="space-y-2">
            <h2 id="shap-global" className="text-lg font-semibold text-ink">Global view</h2>
            {global.data && <ExplanationMeta model={global.data.model} scope="global" what={`Across all ${global.data.n_samples} held-out test samples`} />}
          </div>
          <Segmented label="Explained class" options={CLASS_ORDER} value={cls} onChange={setCls} />
        </div>

        {global.error ? (
          <ErrorState message={global.error} onRetry={global.reload} />
        ) : !global.data ? (
          <Loading label="Computing SHAP summary…" />
        ) : (
          <div className={`grid gap-6 lg:grid-cols-2 ${global.loading ? "opacity-60" : ""} transition-opacity`}>
            <FeatureImportanceChart
              title="Global SHAP importance"
              subtitle={`Mean |SHAP value| for “${cls}”`}
              items={global.data.importance
                .map((f) => ({ name: f.feature, value: f.mean_abs_shap_class }))
                .sort((a, b) => b.value - a.value)
                .slice(0, 12)}
              valueLabel="Mean |SHAP|"
            />
            <ChartFrame
              title="SHAP summary"
              subtitle={`Each dot is one test sample; position = impact on “${cls}” probability`}
            >
              <Beeswarm summary={global.data.summary} />
            </ChartFrame>
          </div>
        )}
      </section>

      {/* Local */}
      <section className="space-y-4" aria-labelledby="shap-local">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="space-y-2">
            <h2 id="shap-local" className="text-lg font-semibold text-ink">Individual explanation</h2>
            {local.data && (
              <ExplanationMeta
                model={local.data.model}
                scope="local"
                what={local.data.sample_index !== null ? `Explaining held-out test sample #${local.data.sample_index}` : "Explaining your Playground values"}
              />
            )}
          </div>
          {picker}
        </div>

        {!input ? (
          <EmptyState>No Playground values yet — pick a test sample, or run a prediction first.</EmptyState>
        ) : local.error ? (
          <ErrorState message={local.error} onRetry={local.reload} />
        ) : !local.data ? (
          <Loading label="Generating explanation…" />
        ) : (
          <div className={`space-y-6 ${local.loading ? "opacity-60" : ""} transition-opacity`}>
            <PredictionSummary actual={local.data.actual_class} predicted={local.data.prediction} confidence={local.data.confidence} />
            <LocalWaterfall d={local.data} />
          </div>
        )}
      </section>

      <Card>
        <div className="flex items-start gap-3 text-sm text-ink-2">
          <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
          <p>
            For this tree model SHAP values are in probability units: the baseline is the average predicted probability of the
            class, and the contributions sum exactly to the model&apos;s output. They describe how the model used each feature —
            not that the feature causes spoilage.
          </p>
        </div>
      </Card>
    </PageContainer>
  );
}
