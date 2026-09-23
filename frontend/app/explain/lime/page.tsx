"use client";

import { Info } from "lucide-react";
import { PageContainer } from "@/components/layout/PageContainer";
import { Card, ExplanationMeta } from "@/components/ui/Card";
import { EmptyState, ErrorState, Loading } from "@/components/ui/States";
import { ContributionChart } from "@/components/charts/ContributionChart";
import { ChartFrame, DataTable } from "@/components/charts/ChartFrame";
import { PredictionSummary } from "@/components/xai/PredictionSummary";
import { useExplainInput } from "@/components/xai/SampleSource";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { sensor } from "@/lib/utils";

export default function LimePage() {
  const { input, picker } = useExplainInput();
  const key = JSON.stringify(input);
  const lime = useApi(() => (input ? api.lime(input) : Promise.resolve(null)), [key]);
  const d = lime.data;

  return (
    <PageContainer
      title="LIME explanation"
      description="Why did the model classify this particular tomato reading as it did?"
      actions={picker}
    >
      {!input ? (
        <EmptyState>No Playground values yet — pick a test sample, or run a prediction first.</EmptyState>
      ) : lime.error ? (
        <ErrorState message={lime.error} onRetry={lime.reload} />
      ) : !d ? (
        <Loading label="Generating explanation…" />
      ) : (
        <div className={lime.loading ? "opacity-60 transition-opacity" : "transition-opacity"}>
          <div className="space-y-6">
            <ExplanationMeta
              model={d.model}
              scope="local"
              what={d.sample_index !== null ? `Explaining held-out test sample #${d.sample_index}` : "Explaining your Playground values"}
            />
            <PredictionSummary actual={d.actual_class} predicted={d.prediction} confidence={d.confidence} />

            <ChartFrame
              title={`Local feature contributions toward “${d.explained_class}”`}
              subtitle="Top 10 LIME weights for this one sample"
              table={
                <DataTable
                  head={["Condition", "Feature value", "Weight"]}
                  rows={d.contributions.map((c) => [c.rule, sensor(c.value), c.contribution.toFixed(4)])}
                />
              }
            >
              <ContributionChart
                explainedClass={d.explained_class}
                items={d.contributions.map((c) => ({ label: c.rule, value: c.contribution, detail: `${c.feature} = ${sensor(c.value)}` }))}
              />
            </ChartFrame>

            <Card>
              <div className="flex items-start gap-3 text-sm text-ink-2">
                <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
                <p>
                  LIME explains this individual prediction by approximating the model locally around the selected sample.
                  Each weight comes from a simple surrogate model fitted to perturbed copies of this sample, so it describes
                  this prediction only — not how the model behaves globally.
                </p>
              </div>
            </Card>
          </div>
        </div>
      )}
    </PageContainer>
  );
}
