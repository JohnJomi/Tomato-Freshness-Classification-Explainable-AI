"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { PageContainer } from "@/components/layout/PageContainer";
import { Card, StatTile } from "@/components/ui/Card";
import { ErrorState, Skeleton } from "@/components/ui/States";
import { ClassDistributionChart } from "@/components/charts/ClassDistributionChart";
import { MetricDotPlot } from "@/components/charts/MetricDotPlot";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { pct } from "@/lib/utils";

export default function Dashboard() {
  const dataset = useApi(api.dataset);
  const models = useApi(api.models);
  const error = dataset.error ?? models.error;

  const selected = models.data?.models.find((m) => m.selected_for_xai);

  return (
    <PageContainer
      title="Tomato Freshness AI"
      description="Classifying tomato freshness from gas-sensor statistics with three supervised models, explained with feature importance, LIME and SHAP."
    >
      {error ? (
        <ErrorState message={error} onRetry={() => { dataset.reload(); models.reload(); }} />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {dataset.data && models.data ? (
              <>
                <StatTile label="Samples" value={dataset.data.samples} hint={`${dataset.data.train_size} train · ${dataset.data.test_size} test`} />
                <StatTile label="Features" value={dataset.data.features} hint="Gas sensors, environment, ratios" />
                <StatTile label="Classes" value={dataset.data.classes.length} hint={dataset.data.classes.join(" · ")} />
                <StatTile label="Models" value={models.data.models.length} hint={models.data.models.map((m) => m.name).join(" · ")} />
              </>
            ) : (
              [0, 1, 2, 3].map((i) => <Skeleton key={i} />)
            )}
          </div>

          {selected && models.data ? (
            <Card
              title={<>Selected model: {selected.name}</>}
              subtitle={`Chosen by mean 5-fold CV weighted F1 on the training set (${selected.cv_f1_mean.toFixed(4)}); test-set scores below are the final, held-out evaluation.`}
              action={
                <Link href="/models" className="inline-flex items-center gap-1 text-xs text-ink-2 hover:text-ink">
                  All models <ArrowRight className="size-3.5" aria-hidden />
                </Link>
              }
            >
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                {(
                  [
                    ["Accuracy", selected.accuracy],
                    ["Precision", selected.precision],
                    ["Recall", selected.recall],
                    ["F1-score", selected.f1],
                  ] as const
                ).map(([label, v]) => (
                  <div key={label} className="rounded-lg border border-line bg-surface-2 p-3">
                    <p className="text-xs text-muted">{label}</p>
                    <p className="mt-0.5 text-xl font-semibold text-ink tabular">{pct(v)}</p>
                  </div>
                ))}
              </div>
            </Card>
          ) : (
            <Skeleton className="h-36" />
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            {models.data ? <MetricDotPlot models={models.data.models} /> : <Skeleton className="h-72" />}
            {dataset.data ? <ClassDistributionChart distribution={dataset.data.class_distribution} /> : <Skeleton className="h-72" />}
          </div>

          <Card title="Methodology">
            <ol className="grid gap-3 text-sm text-ink-2 md:grid-cols-2 lg:grid-cols-4">
              {[
                ["Data", "555 sensor windows; Stale + Spoiled merged into Stale to Spoiled; stratified 80/20 split."],
                ["Validation", "Stratified 5-fold CV on the training set; preprocessing fitted inside each fold."],
                ["Selection", "Model chosen by CV weighted F1 only — the test set stays locked until final evaluation."],
                ["Explainability", "Feature importance and SHAP (global), LIME and SHAP (local) on the selected model."],
              ].map(([t, d], i) => (
                <li key={t} className="rounded-lg border border-line bg-surface-2 p-3">
                  <p className="text-xs font-medium text-muted">{i + 1}. {t}</p>
                  <p className="mt-1">{d}</p>
                </li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </PageContainer>
  );
}
