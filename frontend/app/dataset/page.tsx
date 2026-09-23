"use client";

import { CheckCircle2 } from "lucide-react";
import { PageContainer } from "@/components/layout/PageContainer";
import { Card, StatTile } from "@/components/ui/Card";
import { ErrorState, Loading } from "@/components/ui/States";
import { ClassDistributionChart } from "@/components/charts/ClassDistributionChart";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { FEATURE_GROUPS, sensor } from "@/lib/utils";

const RAW_ORDER = ["Pure Fresh", "Good", "Stale", "Spoiled"];

export default function DatasetPage() {
  const { data, error, reload } = useApi(api.dataset);

  return (
    <PageContainer title="Dataset" description="window_statistical_features.csv — statistics of gas-sensor and environment readings over observation windows.">
      {error ? (
        <ErrorState message={error} onRetry={reload} />
      ) : !data ? (
        <Loading label="Loading dataset summary…" />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatTile label="Samples" value={data.samples} />
            <StatTile label="Features" value={data.features} hint="All numeric" />
            <StatTile label="Classes" value={data.classes.length} hint="After target mapping" />
            <StatTile
              label="Missing values"
              value={
                <span className="inline-flex items-center gap-2">
                  {data.missing_values}
                  <CheckCircle2 className="size-5 text-good" aria-hidden />
                </span>
              }
              hint="Median imputation still runs inside every pipeline"
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <ClassDistributionChart distribution={data.class_distribution} />
            <Card title="Target mapping" subtitle="Four original labels → three required classes">
              <table className="w-full text-sm tabular">
                <thead>
                  <tr className="border-b border-line text-xs text-muted">
                    <th className="py-2 text-left font-medium">Original label</th>
                    <th className="py-2 text-right font-medium">Samples</th>
                    <th className="py-2 pl-4 text-left font-medium">Mapped to</th>
                  </tr>
                </thead>
                <tbody>
                  {RAW_ORDER.map((l) => (
                    <tr key={l} className="border-b border-line/50 last:border-0">
                      <td className="py-2 text-ink">{l}</td>
                      <td className="py-2 text-right text-ink-2">{data.raw_class_distribution[l]}</td>
                      <td className="py-2 pl-4 text-ink-2">{l === "Stale" || l === "Spoiled" ? "Stale to Spoiled" : l}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-4 text-xs text-muted">
                Stratified 80/20 split: {data.train_size} training / {data.test_size} held-out test samples.
              </p>
            </Card>
          </div>

          <Card title="Features" subtitle="Descriptive statistics over all 555 samples">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm tabular">
                <thead>
                  <tr className="border-b border-line text-xs text-muted">
                    {["Feature", "Mean", "Std", "Min", "Median", "Max"].map((h, i) => (
                      <th key={h} className={`py-2 pr-4 font-medium ${i ? "text-right" : "text-left"}`}>{h}</th>
                    ))}
                  </tr>
                </thead>
                {FEATURE_GROUPS.map((g) => (
                  <tbody key={g.title}>
                    <tr>
                      <th colSpan={6} className="pb-1 pt-4 text-left text-xs font-semibold uppercase tracking-wide text-muted">{g.title}</th>
                    </tr>
                    {g.features.map((f) => {
                      const s = data.feature_stats[f];
                      return (
                        <tr key={f} className="border-b border-line/50">
                          <td className="py-1.5 pr-4 text-ink">{f}</td>
                          {[s.mean, s.std, s.min, s.median, s.max].map((v, i) => (
                            <td key={i} className="py-1.5 pr-4 text-right text-ink-2">{sensor(v)}</td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                ))}
              </table>
            </div>
          </Card>
        </>
      )}
    </PageContainer>
  );
}
