"use client";

import { useState } from "react";
import { ArrowDown, ArrowUp, ShieldCheck } from "lucide-react";
import { PageContainer } from "@/components/layout/PageContainer";
import { Card } from "@/components/ui/Card";
import { ErrorState, Loading } from "@/components/ui/States";
import { MetricDotPlot } from "@/components/charts/MetricDotPlot";
import { ConfusionMatrix } from "@/components/charts/ConfusionMatrix";
import { api } from "@/lib/api";
import type { ModelMetrics } from "@/lib/types";
import { useApi } from "@/lib/useApi";
import { MODEL_COLORS, pct } from "@/lib/utils";

type SortKey = keyof Pick<ModelMetrics, "name" | "accuracy" | "precision" | "recall" | "f1" | "f1_macro" | "cv_f1_mean" | "cv_accuracy_mean">;

const COLUMNS: { key: SortKey; label: string; group: "test" | "cv" | "" }[] = [
  { key: "name", label: "Model", group: "" },
  { key: "accuracy", label: "Accuracy", group: "test" },
  { key: "precision", label: "Precision", group: "test" },
  { key: "recall", label: "Recall", group: "test" },
  { key: "f1", label: "F1", group: "test" },
  { key: "f1_macro", label: "F1 macro", group: "test" },
  { key: "cv_f1_mean", label: "CV F1", group: "cv" },
  { key: "cv_accuracy_mean", label: "CV accuracy", group: "cv" },
];

function ModelTable({ models }: { models: ModelMetrics[] }) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "cv_f1_mean", desc: true });
  const rows = [...models].sort((a, b) => {
    const d = a[sort.key] < b[sort.key] ? -1 : a[sort.key] > b[sort.key] ? 1 : 0;
    return sort.desc ? -d : d;
  });
  const std = (m: ModelMetrics, k: SortKey) =>
    k === "cv_f1_mean" ? m.cv_f1_std : k === "cv_accuracy_mean" ? m.cv_accuracy_std : null;

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-sm tabular">
        <thead>
          <tr className="text-[11px] uppercase tracking-wide text-muted">
            <th />
            <th colSpan={5} className="pb-1 text-center font-medium">Held-out test set</th>
            <th colSpan={2} className="pb-1 text-center font-medium">5-fold CV (train) · mean ± std</th>
          </tr>
          <tr className="border-b border-line text-xs text-muted">
            {COLUMNS.map((c) => (
              <th key={c.key} className={`py-2 pr-3 font-medium ${c.key === "name" ? "text-left" : "text-right"}`} aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : "none"}>
                <button
                  onClick={() => setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : c.key !== "name" }))}
                  className="inline-flex items-center gap-1 hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-series-1"
                >
                  {c.label}
                  {sort.key === c.key && (sort.desc ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />)}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.name} className="border-b border-line/50 last:border-0">
              {COLUMNS.map((c) =>
                c.key === "name" ? (
                  <td key={c.key} className="py-2.5 pr-3 text-left text-ink">
                    <span className="inline-flex items-center gap-2">
                      <span className="size-2.5 rounded-full" style={{ background: MODEL_COLORS[m.name] }} aria-hidden />
                      {m.name}
                      {m.selected_for_xai && (
                        <span className="rounded-full border border-line bg-surface-2 px-1.5 py-0.5 text-[10px] text-ink-2">selected</span>
                      )}
                    </span>
                  </td>
                ) : (
                  <td key={c.key} className="py-2.5 pr-3 text-right text-ink-2">
                    {pct(m[c.key] as number, 2)}
                    {std(m, c.key) !== null && <span className="text-muted"> ± {pct(std(m, c.key)!, 2)}</span>}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelsPage() {
  const { data, error, reload } = useApi(api.models);

  return (
    <PageContainer
      title="Model comparison"
      description="Random Forest, SVM (RBF) and XGBoost. Cross-validation scores drove model selection; test-set scores are the final held-out evaluation."
    >
      {error ? (
        <ErrorState message={error} onRetry={reload} />
      ) : !data ? (
        <Loading label="Loading model results…" />
      ) : (
        <>
          <Card>
            <div className="flex items-start gap-3 text-sm">
              <ShieldCheck className="mt-0.5 size-5 shrink-0 text-good" aria-hidden />
              <p className="text-ink-2">
                <strong className="text-ink">{data.selection.xai_selected_model}</strong> was selected for the explainability
                analysis by mean 5-fold CV weighted F1 on the training set ({data.selection.selection_value.toFixed(4)},
                tie-breaker: CV accuracy). The test set was not used for selection
                {data.selection.test_set_used_for_selection ? "" : " — it was evaluated once, afterwards"}.
              </p>
            </div>
          </Card>

          <MetricDotPlot models={data.models} />

          <Card title="All metrics" subtitle="Click a column to sort">
            <ModelTable models={data.models} />
          </Card>

          <Card title="Confusion matrices" subtitle="Held-out test set, 111 samples">
            <div className="grid gap-6 md:grid-cols-3">
              {data.models.map((m) => (
                <ConfusionMatrix key={m.name} name={m.name} matrix={m.confusion_matrix} accent={MODEL_COLORS[m.name]} />
              ))}
            </div>
          </Card>
        </>
      )}
    </PageContainer>
  );
}
