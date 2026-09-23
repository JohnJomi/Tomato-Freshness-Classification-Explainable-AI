"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Microscope, RotateCcw, Sigma, Sparkles } from "lucide-react";
import { PageContainer } from "@/components/layout/PageContainer";
import { Card } from "@/components/ui/Card";
import { Button, Select, inputClass } from "@/components/ui/Controls";
import { ErrorState, Loading } from "@/components/ui/States";
import { ProbabilityBars } from "@/components/charts/ProbabilityBars";
import { api } from "@/lib/api";
import { savePlaygroundFeatures } from "@/lib/playground";
import type { Features, Prediction } from "@/lib/types";
import { useApi } from "@/lib/useApi";
import { FEATURE_GROUPS, cn, pct, sensor } from "@/lib/utils";

type FormValues = Record<string, string>;

const toForm = (f: Features): FormValues => Object.fromEntries(Object.entries(f).map(([k, v]) => [k, String(v)]));

export default function PredictPage() {
  const router = useRouter();
  const dataset = useApi(api.dataset);
  const models = useApi(api.models);
  const samples = useApi(api.samples);

  // null = untouched: fall back to dataset medians / the CV-selected model.
  const [edited, setValues] = useState<FormValues | null>(null);
  const [source, setSource] = useState("median");
  const [pickedModel, setModel] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const medians: FormValues = dataset.data
    ? Object.fromEntries(Object.entries(dataset.data.feature_stats).map(([k, s]) => [k, String(s.median)]))
    : {};
  const values = edited ?? medians;
  const selectedName = models.data?.models.find((m) => m.selected_for_xai)?.name;
  const model = pickedModel ?? selectedName ?? "";

  async function loadSource(v: string) {
    setSource(v);
    setResult(null);
    setError(null);
    if (v === "median") {
      setValues(null);
    } else {
      try {
        setValues(toForm((await api.sample(Number(v))).features));
      } catch (e) {
        setError((e as Error).message);
      }
    }
  }

  const invalid = Object.entries(values).filter(([, v]) => v.trim() === "" || !Number.isFinite(Number(v))).map(([k]) => k);
  const features = (): Features => Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Number(v)]));

  async function predict(e: React.FormEvent) {
    e.preventDefault();
    if (invalid.length) return;
    setPending(true);
    setError(null);
    try {
      setResult(await api.predict(features(), model || undefined));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPending(false);
    }
  }

  function explain(kind: "lime" | "shap") {
    savePlaygroundFeatures(features());
    router.push(`/explain/${kind}?source=playground`);
  }

  const bootError = dataset.error ?? models.error;
  const actual = source !== "median" ? samples.data?.samples.find((s) => String(s.index) === source)?.actual_class : undefined;

  return (
    <PageContainer
      title="Prediction playground"
      description="Enter or adjust the 26 sensor features and ask a trained model for a freshness prediction."
    >
      {bootError ? (
        <ErrorState message={bootError} onRetry={() => { dataset.reload(); models.reload(); }} />
      ) : !dataset.data || !models.data ? (
        <Loading label="Loading feature ranges…" />
      ) : (
        <div className="grid gap-6 xl:grid-cols-[1fr_22rem]">
          <form onSubmit={predict} className="space-y-4" noValidate>
            <Card>
              <div className="flex flex-wrap items-end gap-3">
                <Select label="Start from" value={source} onChange={(e) => loadSource(e.target.value)} className="w-full sm:w-72">
                  <option value="median">Dataset medians</option>
                  <optgroup label="Held-out test samples (#index · actual class)">
                    {samples.data?.samples.map((s) => (
                      <option key={s.index} value={s.index}>#{s.index} · {s.actual_class}</option>
                    ))}
                  </optgroup>
                </Select>
                <Select label="Model" value={model} onChange={(e) => { setModel(e.target.value); setResult(null); }} className="w-full sm:w-56">
                  {models.data.models.map((m) => (
                    <option key={m.name} value={m.name}>{m.name}{m.name === selectedName ? " · selected" : ""}</option>
                  ))}
                </Select>
                <Button type="button" variant="secondary" onClick={() => loadSource(source)}>
                  <RotateCcw className="size-4" aria-hidden /> Reset
                </Button>
              </div>
            </Card>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {FEATURE_GROUPS.map((g) => (
                <fieldset key={g.title} className="rounded-card border border-line bg-surface p-4">
                  <legend className="px-1 text-sm font-semibold text-ink">{g.title}</legend>
                  <div className="space-y-2.5">
                    {g.features.map((f) => {
                      const s = dataset.data!.feature_stats[f];
                      const bad = invalid.includes(f);
                      return (
                        <label key={f} className="block text-xs">
                          <span className="mb-1 flex justify-between gap-2 text-ink-2">
                            <span className="truncate">{f.replace(`${g.title}_`, "") || f}</span>
                            <span className="shrink-0 text-muted tabular" title="Range in the dataset">
                              {sensor(s.min)} – {sensor(s.max)}
                            </span>
                          </span>
                          <input
                            inputMode="decimal"
                            name={f}
                            value={values[f] ?? ""}
                            onChange={(e) => { setValues({ ...values, [f]: e.target.value }); setResult(null); }}
                            aria-invalid={bad}
                            className={cn(inputClass, bad && "border-critical")}
                          />
                        </label>
                      );
                    })}
                  </div>
                </fieldset>
              ))}
            </div>

            <div className="sticky bottom-4 z-10 flex flex-wrap items-center gap-3 rounded-card border border-line bg-surface/95 p-3 backdrop-blur">
              <Button type="submit" disabled={pending || invalid.length > 0}>
                {pending ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Sparkles className="size-4" aria-hidden />}
                {pending ? "Analyzing tomato…" : "Predict freshness"}
              </Button>
              {invalid.length > 0 && (
                <p className="text-xs text-ink-2">Enter a number for: {invalid.join(", ")}</p>
              )}
            </div>
          </form>

          <aside className="space-y-4 xl:sticky xl:top-6 xl:self-start" aria-live="polite">
            {error && <ErrorState message={error} />}
            {result ? (
              <Card title="Prediction" subtitle={`Model: ${result.model}`}>
                <div className="rounded-control border border-line bg-surface-2 p-4 text-center">
                  <p className="text-xl font-semibold uppercase tracking-wide text-ink">{result.predicted_class}</p>
                  <p className="mt-1 text-sm text-ink-2 tabular">Confidence: {pct(result.confidence)}</p>
                  {actual && <p className="mt-1 text-xs text-muted">Actual class of this test sample: {actual}</p>}
                </div>
                <div className="mt-4">
                  <ProbabilityBars probabilities={result.probabilities} predicted={result.predicted_class} />
                </div>
                <div className="mt-4 grid grid-cols-2 gap-2">
                  <Button variant="secondary" onClick={() => explain("lime")}><Microscope className="size-4" aria-hidden /> Explain with LIME</Button>
                  <Button variant="secondary" onClick={() => explain("shap")}><Sigma className="size-4" aria-hidden /> Explain with SHAP</Button>
                </div>
                <p className="mt-2 text-[11px] text-muted">
                  Explanations always use the CV-selected model ({selectedName}).
                </p>
              </Card>
            ) : (
              !error && (
                <Card>
                  <p className="text-sm text-ink-2">Adjust the values and press <strong className="text-ink">Predict freshness</strong> to see the class probabilities.</p>
                </Card>
              )
            )}
          </aside>
        </div>
      )}
    </PageContainer>
  );
}
