"use client";

import { useState } from "react";
import type { ModelMetrics } from "@/lib/types";
import { MODEL_COLORS, pct } from "@/lib/utils";
import { ChartFrame, DataTable, Legend } from "./ChartFrame";

const METRICS: { key: keyof ModelMetrics; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1-score" },
];

/** Dot plot, not bars: every score sits in 0.97–1.00, and a dot's position
 *  (unlike a bar's length) stays honest on a zoomed axis. */
export function MetricDotPlot({ models }: { models: ModelMetrics[] }) {
  const [hover, setHover] = useState<string | null>(null);
  const values = models.flatMap((m) => METRICS.map((k) => m[k.key] as number));
  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.005) * 100) / 100);
  const hi = 1;
  const x = (v: number) => ((v - lo) / (hi - lo)) * 100;
  const step = hi - lo > 0.04 ? 0.01 : 0.005;
  const ticks = Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step);

  return (
    <ChartFrame
      title="Model comparison (held-out test set)"
      subtitle={`Weighted metrics · axis zoomed to ${pct(lo, 0)}–100%`}
      table={
        <DataTable
          head={["Model", "Accuracy", "Precision", "Recall", "F1-score"]}
          rows={models.map((m) => [m.name, ...METRICS.map((k) => pct(m[k.key] as number, 2))])}
        />
      }
    >
      <div className="space-y-4">
        <Legend mark="dot" items={models.map((m) => ({ label: m.name, color: MODEL_COLORS[m.name] }))} />
        <div className="space-y-1">
          {METRICS.map((metric) => (
            <div key={metric.key} className="grid grid-cols-[5.5rem_1fr] items-center gap-3">
              <span className="text-xs text-ink-2">{metric.label}</span>
              <div className="relative h-9">
                <span className="absolute inset-x-0 top-1/2 h-px bg-[var(--grid)]" aria-hidden />
                {models.map((m) => {
                  const v = m[metric.key] as number;
                  const id = `${metric.key}-${m.name}`;
                  return (
                    <button
                      key={m.name}
                      onPointerEnter={() => setHover(id)}
                      onPointerLeave={() => setHover(null)}
                      onFocus={() => setHover(id)}
                      onBlur={() => setHover(null)}
                      className="absolute top-1/2 grid size-6 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-series-1"
                      style={{ left: `${x(v)}%`, zIndex: hover === id ? 10 : 1 }}
                      aria-label={`${m.name} ${metric.label} ${pct(v, 2)}`}
                    >
                      <span
                        className="block size-3 rounded-full"
                        style={{ background: MODEL_COLORS[m.name], boxShadow: "0 0 0 2px var(--surface)" }}
                      />
                      {hover === id && (
                        <span className="pointer-events-none absolute bottom-full mb-1 whitespace-nowrap rounded-control border border-line bg-page px-2 py-1 text-left text-[11px] shadow-lg">
                          <strong className="block text-sm text-ink tabular">{pct(v, 2)}</strong>
                          <span className="text-ink-2">{m.name} · {metric.label}</span>
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
          <div className="grid grid-cols-[5.5rem_1fr] gap-3">
            <span />
            <div className="relative h-4 text-[11px] text-muted tabular">
              {ticks.map((t, i) => (
                <span
                  key={t}
                  className={`absolute ${i === ticks.length - 1 ? "-translate-x-full" : i === 0 ? "" : "-translate-x-1/2"}`}
                  style={{ left: `${x(t)}%` }}
                >
                  {pct(t, 1)}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </ChartFrame>
  );
}
