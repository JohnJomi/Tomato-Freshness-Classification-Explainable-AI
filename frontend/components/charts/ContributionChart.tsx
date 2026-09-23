"use client";

import { useState } from "react";
import { Legend } from "./ChartFrame";

export interface Contribution {
  label: string; // feature name or LIME rule
  value: number; // signed contribution
  detail?: string; // e.g. "value = 0.011"
}

/** Diverging horizontal bars around zero. Positive = supports the explained
 *  class (red pole), negative = opposes (blue pole). Sign is also shown as
 *  text, so colour never carries it alone. */
export function ContributionChart({
  items,
  explainedClass,
  format = (v) => v.toFixed(4),
}: {
  items: Contribution[];
  explainedClass: string;
  format?: (v: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);

  return (
    <div className="space-y-3">
      <Legend
        items={[
          { label: `Supports ${explainedClass}`, color: "var(--pos)" },
          { label: `Opposes ${explainedClass}`, color: "var(--neg)" },
        ]}
      />
      <ul className="space-y-1" role="list">
        {items.map((it, i) => {
          const w = (Math.abs(it.value) / max) * 50;
          const pos = it.value >= 0;
          return (
            <li
              key={i}
              tabIndex={0}
              onPointerEnter={() => setHover(i)}
              onPointerLeave={() => setHover(null)}
              onFocus={() => setHover(i)}
              onBlur={() => setHover(null)}
              className="grid grid-cols-[minmax(0,12rem)_1fr_4.5rem] items-center gap-3 rounded px-1 py-1 text-xs outline-none focus-visible:ring-2 focus-visible:ring-series-1 sm:grid-cols-[minmax(0,16rem)_1fr_5rem]"
              aria-label={`${it.label}: ${format(it.value)} (${pos ? "supports" : "opposes"})`}
            >
              <span className="truncate text-ink-2" title={it.label}>{it.label}</span>
              <span className="relative h-5">
                <span className="absolute inset-y-0 left-1/2 w-px bg-[var(--axis)]" aria-hidden />
                <span
                  className="absolute inset-y-0.5 transition-opacity"
                  style={{
                    left: pos ? "50%" : `${50 - w}%`,
                    width: `${w}%`,
                    background: pos ? "var(--pos)" : "var(--neg)",
                    borderRadius: pos ? "0 4px 4px 0" : "4px 0 0 4px",
                    opacity: hover === null || hover === i ? 1 : 0.45,
                  }}
                />
                {hover === i && it.detail && (
                  <span className="pointer-events-none absolute -top-8 left-1/2 z-10 -translate-x-1/2 whitespace-nowrap rounded-md border border-line bg-page px-2 py-1 text-[11px] text-ink-2 shadow-lg">
                    <strong className="text-ink tabular">{format(it.value)}</strong> · {it.detail}
                  </span>
                )}
              </span>
              <span className="text-right text-ink tabular">{it.value >= 0 ? "+" : "−"}{format(Math.abs(it.value))}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
