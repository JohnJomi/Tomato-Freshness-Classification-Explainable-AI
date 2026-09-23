"use client";

import { useEffect, useRef, useState } from "react";
import type { ShapGlobal } from "@/lib/types";

// Sequential single-hue blue ramp for feature value (low -> high), validated
// on the white card surface. A different hue from the green categorical/
// diverging colors used elsewhere on this page, so "feature value" doesn't
// read as "supports/opposes".
const RAMP = ["#1e3a8a", "#1d4ed8", "#3b82f6", "#60a5fa"];
const colorFor = (t: number) => RAMP[Math.min(RAMP.length - 1, Math.max(0, Math.round(t * (RAMP.length - 1))))];

const ROW = 30;
const LABEL_W = 150;

/** SHAP summary (beeswarm): one row per feature, one dot per test sample,
 *  x = SHAP value, colour = the sample's feature value (low -> high). */
export function Beeswarm({ summary, top = 12 }: { summary: ShapGlobal["summary"]; top?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0); // measured; never forces the layout wider
  const [hover, setHover] = useState<{ x: number; y: number; text: string; value: string } | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const rows = summary.slice(0, top);
  const all = rows.flatMap((r) => r.points.map((p) => p.shap));
  const ext = Math.max(Math.abs(Math.min(...all)), Math.abs(Math.max(...all)), 1e-9);
  const narrow = width < 480;
  const labelW = narrow ? 118 : LABEL_W;
  // On phones, shorten ratio names so they fit the label column.
  const label = (f: string) => (narrow ? f.replace("_over_", "/") : f);
  const plotW = Math.max(width - labelW - 12, 100);
  const sx = (v: number) => labelW + ((v + ext) / (2 * ext)) * plotW;
  const height = rows.length * ROW + 28;

  return (
    <div ref={ref} className="relative w-full min-w-0">
      {width > 0 && <svg width={width} height={height} role="img" aria-label="SHAP summary plot">
        <line x1={sx(0)} x2={sx(0)} y1={0} y2={rows.length * ROW} stroke="var(--axis)" />
        {rows.map((r, ri) => {
          const cy = ri * ROW + ROW / 2;
          const span = r.feature_max - r.feature_min || 1;
          // Deterministic stacking: bin by x, offset alternately above/below.
          const bins = new Map<number, number>();
          return (
            <g key={r.feature}>
              <line x1={labelW} x2={labelW + plotW} y1={cy} y2={cy} stroke="var(--grid)" />
              <text x={labelW - 8} y={cy} textAnchor="end" dominantBaseline="middle" fill="var(--text-2)" fontSize={11}>
                <title>{r.feature}</title>
                {label(r.feature)}
              </text>
              {r.points.map((p, pi) => {
                const px = sx(p.shap);
                const bin = Math.round(px / 4);
                const n = bins.get(bin) ?? 0;
                bins.set(bin, n + 1);
                const off = (n % 2 ? 1 : -1) * Math.ceil(n / 2) * 3;
                const py = cy + Math.max(-ROW / 2 + 3, Math.min(ROW / 2 - 3, off));
                const t = (p.value - r.feature_min) / span;
                return (
                  <circle
                    key={pi}
                    cx={px}
                    cy={py}
                    r={2.6}
                    fill={colorFor(t)}
                    onPointerEnter={() =>
                      setHover({ x: px, y: py, value: `SHAP ${p.shap >= 0 ? "+" : ""}${p.shap.toFixed(4)}`, text: `${r.feature} = ${p.value.toPrecision(4)}` })
                    }
                    onPointerLeave={() => setHover(null)}
                    stroke="transparent"
                    strokeWidth={6}
                  />
                );
              })}
            </g>
          );
        })}
        <text x={sx(0)} y={rows.length * ROW + 18} textAnchor="middle" fill="var(--muted)" fontSize={11}>0</text>
        <text x={labelW} y={rows.length * ROW + 18} fill="var(--muted)" fontSize={11}>← lowers probability</text>
        <text x={labelW + plotW} y={rows.length * ROW + 18} textAnchor="end" fill="var(--muted)" fontSize={11}>raises probability →</text>
      </svg>}
      {hover && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-control border border-line bg-page px-2 py-1 text-[11px] shadow-lg"
          style={{ left: hover.x, top: hover.y - 8 }}
        >
          <strong className="block text-ink tabular">{hover.value}</strong>
          <span className="text-ink-2">{hover.text}</span>
        </div>
      )}
      <div className="mt-2 flex items-center gap-2 text-[11px] text-muted" style={{ paddingLeft: labelW }}>
        Feature value: low
        <span className="h-2 w-28 rounded-full" style={{ background: `linear-gradient(90deg, ${RAMP.join(",")})` }} aria-hidden />
        high
      </div>
    </div>
  );
}
