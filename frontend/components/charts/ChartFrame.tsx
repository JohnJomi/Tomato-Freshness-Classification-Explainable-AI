"use client";

import { useState } from "react";
import { Table2, BarChart3 } from "lucide-react";
import { Card } from "@/components/ui/Card";

/** Card with a chart/table toggle, so every value is reachable without hover. */
export function ChartFrame({
  title,
  subtitle,
  table,
  extra,
  children,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  table?: React.ReactNode;
  extra?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <Card
      title={title}
      subtitle={subtitle}
      action={
        <div className="flex items-center gap-2">
          {extra}
          {table && (
            <button
              onClick={() => setShowTable((v) => !v)}
              className="inline-flex items-center gap-1.5 rounded-control border border-line bg-surface-2 px-2.5 py-1 text-xs text-ink-2 hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-series-1"
              aria-pressed={showTable}
            >
              {showTable ? <BarChart3 className="size-3.5" aria-hidden /> : <Table2 className="size-3.5" aria-hidden />}
              {showTable ? "Chart" : "Table"}
            </button>
          )}
        </div>
      }
    >
      {showTable ? <div className="overflow-x-auto">{table}</div> : children}
    </Card>
  );
}

export function DataTable({ head, rows }: { head: string[]; rows: (React.ReactNode)[][] }) {
  return (
    <table className="w-full text-left text-sm tabular">
      <thead>
        <tr className="border-b border-line text-xs text-muted">
          {head.map((h, i) => (
            <th key={h} className={`py-2 pr-4 font-medium ${i ? "text-right" : ""}`}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-b border-line/50 last:border-0">
            {r.map((c, j) => (
              <td key={j} className={`py-2 pr-4 ${j ? "text-right text-ink-2" : "text-ink"}`}>{c}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Legend: mirrors the mark (dot for dot plots, rect for bars). Text stays in ink. */
export function Legend({ items, mark = "rect" }: { items: { label: string; color: string }[]; mark?: "rect" | "dot" }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5">
          <span
            className={mark === "dot" ? "size-2.5 rounded-full" : "h-2.5 w-3 rounded-[2px]"}
            style={{ background: i.color }}
            aria-hidden
          />
          {i.label}
        </li>
      ))}
    </ul>
  );
}

/** Recharts tooltip: value leads, label follows. */
export function ChartTooltip({
  active,
  payload,
  label,
  format,
}: {
  active?: boolean;
  payload?: { value?: number | string; payload?: Record<string, unknown> }[];
  label?: string | number;
  format: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  const v = Number(payload[0].value);
  return (
    <div className="rounded-control border border-line bg-page/95 px-3 py-2 text-xs shadow-lg">
      <p className="text-sm font-semibold text-ink tabular">{format(v)}</p>
      <p className="text-ink-2">{String(label ?? "")}</p>
    </div>
  );
}
