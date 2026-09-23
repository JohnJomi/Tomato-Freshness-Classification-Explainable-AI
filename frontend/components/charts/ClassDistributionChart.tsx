"use client";

import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartFrame, ChartTooltip, DataTable } from "./ChartFrame";
import { CLASS_ORDER, pct } from "@/lib/utils";

export function ClassDistributionChart({ distribution }: { distribution: Record<string, number> }) {
  const total = Object.values(distribution).reduce((a, b) => a + b, 0);
  const data = CLASS_ORDER.map((c) => ({ name: c, count: distribution[c] ?? 0 }));
  return (
    <ChartFrame
      title="Class distribution"
      subtitle="After merging Stale + Spoiled into Stale to Spoiled"
      table={<DataTable head={["Class", "Samples", "Share"]} rows={data.map((d) => [d.name, d.count, pct(d.count / total)])} />}
    >
      <div className="h-64" role="img" aria-label={`Class distribution: ${data.map((d) => `${d.name} ${d.count}`).join(", ")}`}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 20, right: 8, bottom: 0, left: -12 }} barCategoryGap="30%">
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: "var(--axis)" }} tick={{ fill: "var(--muted)", fontSize: 12 }} />
            <YAxis tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 12 }} />
            <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip format={(v) => `${v} samples (${pct(v / total)})`} />} />
            <Bar dataKey="count" fill="var(--series-1)" radius={[4, 4, 0, 0]} isAnimationActive={false}>
              <LabelList dataKey="count" position="top" fill="var(--text-2)" fontSize={12} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}
