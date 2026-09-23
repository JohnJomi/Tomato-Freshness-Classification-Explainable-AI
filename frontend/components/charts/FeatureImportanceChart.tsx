"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartFrame, ChartTooltip, DataTable } from "./ChartFrame";

export function FeatureImportanceChart({
  title,
  subtitle,
  items,
  valueLabel,
  extra,
  digits = 4,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  items: { name: string; value: number }[];
  valueLabel: string;
  extra?: React.ReactNode;
  digits?: number;
}) {
  const fmt = (v: number) => v.toFixed(digits);
  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      extra={extra}
      table={<DataTable head={["Rank", "Feature", valueLabel]} rows={items.map((d, i) => [i + 1, d.name, fmt(d.value)])} />}
    >
      <div style={{ height: 28 * items.length + 40 }} role="img" aria-label={`${valueLabel} by feature, highest first: ${items.slice(0, 5).map((d) => d.name).join(", ")}`}>
        <ResponsiveContainer>
          <BarChart data={items} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 8 }} barCategoryGap={4}>
            <CartesianGrid horizontal={false} stroke="var(--grid)" />
            <XAxis type="number" tickLine={false} axisLine={{ stroke: "var(--axis)" }} tick={{ fill: "var(--muted)", fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              width={160}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "var(--text-2)", fontSize: 12 }}
            />
            <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip format={fmt} />} />
            <Bar dataKey="value" fill="var(--series-1)" radius={[0, 4, 4, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}
