import { CLASS_ORDER, pct } from "@/lib/utils";

/** Class probabilities: a labelled list, one bar each, predicted class emphasised. */
export function ProbabilityBars({ probabilities, predicted }: { probabilities: Record<string, number>; predicted: string }) {
  return (
    <ul className="space-y-2.5">
      {CLASS_ORDER.map((c) => {
        const p = probabilities[c] ?? 0;
        const isPred = c === predicted;
        return (
          <li key={c} className="grid grid-cols-[8.5rem_1fr_3.5rem] items-center gap-3 text-sm">
            <span className={isPred ? "font-medium text-ink" : "text-ink-2"}>{c}</span>
            <span className="h-2.5 rounded-full bg-surface-2" role="presentation">
              <span
                className="block h-full rounded-full transition-[width] duration-500"
                style={{ width: `${Math.max(p * 100, p > 0 ? 1 : 0)}%`, background: isPred ? "var(--series-1)" : "var(--muted)" }}
              />
            </span>
            <span className="text-right text-ink tabular">{pct(p)}</span>
          </li>
        );
      })}
    </ul>
  );
}
