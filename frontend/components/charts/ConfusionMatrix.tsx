import { CLASS_ORDER } from "@/lib/utils";

// Sequential single-hue blue ramp (reference palette), dark-surface steps.
const RAMP = ["#1a2433", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"];

function cellColor(t: number) {
  return RAMP[Math.min(RAMP.length - 1, Math.round(t * (RAMP.length - 1)))];
}

export function ConfusionMatrix({ name, matrix, accent }: { name: string; matrix: number[][]; accent: string }) {
  const max = Math.max(...matrix.flat(), 1);
  const short = ["Pure Fresh", "Good", "Stale→Spoiled"];
  return (
    <figure className="min-w-0">
      <figcaption className="mb-2 flex items-center gap-2 text-sm font-medium text-ink">
        <span className="size-2.5 rounded-full" style={{ background: accent }} aria-hidden />
        {name}
      </figcaption>
      <div className="grid grid-cols-[auto_repeat(3,minmax(0,1fr))] gap-0.5 text-xs">
        <span />
        {short.map((c) => (
          <span key={c} className="truncate px-1 pb-1 text-center text-muted" title={c}>{c}</span>
        ))}
        {matrix.map((row, i) => (
          <div key={i} className="contents">
            <span className="truncate pr-2 text-right leading-[3rem] text-muted" title={CLASS_ORDER[i]}>{short[i]}</span>
            {row.map((v, j) => {
              const t = v / max;
              return (
                <span
                  key={j}
                  className="grid h-12 place-items-center rounded-[4px] text-sm font-semibold tabular"
                  style={{ background: v ? cellColor(t) : "var(--surface-2)", color: t > 0.6 ? "#0b0b0b" : "#ffffff" }}
                  title={`Actual ${CLASS_ORDER[i]}, predicted ${CLASS_ORDER[j]}: ${v}`}
                >
                  {v}
                </span>
              );
            })}
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-muted">Rows: actual · Columns: predicted</p>
    </figure>
  );
}
