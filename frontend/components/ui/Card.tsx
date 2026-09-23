import { cn } from "@/lib/utils";

export function Card({
  title,
  subtitle,
  action,
  children,
  className,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("min-w-0 rounded-xl border border-line bg-surface p-4 sm:p-5 animate-fade-in", className)}>
      {(title || action) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatTile({ label, value, hint }: { label: string; value: React.ReactNode; hint?: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4 animate-fade-in">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-ink">{value}</p>
      {hint && <p className="mt-1 text-xs text-ink-2">{hint}</p>}
    </div>
  );
}

export function Badge({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border border-line bg-surface-2 px-2 py-0.5 text-xs text-ink-2", className)}>
      {children}
    </span>
  );
}

/** Tells the reader what an explanation covers, per the UX requirements. */
export function ExplanationMeta({ model, scope, what }: { model: string; scope: "local" | "global"; what: string }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <Badge>{scope === "local" ? "Local explanation" : "Global explanation"}</Badge>
      <Badge>Model: {model}</Badge>
      <span className="text-muted">{what}</span>
    </div>
  );
}
