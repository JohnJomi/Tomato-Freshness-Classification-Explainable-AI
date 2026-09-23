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
    <section
      className={cn(
        "min-w-0 rounded-card border border-line bg-surface p-4 shadow-[0_4px_20px_rgba(0,0,0,0.04)] transition-[transform,box-shadow,border-color] duration-200 ease-out animate-fade-in hover:-translate-y-0.5 hover:border-line hover:shadow-[0_8px_28px_rgba(0,0,0,0.07)] sm:p-5",
        className,
      )}
    >
      {(title || action) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
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
    <div className="rounded-card border border-line bg-surface p-4 shadow-[0_4px_20px_rgba(0,0,0,0.04)] transition-[transform,box-shadow] duration-200 ease-out animate-fade-in hover:-translate-y-0.5 hover:shadow-[0_8px_28px_rgba(0,0,0,0.07)]">
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
