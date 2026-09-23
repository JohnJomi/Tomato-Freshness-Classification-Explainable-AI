import { AlertTriangle, Loader2, RotateCw } from "lucide-react";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-10 justify-center text-sm text-ink-2">
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </div>
  );
}

export function Skeleton({ className = "h-24" }: { className?: string }) {
  return <div className={`animate-pulse rounded-card border border-line bg-surface ${className}`} aria-hidden />;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-3 rounded-card border border-critical/40 bg-critical/10 p-4 text-sm">
      <div className="flex items-start gap-2 text-ink">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-critical" aria-hidden />
        <span>{message}</span>
      </div>
      {onRetry && (
        <button onClick={onRetry} className="inline-flex items-center gap-1.5 rounded-control border border-line bg-surface-2 px-3 py-1.5 text-xs text-ink hover:bg-surface">
          <RotateCw className="size-3.5" aria-hidden /> Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="rounded-card border border-dashed border-line p-8 text-center text-sm text-muted">{children}</div>;
}
