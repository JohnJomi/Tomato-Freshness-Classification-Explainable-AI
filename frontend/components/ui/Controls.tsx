import { cn } from "@/lib/utils";

const base =
  "rounded-md border border-line bg-surface-2 text-sm text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-series-1";

export function Button({
  variant = "primary",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" }) {
  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-series-1",
        variant === "primary"
          ? "bg-ink text-page hover:bg-ink-2"
          : "border border-line bg-surface-2 text-ink hover:bg-surface",
        className,
      )}
    />
  );
}

export function Select({
  label,
  className,
  children,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement> & { label: string }) {
  return (
    <label className={cn("flex flex-col gap-1 text-xs text-muted", className)}>
      {label}
      <select {...props} className={cn(base, "px-3 py-2")}>
        {children}
      </select>
    </label>
  );
}

export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap rounded-md border border-line bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o}
          role="radio"
          aria-checked={o === value}
          onClick={() => onChange(o)}
          className={cn(
            "rounded px-3 py-1.5 text-xs font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-series-1",
            o === value ? "bg-ink text-page" : "text-ink-2 hover:text-ink",
          )}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

export const inputClass = cn(base, "w-full px-2.5 py-1.5 tabular");
