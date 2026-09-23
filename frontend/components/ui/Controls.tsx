import { cn } from "@/lib/utils";

const base =
  "rounded-control border border-line bg-surface-2 text-sm text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-primary";

export function Button({
  variant = "primary",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" }) {
  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-control px-4 py-2 text-sm font-medium transition-[transform,background-color,border-color,box-shadow] duration-150 ease-out active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
        variant === "primary"
          ? "bg-primary text-white shadow-[0_1px_2px_rgba(6,78,59,0.15)] hover:-translate-y-px hover:scale-[1.01] hover:bg-primary-dark"
          : "border border-primary/30 bg-surface text-ink hover:-translate-y-px hover:scale-[1.01] hover:border-primary/50 hover:bg-primary-soft/40",
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
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap rounded-control border border-line bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o}
          role="radio"
          aria-checked={o === value}
          onClick={() => onChange(o)}
          className={cn(
            "rounded-control px-3 py-1.5 text-xs font-medium transition-colors duration-150 ease-out focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
            o === value ? "bg-primary text-white" : "text-ink-2 hover:text-primary",
          )}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

export const inputClass = cn(
  base,
  "w-full px-2.5 py-1.5 tabular transition-colors duration-150 ease-out focus-visible:border-primary",
);
