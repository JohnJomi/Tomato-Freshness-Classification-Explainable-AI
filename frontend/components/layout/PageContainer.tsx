export function PageContainer({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:py-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-3xl font-bold tracking-tight text-ink sm:text-4xl">{title}</h1>
          {description && <p className="mt-1 max-w-3xl text-sm text-ink-2">{description}</p>}
        </div>
        {actions}
      </header>
      <div className="space-y-6">{children}</div>
    </div>
  );
}
