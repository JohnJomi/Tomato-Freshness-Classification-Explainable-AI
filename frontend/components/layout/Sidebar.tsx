"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  BarChart3,
  Database,
  FlaskConical,
  LayoutDashboard,
  ListOrdered,
  Menu,
  Microscope,
  Sigma,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { BackendStatus } from "./BackendStatus";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/predict", label: "Prediction", icon: FlaskConical },
  { href: "/models", label: "Models", icon: BarChart3 },
  { href: "/features", label: "Features", icon: ListOrdered },
  { href: "/explain/lime", label: "LIME", icon: Microscope },
  { href: "/explain/shap", label: "SHAP", icon: Sigma },
  { href: "/dataset", label: "Dataset", icon: Database },
];

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-2 px-2 text-ink">
      <span className="grid size-7 place-items-center rounded-md bg-surface-2 text-base" aria-hidden>
        🍅
      </span>
      <span className="font-semibold tracking-tight">TomatoAI</span>
    </Link>
  );
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const path = usePathname();
  return (
    <nav className="flex flex-col gap-0.5" aria-label="Main">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
              active ? "bg-surface-2 text-ink font-medium" : "text-ink-2 hover:bg-surface-2/60 hover:text-ink",
            )}
          >
            <Icon className="size-4" aria-hidden />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}

function Footer() {
  return (
    <div className="mt-auto space-y-3 border-t border-line px-2 pt-4 text-xs">
      <BackendStatus />
      <p className="text-muted">Experiment v1.0</p>
    </div>
  );
}

export function Sidebar() {
  const path = usePathname();
  // The drawer belongs to the page it was opened on, so any navigation closes it.
  const [openOn, setOpenOn] = useState<string | null>(null);
  const open = openOn === path;
  const setOpen = (v: boolean) => setOpenOn(v ? path : null);

  return (
    <>
      {/* desktop */}
      <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 flex-col gap-6 border-r border-line bg-surface p-4 lg:flex">
        <Brand />
        <NavLinks />
        <Footer />
      </aside>

      {/* mobile top bar + drawer */}
      <div className="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-surface/95 px-4 py-3 backdrop-blur lg:hidden">
        <Brand />
        <button
          onClick={() => setOpen(true)}
          className="rounded-md p-2 text-ink-2 hover:bg-surface-2 hover:text-ink"
          aria-label="Open navigation"
          aria-expanded={open}
        >
          <Menu className="size-5" />
        </button>
      </div>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 flex w-64 flex-col gap-6 border-r border-line bg-surface p-4 animate-fade-in">
            <div className="flex items-center justify-between">
              <Brand />
              <button onClick={() => setOpen(false)} className="rounded-md p-2 text-ink-2 hover:bg-surface-2" aria-label="Close navigation">
                <X className="size-5" />
              </button>
            </div>
            <NavLinks onNavigate={() => setOpen(false)} />
            <Footer />
          </aside>
        </div>
      )}
    </>
  );
}
