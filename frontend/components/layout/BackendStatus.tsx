"use client";

import { CheckCircle2, CircleAlert, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";

/** Status = icon + label, never colour alone. */
export function BackendStatus() {
  const { data, error, loading } = useApi(api.health);
  if (loading && !data)
    return (
      <span className="flex items-center gap-1.5 text-ink-2">
        <Loader2 className="size-3.5 animate-spin" aria-hidden /> Connecting…
      </span>
    );
  if (error || data?.status !== "ok")
    return (
      <span className="flex items-center gap-1.5 text-ink-2" title={error ?? data?.detail}>
        <CircleAlert className="size-3.5 text-critical" aria-hidden /> ML service offline
      </span>
    );
  return (
    <span className="flex items-center gap-1.5 text-ink-2">
      <CheckCircle2 className="size-3.5 text-good" aria-hidden /> ML service online
    </span>
  );
}
