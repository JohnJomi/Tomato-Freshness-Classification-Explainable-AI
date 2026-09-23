"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useFromPlayground, usePlaygroundFeatures } from "@/lib/playground";
import type { ExplainInput } from "@/lib/types";
import { useApi } from "@/lib/useApi";
import { Select } from "@/components/ui/Controls";

export const DEFAULT_SAMPLE = 6; // same sample the report explains

/** Choose what to explain: a held-out test sample, or the Playground values. */
export function useExplainInput(): {
  input: ExplainInput | null;
  picker: React.ReactNode;
} {
  const samples = useApi(api.samples);
  const playground = usePlaygroundFeatures();
  const fromPlayground = useFromPlayground();
  // null = not chosen yet: Playground values if we came from there, else the default sample.
  const [picked, setChoice] = useState<string | null>(null);
  const choice = picked ?? (playground && fromPlayground ? "playground" : String(DEFAULT_SAMPLE));

  const input: ExplainInput | null =
    choice === "playground" ? (playground ? { features: playground } : null) : { sample_index: Number(choice) };

  const picker = (
    <Select label="Explain" value={choice} onChange={(e) => setChoice(e.target.value)} className="w-full sm:w-72" disabled={!samples.data}>
      {playground && <option value="playground">Values from Prediction Playground</option>}
      {samples.data ? (
        <optgroup label="Held-out test samples (#index · actual class)">
          {samples.data.samples.map((s) => (
            <option key={s.index} value={s.index}>
              #{s.index} · {s.actual_class}
            </option>
          ))}
        </optgroup>
      ) : (
        <option value={choice}>{samples.error ? "Samples unavailable" : "Loading samples…"}</option>
      )}
    </Select>
  );

  return { input, picker };
}
