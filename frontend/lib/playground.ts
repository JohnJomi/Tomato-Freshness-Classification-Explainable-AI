"use client";

import { useMemo, useSyncExternalStore } from "react";
import type { Features } from "./types";

// Hand-off from the Prediction Playground to the LIME/SHAP pages.
const KEY = "tomato:playground-features";

export function savePlaygroundFeatures(features: Features) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(features));
  } catch {
    /* storage unavailable: the explain pages fall back to test samples */
  }
}

const noopSubscribe = () => () => {};

function readRaw(): string | null {
  try {
    return sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

/** Playground values (null on the server and when none were saved). */
export function usePlaygroundFeatures(): Features | null {
  const raw = useSyncExternalStore(noopSubscribe, readRaw, () => null);
  return useMemo(() => {
    try {
      return raw ? (JSON.parse(raw) as Features) : null;
    } catch {
      return null;
    }
  }, [raw]);
}

/** True when the page was opened via "Explain with …" from the Playground. */
export function useFromPlayground(): boolean {
  return useSyncExternalStore(
    noopSubscribe,
    () => new URLSearchParams(window.location.search).get("source") === "playground",
    () => false,
  );
}
