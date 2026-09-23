import { CheckCircle2, XCircle } from "lucide-react";
import { pct } from "@/lib/utils";

/** Actual vs predicted + confidence for an explained sample. */
export function PredictionSummary({
  actual,
  predicted,
  confidence,
}: {
  actual: string | null;
  predicted: string;
  confidence: number;
}) {
  const correct = actual === predicted;
  return (
    <dl className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <div className="rounded-control border border-line bg-surface-2 p-3">
        <dt className="text-xs text-muted">Actual class</dt>
        <dd className="mt-0.5 font-medium text-ink">{actual ?? "Unknown (custom input)"}</dd>
      </div>
      <div className="rounded-control border border-line bg-surface-2 p-3">
        <dt className="text-xs text-muted">Predicted class</dt>
        <dd className="mt-0.5 flex items-center gap-1.5 font-medium text-ink">
          {predicted}
          {actual &&
            (correct ? (
              <span className="inline-flex items-center gap-1 text-xs font-normal text-ink-2">
                <CheckCircle2 className="size-3.5 text-good" aria-hidden /> correct
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-xs font-normal text-ink-2">
                <XCircle className="size-3.5 text-critical" aria-hidden /> misclassified
              </span>
            ))}
        </dd>
      </div>
      <div className="rounded-control border border-line bg-surface-2 p-3">
        <dt className="text-xs text-muted">Confidence</dt>
        <dd className="mt-0.5 font-medium text-ink tabular">{pct(confidence)}</dd>
      </div>
    </dl>
  );
}
