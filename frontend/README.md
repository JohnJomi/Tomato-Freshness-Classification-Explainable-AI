# TomatoAI frontend

Next.js (App Router, TypeScript, Tailwind CSS, Recharts, Lucide) dashboard over
the FastAPI backend in `../backend`. All ML logic stays in Python; this app
only renders what the API returns.

```bash
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL if the backend isn't on :8000
npm run dev                  # http://localhost:3000
```

| Route | Page |
|---|---|
| `/` | Overview: dataset size, CV-selected model, test metrics, model comparison, class distribution |
| `/predict` | Prediction Playground: 26 grouped inputs, model choice, probabilities, hand-off to LIME/SHAP |
| `/models` | Dot-plot comparison, sortable test + CV table, three confusion matrices |
| `/features` | Feature importance (model importance or mean \|SHAP\|), top 10/15/26 |
| `/explain/lime` | Local LIME contributions for a test sample or Playground values |
| `/explain/shap` | Global SHAP importance + beeswarm per class; local base + contributions = prediction |
| `/dataset` | Sizes, target mapping, missing values, per-feature statistics |

`NEXT_PUBLIC_API_URL` is inlined at build time, so rebuild after changing it.

## Design

Light, green-on-white analytics theme per `../design.md`. Colors, radii,
typography (Inter) and hover/motion are themed centrally in `app/globals.css`
and the shared primitives (`components/ui/Card.tsx`, `Controls.tsx`); no page
structure changed for this pass. Chart colors are still checked against the
dataviz skill's colorblind/contrast gates for the white card surface:

- Categorical (model identity): green `--series-1`/`--series-2` + a neutral
  slate `--series-3` — three same-hue greens fail the CVD-distinguishability
  gate, so the third model uses ink-neutral instead of a third green.
- Diverging (LIME/SHAP contribution sign): green `--pos` (supports) / red
  `--neg` (opposes).
- Sequential: a 4-step green ramp for confusion-matrix magnitude, and a
  separate 4-step blue ramp for the SHAP beeswarm's feature-value encoding —
  kept a different hue from the green diverging pair so "feature value" and
  "supports/opposes" don't read as the same signal on the same SHAP page.

Every chart still has a table view and value labels, so colour is never the
only way information is conveyed.
