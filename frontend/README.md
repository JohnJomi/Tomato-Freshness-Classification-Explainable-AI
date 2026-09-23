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
Charts use the validated reference data-viz palette (dark steps): one fixed
colour per model, a single hue for single-series bars, a sequential blue for
magnitude, and a red/blue diverging pair for contribution sign. Every chart has
a table view.
