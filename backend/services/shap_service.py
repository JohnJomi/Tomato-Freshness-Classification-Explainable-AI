"""SHAP global + local explanations as structured JSON for the frontend."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model_service import CLASS_ORDER, FEATURE_COLUMNS, ModelService

from src.shap_explainer import build_explanation  # noqa: E402  (ml/src on sys.path)


def _class_values(values: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize to values (n, features, classes) and base (n, classes)."""
    values = np.asarray(values)
    base = np.asarray(base)
    if base.ndim == 1:
        base = np.tile(base, (values.shape[0], 1))
    return values, base


class ShapService:
    def __init__(self, service: ModelService):
        self.service = service
        self.model_name, self.model = service.resolve_model(None)
        # Global view over the held-out test set, computed once at startup.
        explanation, X_eval = build_explanation(
            self.model, service.X_test, X_background=service.X_train
        )
        self.global_values, self.global_base = _class_values(
            explanation.values, explanation.base_values
        )
        self.global_features = np.asarray(X_eval)

    def global_explanation(self, class_name: str, max_points: int | None = None) -> dict[str, Any]:
        c = CLASS_ORDER.index(class_name)
        mean_abs_all = np.abs(self.global_values).mean(axis=(0, 2))
        mean_abs_class = np.abs(self.global_values[:, :, c]).mean(axis=0)
        order = np.argsort(mean_abs_all)[::-1]

        # Beeswarm data: per feature, (shap value, feature value) per sample,
        # plus min/max so the frontend can colour points by feature value.
        n = self.global_values.shape[0] if max_points is None else min(max_points, self.global_values.shape[0])
        summary = []
        for i in order:
            vals = self.global_features[:n, i]
            summary.append(
                {
                    "feature": FEATURE_COLUMNS[i],
                    "feature_min": float(vals.min()),
                    "feature_max": float(vals.max()),
                    "points": [
                        {"shap": float(s), "value": float(v)}
                        for s, v in zip(self.global_values[:n, i, c], vals)
                    ],
                }
            )

        return {
            "model": self.model_name,
            "explanation_type": "global",
            "method": "SHAP",
            "explained_class": class_name,
            "n_samples": int(self.global_values.shape[0]),
            "importance": [
                {
                    "feature": FEATURE_COLUMNS[i],
                    "mean_abs_shap": float(mean_abs_all[i]),
                    "mean_abs_shap_class": float(mean_abs_class[i]),
                }
                for i in order
            ],
            "summary": summary,
        }

    def local_explanation(self, features: dict[str, float]) -> dict[str, Any]:
        X = self.service.to_frame(features)
        explanation, _ = build_explanation(self.model, X, X_background=self.service.X_train)
        values, base = _class_values(explanation.values, explanation.base_values)
        proba = self.model.predict_proba(X)[0]
        label = int(np.argmax(proba))

        return {
            "model": self.model_name,
            "explanation_type": "local",
            "method": "SHAP",
            "prediction": CLASS_ORDER[label],
            "confidence": float(proba[label]),
            "probabilities": {c: float(p) for c, p in zip(CLASS_ORDER, proba)},
            "feature_names": FEATURE_COLUMNS,
            "feature_values": [float(features[f]) for f in FEATURE_COLUMNS],
            "class_names": CLASS_ORDER,
            # per class: baseline + sum(contributions) = model output for that class
            "base_values": {c: float(base[0, k]) for k, c in enumerate(CLASS_ORDER)},
            "shap_values": {
                c: [float(v) for v in values[0, :, k]] for k, c in enumerate(CLASS_ORDER)
            },
            "output_values": {
                c: float(base[0, k] + values[0, :, k].sum()) for k, c in enumerate(CLASS_ORDER)
            },
        }
