"""LIME local explanations as structured JSON for the frontend."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from .model_service import CLASS_ORDER, FEATURE_COLUMNS, ModelService


class LimeService:
    def __init__(self, service: ModelService, num_features: int = 10):
        self.service = service
        self.num_features = num_features
        # Same configuration as ml/src/lime_explainer.py (training data, seed 42).
        self.explainer = LimeTabularExplainer(
            training_data=service.X_train.values,
            feature_names=FEATURE_COLUMNS,
            class_names=CLASS_ORDER,
            mode="classification",
            random_state=42,
        )

    def explain(self, features: dict[str, float]) -> dict[str, Any]:
        name, model = self.service.resolve_model(None)

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            # Keep feature names so the fitted pipeline doesn't warn.
            return model.predict_proba(pd.DataFrame(arr, columns=FEATURE_COLUMNS))

        x = np.array([features[f] for f in FEATURE_COLUMNS])
        proba = predict_fn(x.reshape(1, -1))[0]
        label = int(np.argmax(proba))

        exp = self.explainer.explain_instance(
            x, predict_fn, num_features=self.num_features, labels=(label,)
        )
        # as_map() (feature indices) and as_list() (readable rules) share one order.
        contributions = [
            {
                "feature": FEATURE_COLUMNS[idx],
                "rule": rule,
                "value": float(x[idx]),
                "contribution": float(weight),
            }
            for (idx, weight), (rule, _) in zip(exp.as_map()[label], exp.as_list(label=label))
        ]

        return {
            "model": name,
            "explanation_type": "local",
            "method": "LIME",
            "explained_class": CLASS_ORDER[label],
            "prediction": CLASS_ORDER[label],
            "confidence": float(proba[label]),
            "probabilities": {c: float(p) for c, p in zip(CLASS_ORDER, proba)},
            "intercept": float(exp.intercept[label]),
            "contributions": contributions,
        }
