"""Request models. Responses are plain dicts shaped by the services."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator

from .services.model_service import FEATURE_COLUMNS


def _validate_features(features: dict[str, float]) -> dict[str, float]:
    missing = [f for f in FEATURE_COLUMNS if f not in features]
    unknown = [f for f in features if f not in FEATURE_COLUMNS]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing features: {missing}")
        if unknown:
            parts.append(f"unknown features: {unknown}")
        raise ValueError("; ".join(parts))
    bad = [f for f, v in features.items() if not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite values for: {bad}")
    return features


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="All 26 sensor features.")
    model: str | None = Field(
        None, description="Random Forest | SVM | XGBoost. Defaults to the CV-selected model."
    )

    @field_validator("features")
    @classmethod
    def _check_features(cls, v):
        return _validate_features(v)


class ExplainRequest(BaseModel):
    """Explain either custom feature values or a held-out test sample (exactly one)."""

    features: dict[str, float] | None = None
    sample_index: int | None = Field(None, ge=0)

    @field_validator("features")
    @classmethod
    def _check_features(cls, v):
        return None if v is None else _validate_features(v)

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.features is None) == (self.sample_index is None):
            raise ValueError("Provide exactly one of 'features' or 'sample_index'.")
        return self
