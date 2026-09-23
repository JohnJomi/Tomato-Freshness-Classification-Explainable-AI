"""Shared dependency: the services loaded at startup."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request

from .services.lime_service import LimeService
from .services.model_service import ModelService
from .services.shap_service import ShapService


@dataclass
class Services:
    model: ModelService
    lime: LimeService
    shap: ShapService


def get_services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(request.app.state, "load_error", None)
            or "ML service is not ready.",
        )
    return services
