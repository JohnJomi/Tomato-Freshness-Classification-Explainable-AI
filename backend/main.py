"""FastAPI backend for the tomato freshness ML/XAI pipeline.

Run from the project root (after `python ml/run_pipeline.py`):

    source .venv/bin/activate
    uvicorn backend.main:app --reload --port 8000

Artifacts are loaded once at startup; nothing is retrained per request.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import dataset, explanations, models, predict
from .deps import Services
from .services.lime_service import LimeService
from .services.model_service import ArtifactsMissingError, ModelService
from .services.shap_service import ShapService

logger = logging.getLogger("tomato-backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = None
    app.state.load_error = None
    try:
        model_service = ModelService.load()
        app.state.services = Services(
            model=model_service,
            lime=LimeService(model_service),
            shap=ShapService(model_service),
        )
        logger.info("Loaded ML artifacts (XAI model: %s)", model_service.xai_model_name)
    except ArtifactsMissingError as exc:
        app.state.load_error = str(exc)
        logger.error(app.state.load_error)
    yield


app = FastAPI(title="Tomato Freshness ML API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    # Log the full traceback server-side; never expose it to clients.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", tags=["health"])
def health(request: Request):
    services = request.app.state.services
    if services is None:
        return {"status": "degraded", "detail": request.app.state.load_error}
    return {"status": "ok", "xai_model": services.model.xai_model_name}


for module in (dataset, models, predict, explanations):
    app.include_router(module.router)
