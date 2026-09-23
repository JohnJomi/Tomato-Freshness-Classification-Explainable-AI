from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import Services, get_services
from ..schemas import ExplainRequest
from ..services.model_service import CLASS_ORDER

router = APIRouter(tags=["explanations"])


def _resolve(req: ExplainRequest, s: Services) -> tuple[dict, dict]:
    """Return (features, extra response fields) for a request."""
    if req.sample_index is None:
        return req.features, {"sample_index": None, "actual_class": None}
    try:
        features, actual = s.model.test_sample(req.sample_index)
    except IndexError:
        raise HTTPException(404, f"Test sample {req.sample_index} does not exist.")
    return features, {"sample_index": req.sample_index, "actual_class": actual}


@router.get("/feature-importance")
def feature_importance(s: Services = Depends(get_services)):
    fi = s.model.feature_importance
    return {
        "model": fi["model"],
        "method": fi["method"],
        "explanation_type": "global",
        "features": [{"name": f["feature"], "importance": f["importance"]} for f in fi["features"]],
    }


@router.post("/explain/lime")
def explain_lime(req: ExplainRequest, s: Services = Depends(get_services)):
    features, extra = _resolve(req, s)
    return {**extra, **s.lime.explain(features)}


@router.post("/explain/shap")
def explain_shap(req: ExplainRequest, s: Services = Depends(get_services)):
    features, extra = _resolve(req, s)
    return {**extra, **s.shap.local_explanation(features)}


@router.get("/explain/shap/global")
def explain_shap_global(
    class_name: str = Query("Stale to Spoiled", description=f"One of {CLASS_ORDER}"),
    s: Services = Depends(get_services),
):
    if class_name not in CLASS_ORDER:
        raise HTTPException(422, f"class_name must be one of {CLASS_ORDER}.")
    return s.shap.global_explanation(class_name)
