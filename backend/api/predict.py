from fastapi import APIRouter, Depends, HTTPException

from ..deps import Services, get_services
from ..schemas import PredictRequest

router = APIRouter(tags=["predict"])


@router.post("/predict")
def predict(req: PredictRequest, s: Services = Depends(get_services)):
    try:
        return s.model.predict(req.features, req.model)
    except KeyError:
        raise HTTPException(
            422, f"Unknown model '{req.model}'. Choose one of {list(s.model.models)}."
        )
