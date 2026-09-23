from fastapi import APIRouter, Depends

from ..deps import Services, get_services

router = APIRouter(tags=["models"])


@router.get("/models")
def models(s: Services = Depends(get_services)):
    cms = s.model.confusion_matrices
    return {
        "class_order": cms["class_order"],
        "selection": s.model.model_selection,
        "models": [
            {
                "name": m["Model"],
                # final evaluation on the held-out test set
                "accuracy": m["Accuracy"],
                "precision": m["Precision"],
                "recall": m["Recall"],
                "f1": m["F1-score"],
                "f1_macro": m["F1-macro"],
                # training-set 5-fold CV (used for model selection)
                "cv_accuracy_mean": m["CV Accuracy Mean"],
                "cv_accuracy_std": m["CV Accuracy Std"],
                "cv_f1_mean": m["CV F1-weighted Mean"],
                "cv_f1_std": m["CV F1-weighted Std"],
                "confusion_matrix": cms["matrices"][m["Model"]],
                "selected_for_xai": m["Model"] == s.model.xai_model_name,
            }
            for m in s.model.metrics
        ],
    }
