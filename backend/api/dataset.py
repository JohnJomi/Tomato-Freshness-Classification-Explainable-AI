from fastapi import APIRouter, Depends, HTTPException

from ..deps import Services, get_services

router = APIRouter(tags=["dataset"])


@router.get("/dataset")
def dataset(s: Services = Depends(get_services)):
    summary = s.model.dataset_summary
    return {
        "samples": summary["samples"],
        "features": summary["n_features"],
        "classes": summary["classes"],
        "feature_names": summary["features"],
        "class_distribution": summary["mapped_distribution"],
        "raw_class_distribution": summary["raw_distribution"],
        "missing_values": 0,
        "train_size": len(s.model.X_train),
        "test_size": len(s.model.X_test),
        "feature_stats": s.model.feature_stats,
    }


@router.get("/dataset/samples")
def samples(s: Services = Depends(get_services)):
    """Held-out test samples the UI can pick for predictions/explanations."""
    return {
        "samples": [
            {"index": i, "actual_class": s.model.test_sample(i)[1]}
            for i in range(len(s.model.X_test))
        ]
    }


@router.get("/dataset/samples/{index}")
def sample(index: int, s: Services = Depends(get_services)):
    try:
        features, actual = s.model.test_sample(index)
    except IndexError:
        raise HTTPException(404, f"Test sample {index} does not exist.")
    return {"index": index, "actual_class": actual, "features": features}
