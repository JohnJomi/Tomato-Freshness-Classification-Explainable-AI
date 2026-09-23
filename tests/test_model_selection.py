"""Regression tests: XAI model selection must use training-set CV scores only,
never held-out test-set scores."""

from src.feature_importance import select_model_for_xai


def _result(cv_f1, cv_acc, test_f1, test_acc):
    return {
        "cv_f1_weighted_mean": cv_f1,
        "cv_accuracy_mean": cv_acc,
        "metrics": {"test_f1_weighted": test_f1, "test_accuracy": test_acc},
    }


def test_selects_best_cv_f1_even_when_test_f1_disagrees():
    results = {
        "Model A": _result(cv_f1=0.90, cv_acc=0.90, test_f1=0.80, test_acc=0.80),
        "Model B": _result(cv_f1=0.85, cv_acc=0.85, test_f1=0.99, test_acc=0.99),
    }
    assert select_model_for_xai(results) == "Model A"


def test_cv_accuracy_breaks_ties_on_cv_f1():
    results = {
        "Model A": _result(cv_f1=0.90, cv_acc=0.88, test_f1=0.99, test_acc=0.99),
        "Model B": _result(cv_f1=0.90, cv_acc=0.92, test_f1=0.70, test_acc=0.70),
    }
    assert select_model_for_xai(results) == "Model B"


def test_works_on_cv_only_results_before_test_evaluation():
    # Selection runs before the test set is touched, so no "metrics" key exists yet.
    cv_only = {
        "Random Forest": {"cv_f1_weighted_mean": 0.95, "cv_accuracy_mean": 0.95},
        "SVM": {"cv_f1_weighted_mean": 0.97, "cv_accuracy_mean": 0.96},
        "XGBoost": {"cv_f1_weighted_mean": 0.96, "cv_accuracy_mean": 0.97},
    }
    assert select_model_for_xai(cv_only) == "SVM"
