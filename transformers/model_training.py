"""
Transformer block: train and evaluate the classifiers.

Each model is preprocessing + estimator in one Pipeline, fitted on the training
split and scored on the held-out test set. Returns fitted pipelines, a metrics
table, and Random-Forest feature importances.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from utils.config import MODEL_PARAMS, MODEL_ORDER, SEED
from utils.logging_config import get_logger
from transformers.preprocessing import build_preprocessor

logger = get_logger(__name__)


def _estimator(name: str, seed: int):
    params = MODEL_PARAMS[name]
    if name == "Logistic Regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=seed, **params)
    raise ValueError(name)


def train_and_evaluate(X_train, X_test, y_train, y_test, seed: int = SEED
                       ) -> Tuple[Dict[str, Pipeline], pd.DataFrame, pd.DataFrame]:
    """Fit each model; return (fitted, metrics_table, rf_feature_importance)."""
    fitted: Dict[str, Pipeline] = {}
    rows = []
    for name in MODEL_ORDER:
        clf = Pipeline([("preprocessor", build_preprocessor()),
                        ("model", _estimator(name, seed))])
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, pred, zero_division=0), 4),
            "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
        })
        fitted[name] = clf
        logger.info("%-20s ROC-AUC %.4f  recall %.4f", name, rows[-1]["ROC_AUC"], rows[-1]["Recall"])

    metrics = pd.DataFrame(rows)
    rf = fitted["Random Forest"]
    names = rf.named_steps["preprocessor"].get_feature_names_out()
    importances = rf.named_steps["model"].feature_importances_
    fi = (pd.DataFrame({"feature": names, "importance": importances})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    return fitted, metrics, fi
