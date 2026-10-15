"""
Pipeline orchestration: credit_default.

data_loaders -> transformers (clean -> split -> train/eval) -> data_exporters
Run from the repo root:  python -m pipelines.credit_default.run
"""
from __future__ import annotations
import pandas as pd
from data_loaders.credit_loader import load_data
from transformers.clean import clean_applications
from transformers.split import split_data
from transformers.model_training import train_and_evaluate
from data_exporters.local_exporter import export_metrics
from utils.logging_config import get_logger

logger = get_logger("pipelines.credit_default")


def run() -> pd.DataFrame:
    """Execute the full pipeline and return the metrics table."""
    logger.info("=== credit_default pipeline start ===")
    apps, _inq = load_data()
    clean = clean_applications(apps)
    X_train, X_test, y_train, y_test = split_data(clean)
    _, metrics, fi = train_and_evaluate(X_train, X_test, y_train, y_test)
    export_metrics(metrics, fi)
    best = metrics.sort_values("ROC_AUC", ascending=False).iloc[0]
    logger.info("Best by ROC-AUC: %s (%.4f). Top feature: %s",
                best["Model"], best["ROC_AUC"], fi.iloc[0]["feature"])
    logger.info("=== pipeline complete ===")
    return metrics


if __name__ == "__main__":
    print(run().to_string(index=False))
