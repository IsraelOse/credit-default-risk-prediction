"""Data exporter block: persist metrics and feature importances."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from utils.config import OUTPUTS
from utils.logging_config import get_logger

logger = get_logger(__name__)

def export_metrics(metrics: pd.DataFrame, feature_importance: pd.DataFrame) -> Path:
    """Write model metrics (CSV+JSON) and the top feature importances (CSV)."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    (OUTPUTS / "model_comparison.json").write_text(
        json.dumps(metrics.to_dict(orient="records"), indent=2))
    feature_importance.head(20).to_csv(OUTPUTS / "rf_feature_importance.csv", index=False)
    logger.info("Wrote metrics + feature importances to %s", OUTPUTS)
    return OUTPUTS / "model_comparison.csv"
