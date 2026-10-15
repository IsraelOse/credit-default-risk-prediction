"""Central configuration for the Credit Default Risk pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

SEED: int = 42
TEST_SIZE: float = 0.20

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "datasets"
OUTPUTS: Path = ROOT / "outputs"
APPLICATIONS_CSV: Path = DATA_DIR / "creditrisk_applications_final.csv"
INQUIRIES_CSV: Path = DATA_DIR / "creditrisk_inquiries_final.csv"

TARGET: str = "target_default"
DEFAULT_LABEL: str = "Charged Off"   # outcome_status value that marks a default

NUMERIC_FEATURES: List[str] = [
    "requested_amount_usd", "monthly_installment_usd", "interest_rate",
    "repayment_term", "annual_income_usd", "debt_to_income_ratio",
    "credit_score_low", "credit_score_high", "open_credit_lines",
    "public_records_count", "revolving_utilization", "employment_tenure_years",
]
CATEGORICAL_FEATURES: List[str] = [
    "risk_grade_band", "housing_status", "income_verification_status",
    "loan_purpose", "region_code",
]

MODEL_PARAMS: Dict[str, dict] = {
    "Logistic Regression": {"max_iter": 1000, "class_weight": "balanced"},
    "Random Forest": {"n_estimators": 80, "max_depth": 15, "min_samples_leaf": 5,
                      "class_weight": "balanced", "n_jobs": -1},
}
MODEL_ORDER: List[str] = ["Logistic Regression", "Random Forest"]
