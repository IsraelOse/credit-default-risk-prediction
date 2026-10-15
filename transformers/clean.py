"""
Transformer block: clean the applications table and derive the target.

Pure functions: parse text-encoded numerics, derive the binary default target,
and drop duplicates. Leakage note: ``outcome_status`` (and its derivatives such
as last_payment) describe the loan outcome, so only pre-decision features are
used downstream; the target is derived from outcome_status and that column is
not a model feature.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from utils.config import DEFAULT_LABEL, TARGET
from utils.logging_config import get_logger

logger = get_logger(__name__)


def clean_tenure(value) -> float:
    """Convert employment-tenure text ('< 1 year', '10+ years') to numeric years."""
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    if "10+" in v:
        return 10.0
    if "< 1" in v:
        return 0.0
    m = re.search(r"\d+", v)
    return float(m.group()) if m else np.nan


def clean_applications(apps: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned applications frame with parsed numerics and the target."""
    df = apps.copy()
    df[TARGET] = np.where(df["outcome_status"] == DEFAULT_LABEL, 1, 0)
    df["interest_rate"] = pd.to_numeric(df["interest_rate_text"], errors="coerce")
    df["repayment_term"] = pd.to_numeric(
        df["repayment_term_text"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["revolving_utilization"] = pd.to_numeric(df["revolving_utilization_text"], errors="coerce")
    df["debt_to_income_ratio"] = pd.to_numeric(
        df["debt_to_income_ratio"].astype(str).str.replace("%", "", regex=False), errors="coerce")
    df["employment_tenure_years"] = df["employment_tenure_text"].apply(clean_tenure)
    df = df.drop_duplicates()
    logger.info("Cleaned applications: %d rows, default rate %.2f%%",
                len(df), 100 * df[TARGET].mean())
    return df


def clean_inquiries(inq: pd.DataFrame) -> pd.DataFrame:
    """Clean the inquiries table (secondary EDA source; not joined to applications)."""
    df = inq.copy()
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df["debt_to_income"] = pd.to_numeric(
        df["debt_to_income_text"].astype(str).str.replace("%", "", regex=False), errors="coerce")
    df["employment_tenure_years"] = df["employment_tenure_text"].apply(clean_tenure)
    return df.drop_duplicates()
