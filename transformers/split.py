"""Transformer block: stratified train/test split."""
from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, TEST_SIZE, SEED
from utils.logging_config import get_logger

logger = get_logger(__name__)

def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test using the configured feature set."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    logger.info("Split: train=%d test=%d (default rate train %.3f / test %.3f)",
                len(X_train), len(X_test), y_train.mean(), y_test.mean())
    return X_train, X_test, y_train, y_test
