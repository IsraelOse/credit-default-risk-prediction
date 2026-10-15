"""Data loader block: read the applications and inquiries CSVs."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
from utils.config import APPLICATIONS_CSV, INQUIRIES_CSV
from utils.logging_config import get_logger

logger = get_logger(__name__)

def load_data(apps_path: Path = APPLICATIONS_CSV,
              inq_path: Path = INQUIRIES_CSV) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (applications, inquiries) DataFrames."""
    apps = pd.read_csv(apps_path)
    inq = pd.read_csv(inq_path)
    logger.info("Loaded applications %s and inquiries %s", apps.shape, inq.shape)
    return apps, inq
