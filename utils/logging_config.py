"""Structured logging shared by every block. No raw print statements."""
from __future__ import annotations
import logging, sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg
