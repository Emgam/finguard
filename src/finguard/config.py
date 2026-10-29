"""Project-wide paths and constants.

Paths can be overridden with environment variables so the same code runs on a laptop,
in CI and in Docker:

    FINGUARD_RAW_DIR      directory holding the five Kaggle CSV files
    FINGUARD_INTERIM_DIR  directory for the cleaned, joined tables
    FINGUARD_REPORTS_DIR  directory for figures and tables
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = Path(os.environ.get("FINGUARD_RAW_DIR", ROOT / "data" / "raw"))
INTERIM_DIR = Path(os.environ.get("FINGUARD_INTERIM_DIR", ROOT / "data" / "interim"))
REPORTS_DIR = Path(os.environ.get("FINGUARD_REPORTS_DIR", ROOT / "reports"))
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

RAW_FILES = {
    "train_transaction": "train_transaction.csv",
    "train_identity": "train_identity.csv",
    "test_transaction": "test_transaction.csv",
    "test_identity": "test_identity.csv",
    "sample_submission": "sample_submission.csv",
}

# MD5 checksums of the official Kaggle release (verified on 2026-10-07).
RAW_MD5 = {
    "train_transaction": "58b4038d8715f5e11007b826bef00ce7",
    "train_identity": "8487db5001c8bad139f3318d5d3db416",
    "test_transaction": "7ea6862ef4e078efb309e19fa49178fd",
    "test_identity": "54ae784303f82e5cadeb2d899b05c6a8",
    "sample_submission": "a4dece4fe5e7a398319009b94f7b34a5",
}

KEY = "TransactionID"
TARGET = "isFraud"
TIME = "TransactionDT"
SECONDS_PER_DAY = 86_400

# Time split from Project Report v3.0, section 6.4 (relative day index, 0-based).
# train: days 0-120, validation: days 121-151, test: days 152-181 of the labelled file.
SPLIT_DAYS = {"train": (0, 120), "valid": (121, 151), "test": (152, 181)}

RANDOM_SEED = 42
