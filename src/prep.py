"""Shared constants and helpers for the Day 1 scripts.

Deliberately small: project paths, the raw-to-clean column mapping,
strict recoding, identifiers and hashing. Nothing in here learns anything
from the data, so it is safe to apply before the train/test split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ZIP = PROJECT_ROOT / "data" / "raw" / "archive" / "early_stage_diabetes_risk_prediction_uci529.zip"
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "extracted" / "diabetes_data_upload.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_DIR = PROJECT_ROOT / "sql"

CLEANED_ALL = PROCESSED_DIR / "cleaned_all_rows.csv"
TRAIN_ROWS = PROCESSED_DIR / "train_rows.csv"

# Raw header as observed in the downloaded file -> clean snake_case name.
# s01_validate_raw.py checks the real header against this list exactly.
RAW_TO_CLEAN = {
    "Age": "age",
    "Gender": "gender_male",
    "Polyuria": "polyuria",
    "Polydipsia": "polydipsia",
    "sudden weight loss": "sudden_weight_loss",
    "weakness": "weakness",
    "Polyphagia": "polyphagia",
    "Genital thrush": "genital_thrush",
    "visual blurring": "visual_blurring",
    "Itching": "itching",
    "Irritability": "irritability",
    "delayed healing": "delayed_healing",
    "partial paresis": "partial_paresis",
    "muscle stiffness": "muscle_stiffness",
    "Alopecia": "alopecia",
    "Obesity": "obesity",
    "class": "label",
}
EXPECTED_RAW_COLUMNS = list(RAW_TO_CLEAN)

SYMPTOM_COLUMNS = [
    "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
    "genital_thrush", "visual_blurring", "itching", "irritability",
    "delayed_healing", "partial_paresis", "muscle_stiffness", "alopecia", "obesity",
]
FEATURE_COLUMNS = ["age", "gender_male"] + SYMPTOM_COLUMNS          # 16 features
ALL_COLUMNS = FEATURE_COLUMNS + ["label"]                            # 17 columns

YES_NO = {"Yes": 1, "No": 0}
GENDER = {"Male": 1, "Female": 0}
LABEL = {"Positive": 1, "Negative": 0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw_as_text(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read every cell as the literal string in the file (no NA guessing)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _strict_map(series: pd.Series, mapping: dict, column: str) -> pd.Series:
    unexpected = sorted(set(series.unique()) - set(mapping))
    if unexpected:
        raise ValueError(f"Column {column!r} has unexpected values {unexpected}; refusing to guess.")
    return series.map(mapping).astype("int64")


def recode(raw: pd.DataFrame) -> pd.DataFrame:
    """Deterministic recoding. Raises instead of guessing on any unexpected value."""
    if list(raw.columns) != EXPECTED_RAW_COLUMNS:
        raise ValueError("Raw header does not match the expected column list exactly.")
    out = pd.DataFrame(index=raw.index)
    age = pd.to_numeric(raw["Age"], errors="raise")
    if not (age == age.round()).all():
        raise ValueError("Age contains non-integer values.")
    out["age"] = age.astype("int64")
    out["gender_male"] = _strict_map(raw["Gender"], GENDER, "Gender")
    for raw_name, clean_name in RAW_TO_CLEAN.items():
        if clean_name in SYMPTOM_COLUMNS:
            out[clean_name] = _strict_map(raw[raw_name], YES_NO, raw_name)
    out["label"] = _strict_map(raw["class"], LABEL, "class")
    return out[ALL_COLUMNS]


def add_identifiers(clean: pd.DataFrame) -> pd.DataFrame:
    """row_id = file order (1-based); record_id = identical on all 17 columns;
    profile_id = identical on the 16 features; is_first_copy marks the first
    occurrence (in file order) of each record_id (the deduplicated view)."""
    df = clean.copy()
    df.insert(0, "row_id", range(1, len(df) + 1))
    df.insert(1, "record_id", df.groupby(ALL_COLUMNS, sort=True).ngroup() + 1)
    df.insert(2, "profile_id", df.groupby(FEATURE_COLUMNS, sort=True).ngroup() + 1)
    df.insert(3, "is_first_copy", (~df.duplicated(subset=ALL_COLUMNS, keep="first")).astype("int64"))
    return df


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal Markdown table writer (avoids an extra 'tabulate' dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)
