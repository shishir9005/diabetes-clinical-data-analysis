"""Day 1, step 4a: validate the original CSV against what is actually in the file.

Reads the raw file only (read-only; never modified). Full-file reading is
permitted here because the split has not been established yet.

Run with:  <env python> -I -B src/s01_validate_raw.py
Writes:    reports/01_data_validation.md, reports/01_data_validation.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # needed because -I omits the script dir
import pandas as pd  # noqa: E402

from prep import EXPECTED_RAW_COLUMNS, RAW_CSV, REPORTS_DIR, md_table, read_raw_as_text, sha256_file  # noqa: E402

RECORDED_CSV_SHA256 = "7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137"  # docs/DATA_SOURCE.md
DOCUMENTED = {"rows": 520, "features": 16, "columns_incl_class": 17, "age_range": (20, 65)}
# UCI variable names. The first 10 are from the rendered variables table; the remaining 7
# follow the order of the page's "Additional Variable Information" list.
UCI_VARIABLE_NAMES = [
    "age", "gender", "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
    "genital_thrush", "visual_blurring", "itching", "irritability", "delayed_healing",
    "partial_paresis", "muscle_stiffness", "alopecia", "obesity", "class",
]
MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none", "-", "--", "missing", "unknown"}

result: dict = {"checks": {}}
fail: list[str] = []


def record(name: str, ok: bool, detail):
    result["checks"][name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        fail.append(name)


# 1. Hash before processing -------------------------------------------------
actual_hash = sha256_file(RAW_CSV)
record("sha256_matches_download_record", actual_hash == RECORDED_CSV_SHA256, actual_hash)

# 2. Byte/line level ----------------------------------------------------------
raw_bytes = RAW_CSV.read_bytes()
text = raw_bytes.decode("utf-8")  # raises if not valid UTF-8
record("utf8_decodes", True, "decoded as UTF-8")
record("no_byte_order_mark", not raw_bytes.startswith(b"\xef\xbb\xbf"), "BOM absent" if not raw_bytes.startswith(b"\xef\xbb\xbf") else "BOM present")
crlf = raw_bytes.count(b"\r\n")
lf_total = raw_bytes.count(b"\n")
record("line_endings", True, {"crlf_lines": crlf, "lf_total": lf_total, "style": "CRLF" if crlf == lf_total else "mixed/LF"})
non_ascii = sum(1 for ch in text if ord(ch) > 127)
record("no_non_ascii_characters", non_ascii == 0, non_ascii)

with open(RAW_CSV, newline="", encoding="utf-8") as fh:
    rows = list(csv.reader(fh))
field_counts = pd.Series([len(r) for r in rows]).value_counts().to_dict()
record("every_line_has_same_field_count", len(field_counts) == 1, {int(k): int(v) for k, v in field_counts.items()})

# 3. Dimensions and header ----------------------------------------------------
df = read_raw_as_text()
n_rows, n_cols = df.shape
result["shape"] = {"rows": n_rows, "columns": n_cols}
record("rows_match_documentation", n_rows == DOCUMENTED["rows"], {"observed": n_rows, "documented": DOCUMENTED["rows"]})
record("columns_match_documentation", n_cols == DOCUMENTED["columns_incl_class"], {"observed": n_cols, "documented": DOCUMENTED["columns_incl_class"]})
record("pandas_rows_equal_csv_lines_minus_header", n_rows == len(rows) - 1, {"pandas": n_rows, "csv_lines_minus_header": len(rows) - 1})
header = list(df.columns)
record("header_exact_match_expected", header == EXPECTED_RAW_COLUMNS, header)
ws_cols = [c for c in header if c != c.strip()]
record("no_whitespace_padding_in_column_names", not ws_cols, ws_cols)
normalised = [c.strip().lower().replace(" ", "_") for c in header]
record("header_matches_uci_variable_names_after_normalising", normalised == UCI_VARIABLE_NAMES,
       {"normalised_header": normalised, "uci": UCI_VARIABLE_NAMES})

# 4. Cell-level: whitespace, missing tokens, case variants -------------------
ws_cells = {c: int((df[c] != df[c].str.strip()).sum()) for c in header}
record("no_whitespace_padding_in_cells", sum(ws_cells.values()) == 0, {k: v for k, v in ws_cells.items() if v})
missing = {c: int(df[c].str.strip().str.lower().isin(MISSING_TOKENS).sum()) for c in header}
record("no_missing_or_placeholder_values", sum(missing.values()) == 0, {"per_column": missing, "tokens_searched": sorted(MISSING_TOKENS)})
case_variants = {}
for c in header[1:]:
    vals = df[c].unique().tolist()
    if len({v.lower() for v in vals}) != len(vals):
        case_variants[c] = vals
record("no_case_variants", not case_variants, case_variants)

# 5. Distinct values per column ---------------------------------------------
distinct = {}
for c in header[1:]:
    distinct[c] = {k: int(v) for k, v in df[c].value_counts().sort_index().items()}
result["distinct_values"] = distinct
expected_sets = {"Gender": {"Male", "Female"}, "class": {"Positive", "Negative"}}
unexpected = {}
for c in header[1:]:
    exp = expected_sets.get(c, {"Yes", "No"})
    extra = sorted(set(distinct[c]) - exp)
    if extra:
        unexpected[c] = extra
record("categorical_values_within_expected_sets", not unexpected, unexpected)

# 6. Age ----------------------------------------------------------------------
age_num = pd.to_numeric(df["Age"], errors="coerce")
non_numeric = int(age_num.isna().sum())
record("age_all_numeric", non_numeric == 0, non_numeric)
non_integer = int((age_num.dropna() != age_num.dropna().round()).sum())
record("age_all_integer", non_integer == 0, non_integer)
lo, hi = DOCUMENTED["age_range"]
outside = int(((age_num < lo) | (age_num > hi)).sum())
age_summary = {
    "min": int(age_num.min()), "max": int(age_num.max()), "mean": round(float(age_num.mean()), 2),
    "median": float(age_num.median()), "distinct_values": int(age_num.nunique()),
    "rows_outside_documented_20_65": outside,
    "rows_below_20": int((age_num < lo).sum()), "rows_above_65": int((age_num > hi).sum()),
}
result["age_summary"] = age_summary
# Informational: outside-documented-range ages are reported, not treated as errors or removed.
record("age_within_documented_range_20_65 (informational)", outside == 0, age_summary)

result["hard_failures"] = [f for f in fail if "informational" not in f]
result["informational_mismatches"] = [f for f in fail if "informational" in f]

# 7. Write reports --------------------------------------------------------------
(REPORTS_DIR / "01_data_validation.json").write_text(json.dumps(result, indent=2))

lines = [
    "# 01 · Data validation report (original file)",
    "",
    "Source file: `data/raw/extracted/diabetes_data_upload.csv`, opened read-only.",
    "Every cell was read as its literal text (no automatic NA conversion).",
    "Full-file access is permitted here because the split has not been established yet.",
    "",
    "## Summary",
    "",
    f"- Observed shape: **{n_rows} rows × {n_cols} columns** (16 features + `class`).",
    f"- SHA-256 before processing: `{actual_hash}` "
    f"({'matches' if actual_hash == RECORDED_CSV_SHA256 else 'DOES NOT match'} the download record).",
    f"- Hard failures: **{len(result['hard_failures'])}** {result['hard_failures'] or ''}",
    f"- Informational mismatches with documentation: {result['informational_mismatches'] or 'none'}",
    "",
    "## Checks",
    "",
    md_table(pd.DataFrame(
        [{"check": k, "result": "PASS" if v["ok"] else ("NOTE" if "informational" in k else "FAIL"),
          "detail": json.dumps(v["detail"])[:160]} for k, v in result["checks"].items()]
    )),
    "",
    "## Age",
    "",
    md_table(pd.DataFrame([age_summary])),
    "",
    "## Distinct values per column (counts over all 520 rows)",
    "",
    md_table(pd.DataFrame(
        [{"column": c, "values (count)": ", ".join(f"{k} ({v})" for k, v in d.items())} for c, d in distinct.items()]
    )),
    "",
    "## Missing values",
    "",
    "Searched every cell for empty strings and these placeholder tokens (case-insensitive): "
    + ", ".join(f"`{t}`" if t else "`<empty>`" for t in sorted(MISSING_TOKENS)) + ".",
    f"Total matches: **{sum(missing.values())}**.",
    "",
    "UCI's summary says the dataset has missing values, but its per-variable table says 'no'. "
    "This report records what the file actually contains.",
]
(REPORTS_DIR / "01_data_validation.md").write_text("\n".join(lines) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "distinct_values"}, indent=2))
sys.exit(1 if result["hard_failures"] else 0)
