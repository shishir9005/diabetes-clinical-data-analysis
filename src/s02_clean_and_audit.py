"""Day 1, steps 4b-5: deterministic cleaning + duplicate / conflicting-profile audit.

Full-file access is permitted: auditing and profile identification happen
before the split is established. No rows are removed here.

History: on 7 Nov 2026 only the section 5 report text below was updated to the approved duplicate
policy (primary analysis: all 520 records; sensitivity analysis: 251 unique questionnaire profiles).
No logic changed and the script was not re-run. Pre-edit copy:
archive/day1_final_pass_pre_edit_2026-11-07/src/s02_clean_and_audit.py

Run with:  <env python> -I -B src/s02_clean_and_audit.py
Writes:    data/processed/cleaned_all_rows.csv
           reports/02_duplicate_audit.md, reports/02_duplicate_audit.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

from prep import (ALL_COLUMNS, CLEANED_ALL, FEATURE_COLUMNS, RAW_TO_CLEAN, REPORTS_DIR,  # noqa: E402
                  SYMPTOM_COLUMNS, add_identifiers, md_table, read_raw_as_text, recode)


def pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 2) if b else float("nan")


def class_split(frame: pd.DataFrame) -> dict:
    n = len(frame)
    pos = int(frame["label"].sum())
    return {"rows": n, "positive": pos, "negative": n - pos, "pct_positive": pct(pos, n)}


def conflict_stats(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = frame.groupby("profile_id")["label"].agg(n_rows="size", n_positive="sum")
    g["n_negative"] = g["n_rows"] - g["n_positive"]
    conflicting = g[(g["n_positive"] > 0) & (g["n_negative"] > 0)].copy()
    conflicting["minority_rows"] = conflicting[["n_positive", "n_negative"]].min(axis=1)
    n = len(frame)
    min_errors = int(conflicting["minority_rows"].sum())
    stats = {
        "rows_in_view": n,
        "unique_profiles": int(len(g)),
        "profiles_with_more_than_one_row": int((g["n_rows"] > 1).sum()),
        "conflicting_profiles": int(len(conflicting)),
        "rows_in_conflicting_profiles": int(conflicting["n_rows"].sum()),
        "pct_rows_in_conflicting_profiles": pct(int(conflicting["n_rows"].sum()), n),
        "positive_rows_in_conflicting_profiles": int(conflicting["n_positive"].sum()),
        "negative_rows_in_conflicting_profiles": int(conflicting["n_negative"].sum()),
        "minimum_rows_any_deterministic_classifier_must_misclassify": min_errors,
        "theoretical_max_accuracy_from_conflicts_only": round(1 - min_errors / n, 6) if n else float("nan"),
    }
    return conflicting.reset_index(), stats


# --- Clean -----------------------------------------------------------------
raw = read_raw_as_text()
clean = recode(raw)
df = add_identifiers(clean)
N = len(df)
assert N == len(raw) == 520, "row count changed during cleaning"

# --- Exact duplicate records (all 17 columns) ------------------------------
rec_sizes = df.groupby("record_id").size()
dup_groups = rec_sizes[rec_sizes > 1]
rows_in_dup_groups = int(dup_groups.sum())
n_unique_records = int(len(rec_sizes))
redundant = N - n_unique_records
dedup = df[df["is_first_copy"] == 1]
assert len(dedup) == n_unique_records

exact = {
    "total_rows": N,
    "unique_records": n_unique_records,
    "redundant_copies": redundant,
    "pct_redundant_copies": pct(redundant, N),
    "duplicate_groups": int(len(dup_groups)),
    "rows_in_duplicate_groups": rows_in_dup_groups,
    "pct_rows_in_duplicate_groups": pct(rows_in_dup_groups, N),
    "group_size_distribution": {int(k): int(v) for k, v in dup_groups.value_counts().sort_index().items()},
    "class_split_rows_in_duplicate_groups": class_split(df[df["record_id"].isin(dup_groups.index)]),
    "class_split_redundant_copies": class_split(df[df["is_first_copy"] == 0]),
    "class_split_all_rows": class_split(df),
    "class_split_deduplicated": class_split(dedup),
}

# --- Feature profiles (16 features) and conflicting labels -----------------
conf_all, conf_stats_all = conflict_stats(df)
conf_dedup, conf_stats_dedup = conflict_stats(dedup)

# --- SQL cross-check (temporary in-memory database, nothing written) -------
con = sqlite3.connect(":memory:")
df[ALL_COLUMNS].to_sql("clean", con, index=False)
raw_renamed = raw.rename(columns=RAW_TO_CLEAN)          # literal strings, snake_case names
raw_renamed.to_sql("raw_text", con, index=False)
all_cols = ", ".join(ALL_COLUMNS)
feat_cols = ", ".join(FEATURE_COLUMNS)


def sql_dups(table: str) -> tuple:
    return con.execute(
        f"SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(n-1),0) FROM "
        f"(SELECT COUNT(*) AS n FROM {table} GROUP BY {all_cols} HAVING COUNT(*) > 1)").fetchone()


def sql_conflicts(table: str, pos_expr: str) -> tuple:
    return con.execute(
        f"SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(MIN(npos, n-npos)),0) FROM "
        f"(SELECT COUNT(*) AS n, SUM({pos_expr}) AS npos FROM {table} GROUP BY {feat_cols} "
        f"HAVING COUNT(DISTINCT label) > 1)").fetchone()


sql = {
    "clean_exact_dups(groups, rows_in_groups, redundant)": sql_dups("clean"),
    "raw_text_exact_dups(groups, rows_in_groups, redundant)": sql_dups("raw_text"),
    "clean_conflicts(profiles, rows, minority_rows)": sql_conflicts("clean", "label"),
    "raw_text_conflicts(profiles, rows, minority_rows)": sql_conflicts("raw_text", "CASE WHEN label='Positive' THEN 1 ELSE 0 END"),
}
con.close()
pandas_dups = (exact["duplicate_groups"], exact["rows_in_duplicate_groups"], exact["redundant_copies"])
pandas_conf = (conf_stats_all["conflicting_profiles"], conf_stats_all["rows_in_conflicting_profiles"],
               conf_stats_all["minimum_rows_any_deterministic_classifier_must_misclassify"])
sql_agree = (tuple(sql["clean_exact_dups(groups, rows_in_groups, redundant)"]) == pandas_dups
             and tuple(sql["raw_text_exact_dups(groups, rows_in_groups, redundant)"]) == pandas_dups
             and tuple(sql["clean_conflicts(profiles, rows, minority_rows)"]) == pandas_conf
             and tuple(sql["raw_text_conflicts(profiles, rows, minority_rows)"]) == pandas_conf)

# --- Descriptive view of the largest duplicate groups (no identity claims) -
df["n_symptoms_yes"] = df[SYMPTOM_COLUMNS].sum(axis=1)
top = (df[df["record_id"].isin(dup_groups.index)]
       .drop_duplicates("record_id")
       .assign(copies=lambda d: d["record_id"].map(rec_sizes))
       .sort_values(["copies", "record_id"], ascending=[False, True])
       .head(10)[["record_id", "copies", "age", "gender_male", "n_symptoms_yes", "label"]])
df = df.drop(columns="n_symptoms_yes")

# --- Write cleaned file ------------------------------------------------------
CLEANED_ALL.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEANED_ALL, index=False)

audit = {
    "exact_duplicates": exact,
    "conflicting_profiles_all_rows": conf_stats_all,
    "conflicting_profiles_deduplicated": conf_stats_dedup,
    "conflicting_profile_detail_all_rows": conf_all.to_dict(orient="records"),
    "sql_cross_check": {k: list(v) for k, v in sql.items()},
    "sql_matches_pandas": sql_agree,
    "confirmed_repeated_patients": "cannot be determined: no patient identifier, visit date or record identifier exists",
}
(REPORTS_DIR / "02_duplicate_audit.json").write_text(json.dumps(audit, indent=2))

# --- Markdown report ---------------------------------------------------------
e = exact
L = [
    "# 02 · Duplicate and conflicting-profile audit",
    "",
    "Scope: all 520 rows of the cleaned data. Full-file access is permitted here because auditing",
    "and profile identification happen before the split is established. **No rows were removed.**",
    "",
    "Definitions:",
    "",
    "- **Identical questionnaire record**: two or more rows with the same values in all 16 features and the label.",
    "- **Feature profile**: the 16 feature values. Rows with the same profile are always kept on the same side of any split.",
    "- **Conflicting profile**: one feature profile that appears with both labels.",
    "",
    "## 1. Identical questionnaire records versus repeated patients",
    "",
    "The file has no patient identifier, visit date or record identifier. Identical rows therefore",
    "cannot be attributed to the same person. Each one could be the same patient entered more than",
    "once, a copy made while compiling the data, or different people who gave the same answers.",
    "",
    "| Category | Count |",
    "|---|---|",
    f"| Identical questionnaire records (rows in duplicate groups) | {e['rows_in_duplicate_groups']} |",
    "| Confirmed repeated patients | **0 confirmed.** Cannot be determined from this data. |",
    "",
    "## 2. Exact duplicate records (all 17 columns identical)",
    "",
    "| Measure | Count | % of 520 |",
    "|---|---|---|",
    f"| Total rows | {e['total_rows']} | 100.00 |",
    f"| Unique records | {e['unique_records']} | {pct(e['unique_records'], N)} |",
    f"| Redundant copies (rows − unique records) | {e['redundant_copies']} | {e['pct_redundant_copies']} |",
    f"| Duplicate groups (records appearing ≥ 2 times) | {e['duplicate_groups']} | – |",
    f"| Rows belonging to a duplicate group (incl. first copy) | {e['rows_in_duplicate_groups']} | {e['pct_rows_in_duplicate_groups']} |",
    "",
    "Group-size distribution (copies per duplicate group → number of groups):",
    "",
    md_table(pd.DataFrame(list(e["group_size_distribution"].items()), columns=["copies_in_group", "number_of_groups"])),
    "",
    "Class split:",
    "",
    md_table(pd.DataFrame([
        {"subset": "all rows", **e["class_split_all_rows"]},
        {"subset": "deduplicated (one copy per record)", **e["class_split_deduplicated"]},
        {"subset": "rows in duplicate groups", **e["class_split_rows_in_duplicate_groups"]},
        {"subset": "redundant copies only", **e["class_split_redundant_copies"]},
    ])),
    "",
    "The ten largest duplicate groups are shown below. `n_symptoms_yes` counts the 14 symptoms",
    "answered Yes. This is descriptive only and does not identify anyone.",
    "",
    md_table(top),
    "",
    "## 3. Conflicting profiles (same 16 features, different labels)",
    "",
    md_table(pd.DataFrame([
        {"view": "all rows", **{k: v for k, v in conf_stats_all.items()}},
        {"view": "deduplicated", **{k: v for k, v in conf_stats_dedup.items()}},
    ]).T.reset_index().rename(columns={"index": "measure", 0: "all rows", 1: "deduplicated"}).iloc[1:]),
    "",
    "Conflicting profiles in detail (all-rows view):",
    "",
    md_table(conf_all) if len(conf_all) else "_None found._",
    "",
    "### Accuracy limit attributable to conflicting profiles",
    "",
    "For identical feature profiles, a deterministic classifier produces the same prediction for every",
    "row. It must therefore misclassify at least the number of rows belonging to the minority label",
    "within each conflicting profile. Which label is misclassified depends on the model's prediction.",
    "",
    f"- All rows: at least **{conf_stats_all['minimum_rows_any_deterministic_classifier_must_misclassify']}** of {N} rows must be misclassified, "
    f"so the theoretical maximum accuracy from this source alone is **{conf_stats_all['theoretical_max_accuracy_from_conflicts_only']:.4f}**.",
    f"- Deduplicated: at least **{conf_stats_dedup['minimum_rows_any_deterministic_classifier_must_misclassify']}** of {len(dedup)} rows, "
    f"giving a ceiling of **{conf_stats_dedup['theoretical_max_accuracy_from_conflicts_only']:.4f}**.",
    "",
    "These ceilings are properties of the data. They are **not observed model results**, and they",
    "ignore every other source of error. No model has been trained.",
    "",
    "## 4. SQL cross-check",
    "",
    "The counts were recomputed with SQL `GROUP BY … HAVING` in a temporary in-memory SQLite database,",
    "once on the cleaned values and once on the literal raw strings.",
    "",
    md_table(pd.DataFrame([{"query": k, "result": str(tuple(v))} for k, v in sql.items()])),
    "",
    f"Pandas result: duplicates {pandas_dups}, conflicts {pandas_conf}. "
    f"**SQL matches pandas: {sql_agree}.**",
    "",
    "## 5. Duplicate policy (approved 7 Nov 2026; see docs/PROJECT_PLAN.md §3.5)",
    "",
    "This section was corrected on 7 Nov 2026. The 6 Nov version proposed the opposite assignment (deduplicated data as the primary analysis). It is archived in `archive/day1_pre_correction_2026-11-07/reports/02_duplicate_audit.md`. The audit counts above are unchanged.",
    "",
    # Facts for the hash-pinned download: no conflicting profiles (section 3 above), so the unique records
    # and the unique profiles are the same set. No row is removed from the primary analysis.
    f"- **Primary analysis:** all {N} original records are retained.",
    f"- **Sensitivity analysis:** the {e['unique_records']} unique questionnaire profiles, one record per profile. "
    f"No profile is conflicting, so the {e['unique_records']} unique records and the {conf_stats_all['unique_profiles']} unique profiles are the same set.",
    "- **Both analyses:** use the same frozen split by feature profile, and keep conflicting profiles in (none exist).",
    "- No percentage threshold is used to decide removal. The earlier planning rule that would have made deduplicated data primary when duplicates exceeded 5% is withdrawn.",
]
(REPORTS_DIR / "02_duplicate_audit.md").write_text("\n".join(L) + "\n")
print(json.dumps({k: v for k, v in audit.items() if k != "conflicting_profile_detail_all_rows"}, indent=2))
print("conflicting profile detail:\n", conf_all.to_string(index=False))
print("largest duplicate groups:\n", top.to_string(index=False))
sys.exit(0 if sql_agree else 1)
