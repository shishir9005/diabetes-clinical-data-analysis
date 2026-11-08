"""Day 1, step 7: SQLite queries on training rows only.

Reads ONLY data/processed/train_rows.csv (written by s03 from the training partition).
It never opens the full cleaned file or the split assignment.

Run with:  <env python> -I -B src/s04_sql_train.py
Writes:    data/processed/train_only.sqlite, reports/04_sql_training.md
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

from prep import PROCESSED_DIR, REPORTS_DIR, SQL_DIR, SYMPTOM_COLUMNS, TRAIN_ROWS, md_table  # noqa: E402

DB = PROCESSED_DIR / "train_only.sqlite"
if DB.exists():
    sys.exit(f"{DB} already exists; refusing to overwrite.")

train = pd.read_csv(TRAIN_ROWS)
con = sqlite3.connect(DB)
train.to_sql("train_rows", con, index=False)

blocks: list[tuple[str, str]] = []
name, buf = None, []
for line in (SQL_DIR / "train_queries.sql").read_text().splitlines():
    if line.startswith("-- name:"):
        if name:
            blocks.append((name, "\n".join(buf)))
        name, buf = line.split(":", 1)[1].strip(), []
    elif name:
        buf.append(line)
if name:
    blocks.append((name, "\n".join(buf)))

results: dict[str, pd.DataFrame] = {}
for name, sql in blocks:
    if name == "setup_views":
        con.executescript(sql)
    else:
        results[name] = pd.read_sql_query(sql, con)
con.close()

# Sanity checks against the CSV (both computed from training rows only).
rc = results["row_counts"].set_index("view")
assert int(rc.loc["all_rows", "n_rows"]) == len(train)
assert int(rc.loc["deduplicated", "n_rows"]) == int(train["is_first_copy"].sum())

L = [
    "# 04 · SQLite queries (training rows only)",
    "",
    f"Database: `data/processed/train_only.sqlite`, built only from `data/processed/train_rows.csv` "
    f"({len(train)} training rows). **No test rows were loaded.**",
    "Queries: `sql/train_queries.sql`. Each query reports two views: all training rows, and",
    "deduplicated training rows (one copy per identical record).",
    "",
]
for name, res in results.items():
    L += [f"## {name}", ""]
    if name == "symptom_prevalence_by_class":
        t = res.set_index(["view", "class"])[SYMPTOM_COLUMNS].T
        t.columns = [f"{v} · {c} (% Yes)" for v, c in t.columns]
        L += ["Percentage answering Yes (transposed for readability):", "", md_table(t, index=True), ""]
        L += ["Group sizes: " + "; ".join(f"{r.view} {r['class']} n={r.n}" for _, r in res.iterrows()), ""]
    else:
        L += [md_table(res), ""]
L += [
    "Notes:",
    "",
    "- These are descriptive counts within the training partition.",
    "- They do not show causation and do not describe the general population. The data come from",
    "  a single hospital's questionnaire sample.",
]
(REPORTS_DIR / "04_sql_training.md").write_text("\n".join(L) + "\n")
for name, res in results.items():
    print(f"--- {name}\n{res.to_string(index=False)}")
