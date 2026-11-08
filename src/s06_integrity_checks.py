"""Day 1, step 9: integrity checks after all processing.

Re-verifies raw-file hashes, frozen-split hashes and test-set isolation.
Reading the split assignment here is a permitted integrity/split-count check.

Run with:  <env python> -I -B src/s06_integrity_checks.py
Writes:    reports/06_integrity_checks.md
"""
from __future__ import annotations

import json
import sqlite3
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

from prep import (CLEANED_ALL, PROCESSED_DIR, PROJECT_ROOT, RAW_CSV, RAW_ZIP, REPORTS_DIR,  # noqa: E402
                  SPLITS_DIR, TRAIN_ROWS, md_table, sha256_file)

RECORDED = {  # values recorded at download time (docs/DATA_SOURCE.md)
    RAW_ZIP: "67594c0f11257688b57493798e70737b7156ba37a2e6ac75507ae3a35221a388",
    RAW_CSV: "7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137",
}
rows: list[dict] = []


def check(name: str, ok: bool, detail="") -> None:
    rows.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)})


def rel(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT))


def no_write_bits(p: Path) -> bool:
    return not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


# 1. Raw files: hash after processing equals hash at download; still read-only.
for p, h in RECORDED.items():
    actual = sha256_file(p)
    check(f"raw hash unchanged after processing: {rel(p)}", actual == h, actual)
    check(f"raw file has no write permission bits: {rel(p)}", no_write_bits(p), oct(p.stat().st_mode & 0o777))

# 2. Frozen split files: hashes equal those recorded when frozen; still read-only.
split_report = json.loads((REPORTS_DIR / "03_split_and_cv.json").read_text())
for relpath, h in split_report["config"]["frozen_file_sha256"].items():
    p = PROJECT_ROOT / relpath
    actual = sha256_file(p)
    check(f"frozen file hash unchanged: {relpath}", actual == h, actual)
    check(f"frozen file has no write permission bits: {relpath}", no_write_bits(p), oct(p.stat().st_mode & 0o777))
check("cleaned_all_rows.csv hash equals split input hash",
      sha256_file(CLEANED_ALL) == split_report["config"]["input_sha256"], sha256_file(CLEANED_ALL))

# 3. Test-set isolation (permitted integrity check on identifiers only).
assign = pd.read_csv(SPLITS_DIR / "split_assignment.csv")
train_ids = set(assign.loc[assign["partition"] == "train", "row_id"])
test_ids = set(assign.loc[assign["partition"] == "test", "row_id"])
tr = pd.read_csv(TRAIN_ROWS)
folds = pd.read_csv(SPLITS_DIR / "cv_folds.csv")
con = sqlite3.connect(f"file:{PROCESSED_DIR / 'train_only.sqlite'}?mode=ro", uri=True)
db_ids = {r[0] for r in con.execute("SELECT row_id FROM train_rows")}
db_tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
con.close()
check("train_rows.csv contains exactly the training partition", set(tr["row_id"]) == train_ids, f"{len(tr)} rows")
check("train_rows.csv contains no test row", not (set(tr["row_id"]) & test_ids))
check("cv_folds.csv contains exactly the training partition", set(folds["row_id"]) == train_ids, f"{len(folds)} rows")
check("SQLite database contains exactly the training partition", db_ids == train_ids, f"{len(db_ids)} rows; tables={db_tables}")
check("SQLite database contains no test row", not (db_ids & test_ids))
check("partition sizes", len(train_ids) == 416 and len(test_ids) == 104 and not (train_ids & test_ids),
      f"train={len(train_ids)}, test={len(test_ids)}")

ok = all(r["result"] == "PASS" for r in rows)
L = [
    "# 06 · Integrity checks after Day 1 processing",
    "",
    f"Overall: **{'ALL PASS' if ok else 'FAILURES PRESENT'}** ({sum(r['result'] == 'PASS' for r in rows)}/{len(rows)}).",
    "",
    "Read-only permissions are an integrity safeguard against accidental edits. They do not make",
    "files immutable; the hashes are the actual evidence that nothing changed.",
    "",
    md_table(pd.DataFrame(rows)),
]
(REPORTS_DIR / "06_integrity_checks.md").write_text("\n".join(L) + "\n")
print(pd.DataFrame(rows).to_string(index=False))
sys.exit(0 if ok else 1)
