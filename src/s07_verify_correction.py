"""Day 1 correction (7 Nov 2026): independent verification of the corrected state.

Read-only except for its own report, reports/07_correction_verification.md. It imports no project
module and fits no estimator. It uses the standard library, numpy and scikit-learn's
StratifiedGroupKFold splitter (only to re-derive the fold assignments independently of the generator).

Run with:  <env python> -I -B src/s07_verify_correction.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import stat
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
ARCH_REL = "archive/day1_pre_correction_2026-11-07"
ARCH = ROOT / ARCH_REL
OUT = ROOT / "reports" / "07_correction_verification.md"

RAW_ZIP_REL = "data/raw/archive/early_stage_diabetes_risk_prediction_uci529.zip"
RAW_CSV_REL = "data/raw/extracted/diabetes_data_upload.csv"
REC_ZIP = "67594c0f11257688b57493798e70737b7156ba37a2e6ac75507ae3a35221a388"
REC_RAW = "7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137"
SPLIT_ASSIGN = "8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e"
TRAIN_ROWS = "05dfd30e3ff96f102d95e4f5198037071f5e4484b87f695f8635a55a6e1370e8"
CLEANED = "2678581eedfbffae3d1d2a4668b3cbfffca78bd5369bcbc2a6981ac4cd810ff8"
NEW_SEEDS, OLD_SEEDS = [0, 1, 2], [100, 101, 102]

EXPECTED_CHANGED = {
    "data/splits/cv_folds.csv", "data/splits/split_config.json", "docs/PROJECT_PLAN.md",
    "reports/02_duplicate_audit.md", "reports/03_split_and_cv.json", "reports/03_split_and_cv.md",
    "reports/06_integrity_checks.md", "reports/DAY1_SUMMARY.md", "src/s03_freeze_split.py",
}
EXPECTED_NEW = {".gitignore", "src/s03b_regenerate_cv_folds.py", "src/s07_verify_correction.py"}
WRITTEN_LATER = {"reports/07_correction_verification.md", "reports/07_day1_correction_report.md"}

checks: list[tuple[int, str, str, bool, str]] = []


def chk(area: str, name: str, ok, detail="") -> None:
    checks.append((len(checks) + 1, area, name, bool(ok), str(detail)))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_of(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def manifest_of(p: Path) -> dict[str, str]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            out[rel.strip()] = h
    return out


def no_write(p: Path) -> bool:
    return not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


# --- 1. Scope: what changed since the audited 6 Nov manifest ---------------------------------
old_man = manifest_of(ARCH / "reports/sha256_manifest.txt")
cur: dict[str, str] = {}
for p in sorted(ROOT.rglob("*")):
    if not p.is_file():
        continue
    parts = p.relative_to(ROOT).parts
    rel = str(p.relative_to(ROOT))
    if ".git" in parts or rel == "reports/sha256_manifest.txt":
        continue
    cur[rel] = sha(p)
changed = sorted(r for r in old_man if r in cur and cur[r] != old_man[r])
unchanged = sorted(r for r in old_man if r in cur and cur[r] == old_man[r])
removed = sorted(r for r in old_man if r not in cur)
new = sorted(r for r in cur if r not in old_man)
new_unexpected = [r for r in new if not r.startswith(ARCH_REL + "/") and r not in EXPECTED_NEW | WRITTEN_LATER]

chk("scope", "the 6 Nov manifest (archived) lists 37 files", len(old_man) == 37, len(old_man))
chk("scope", "files whose content changed since the 6 Nov manifest are exactly the intended nine",
    set(changed) == EXPECTED_CHANGED,
    f"unexpected={sorted(set(changed) - EXPECTED_CHANGED)}, missing={sorted(EXPECTED_CHANGED - set(changed))}")
chk("scope", f"the other {len(unchanged)} files from the 6 Nov manifest are byte-identical", len(unchanged) == 37 - len(EXPECTED_CHANGED), len(unchanged))
chk("scope", "no file from the 6 Nov manifest was removed", not removed, removed)
chk("scope", "new files are only the intended scripts, .gitignore, report 07 and the archive",
    not new_unexpected and EXPECTED_NEW <= set(new), new_unexpected)

# --- F1-F4: untouched data, split, training file, SQL and EDA ----------------------------------
chk("F1 raw data", "raw zip and CSV hashes equal the download record", cur[RAW_ZIP_REL] == REC_ZIP and cur[RAW_CSV_REL] == REC_RAW)
chk("F1 raw data", "raw zip and CSV are unchanged and still read-only",
    RAW_ZIP_REL in unchanged and RAW_CSV_REL in unchanged and no_write(ROOT / RAW_ZIP_REL) and no_write(ROOT / RAW_CSV_REL))
chk("F2 split", "split_assignment.csv SHA-256 is exactly 8f0077e5… (full value recorded in the 6 Nov manifest)",
    cur["data/splits/split_assignment.csv"] == SPLIT_ASSIGN and "data/splits/split_assignment.csv" in unchanged,
    cur["data/splits/split_assignment.csv"])
chk("F3 training file", "train_rows.csv and cleaned_all_rows.csv are unchanged",
    cur["data/processed/train_rows.csv"] == TRAIN_ROWS and cur["data/processed/cleaned_all_rows.csv"] == CLEANED
    and "data/processed/train_rows.csv" in unchanged and "data/processed/cleaned_all_rows.csv" in unchanged)
SQL_EDA = ["data/processed/train_only.sqlite", "sql/train_queries.sql", "src/s04_sql_train.py", "src/s05_eda_train.py",
           "reports/04_sql_training.md", "reports/05_eda_training.md"] + sorted(r for r in old_man if r.startswith("reports/figures/"))
chk("F4 SQL/EDA", f"SQL and EDA outputs are byte-identical to 6 Nov ({len(SQL_EDA)} files: database, queries, 2 scripts, reports 04/05, 4 figures)",
    len(SQL_EDA) == 10 and all(r in unchanged for r in SQL_EDA), [r for r in SQL_EDA if r not in unchanged])

# --- Archive ------------------------------------------------------------------------------------
arch_files = sorted(str(p.relative_to(ARCH)) for p in ARCH.rglob("*") if p.is_file())
index_text = (ARCH / "ARCHIVE_INDEX.md").read_text(encoding="utf-8")
arch_bad = []
for rel in arch_files:
    if rel == "ARCHIVE_INDEX.md":
        if not no_write(ARCH / rel):
            arch_bad.append(rel + " (writable)")
        continue
    p = ARCH / rel
    ok_hash = (sha(p) == old_man[rel]) if rel in old_man else (sha(p) in index_text)
    if not ok_hash:
        arch_bad.append(rel + " (hash)")
    if rel not in index_text:
        arch_bad.append(rel + " (not in index)")
    if not no_write(p):
        arch_bad.append(rel + " (writable)")
chk("archive", "11 archived files: hashes equal the 6 Nov manifest (the manifest's own hash is in the index), all listed in ARCHIVE_INDEX.md, all read-only",
    len(arch_files) == 11 and not arch_bad, arch_bad)

# --- 2. CV seeds, independent regeneration ------------------------------------------------------
train_rows = rows_of(ROOT / "data/processed/train_rows.csv")
n = len(train_rows)
row_ids = [int(r["row_id"]) for r in train_rows]
y = np.array([int(r["label"]) for r in train_rows])
g = np.array([int(r["profile_id"]) for r in train_rows])
fcp = np.array([int(r["is_first_copy"]) for r in train_rows])
X = np.arange(n).reshape(-1, 1)          # StratifiedGroupKFold ignores X; a different X than the generator used
new_folds = rows_of(ROOT / "data/splits/cv_folds.csv")
old_folds = rows_of(ARCH / "data/splits/cv_folds.csv")


def regen(seed: int) -> np.ndarray:
    f = np.full(n, -1)
    for k, (_, va) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g)):
        f[va] = k
    return f


def col(rows: list[dict], r: int) -> np.ndarray:
    return np.array([int(x[f"cv_r{r}_fold"]) for x in rows])


chk("CV", "cv_folds.csv has the same 416 training rows, in the same order, as train_rows.csv",
    n == 416 and [int(r["row_id"]) for r in new_folds] == row_ids)
chk("CV", "the archived 6 Nov cv_folds.csv has the same rows in the same order",
    [int(r["row_id"]) for r in old_folds] == row_ids)
for r, seed in enumerate(NEW_SEEDS):
    chk("CV seeds", f"stored repeat {r} equals an independent StratifiedGroupKFold(n_splits=5, shuffle=True, random_state={seed}) on train_rows.csv",
        np.array_equal(col(new_folds, r), regen(seed)))
for r, seed in enumerate(OLD_SEEDS):
    chk("CV seeds", f"archived 6 Nov repeat {r} equals the same regeneration with its old seed {seed} (so the method is reproduced)",
        np.array_equal(col(old_folds, r), regen(seed)))
chk("CV seeds", "no stored repeat equals a regeneration with any of the old seeds 100, 101, 102",
    not any(np.array_equal(col(new_folds, r), regen(s)) for r in range(3) for s in OLD_SEEDS))
cfg = json.loads((ROOT / "data/splits/split_config.json").read_text())
old_cfg = json.loads((ARCH / "data/splits/split_config.json").read_text())
rep = json.loads((ROOT / "reports/03_split_and_cv.json").read_text())
s03_text = (ROOT / "src/s03_freeze_split.py").read_text(encoding="utf-8")
s03b_text = (ROOT / "src/s03b_regenerate_cv_folds.py").read_text(encoding="utf-8")
chk("CV seeds", "split_config.json records random_state_per_repeat == [0, 1, 2]", cfg["cv"]["random_state_per_repeat"] == NEW_SEEDS)
chk("CV seeds", "report 03 records [0, 1, 2] and its 30-row fold table uses seeds 0, 1, 2 only",
    rep["config"]["cv"]["random_state_per_repeat"] == NEW_SEEDS and {d["seed"] for d in rep["cv_folds"]} == set(NEW_SEEDS)
    and len(rep["cv_folds"]) == 30)
chk("CV seeds", "both fold-generating scripts define CV_SEEDS = [0, 1, 2]",
    "CV_SEEDS, CV_N_SPLITS = [0, 1, 2], 5" in s03_text and "CV_SEEDS, CV_N_SPLITS = [0, 1, 2], 5" in s03b_text)
chk("split", "the test-split block, input hashes, group column, creation time and original versions in split_config.json equal the 6 Nov config",
    all(cfg[k] == old_cfg[k] for k in ("test_split", "input_file", "input_sha256", "raw_csv_sha256", "group_column", "created_utc", "versions")))
chk("CV", "split_config.json 'supersedes' block records the archived 6 Nov hashes",
    cfg["cv"]["supersedes"]["cv_folds_sha256"] == old_man["data/splits/cv_folds.csv"]
    and cfg["cv"]["supersedes"]["split_config_sha256"] == old_man["data/splits/split_config.json"]
    and cfg["cv"]["supersedes"]["random_state_per_repeat"] == OLD_SEEDS)
chk("F5 seeds", "the new folds use exactly [0, 1, 2]: config, report, scripts and independent regeneration all agree",
    all(c[3] for c in checks if c[1] in ("CV seeds",)))

# --- 3. Acceptance checks, independently re-implemented ---------------------------------------
assign = rows_of(ROOT / "data/splits/split_assignment.csv")
train_ids = {int(r["row_id"]) for r in assign if r["partition"] == "train"}
test_ids = {int(r["row_id"]) for r in assign if r["partition"] == "test"}
chk("CV acceptance", "CV rows are exactly the 416 training rows, each once", set(row_ids) == train_ids and len(row_ids) == len(set(row_ids)) == 416)
chk("CV acceptance", "no test row appears in the CV folds", not (set(row_ids) & test_ids), len(test_ids))
VIEWS = {"all_rows": np.ones(n, dtype=bool), "deduplicated": fcp == 1}
rep_rows = {(d["repeat"], d["fold"], d["view"]): d for d in rep["cv_folds"]}
total_pass = total_combo = report_mismatch = 0
for r, seed in enumerate(NEW_SEEDS):
    fold = col(new_folds, r)
    once = bool(((fold >= 0) & (fold <= 4)).all()) and sorted(set(fold.tolist())) == [0, 1, 2, 3, 4]
    per_profile = defaultdict(set)
    for p_, f_ in zip(g.tolist(), fold.tolist()):
        per_profile[p_].add(f_)
    chk("CV acceptance", f"repeat {r} (seed {seed}): every training row has exactly one validation fold (0-4, all five used)", once)
    chk("CV acceptance", f"repeat {r} (seed {seed}): every profile sits in exactly one fold", all(len(s) == 1 for s in per_profile.values()))
    for view, mask in VIEWS.items():
        ok_folds = 0
        for k in range(5):
            va, tr = mask & (fold == k), mask & (fold != k)
            both = len(set(y[va].tolist())) == 2 and len(set(y[tr].tolist())) == 2
            overlap = set(g[va].tolist()) & set(g[tr].tolist())
            ok_folds += bool(both and not overlap)
            total_combo += 1
            total_pass += bool(both and not overlap)
            d = rep_rows[(r, k, view)]
            got = (int(tr.sum()), len(set(g[tr].tolist())), int(y[tr].sum()), int(va.sum()), len(set(g[va].tolist())), int(y[va].sum()))
            want = (d["train_rows"], d["train_groups"], d["train_positive"], d["val_rows"], d["val_groups"], d["val_positive"])
            report_mismatch += (got != want)
        chk("CV acceptance", f"repeat {r} (seed {seed}), {'primary (all rows)' if view == 'all_rows' else 'sensitivity (unique profiles)'}: "
                             "both classes in every training and validation part, no shared profile", ok_folds == 5, f"{ok_folds}/5 folds")
chk("F6 acceptance", "all 30 fold-view combinations pass (3 repeats x 5 folds x 2 views)", total_combo == 30 and total_pass == 30, f"{total_pass}/{total_combo}")
chk("F6 acceptance", "fold counts in report 03 equal the independent recount for all 30 combinations", report_mismatch == 0, f"mismatches={report_mismatch}")

# --- 4. Views ---------------------------------------------------------------------------------------
cleaned = rows_of(ROOT / "data/processed/cleaned_all_rows.csv")
c_label = np.array([int(r["label"]) for r in cleaned])
c_prof = np.array([int(r["profile_id"]) for r in cleaned])
c_rec = np.array([int(r["record_id"]) for r in cleaned])
c_fcp = np.array([int(r["is_first_copy"]) for r in cleaned])
c_part = np.array([r["partition"] for r in assign])
aligned = [int(r["row_id"]) for r in cleaned] == [int(r["row_id"]) for r in assign]


def cnt(mask: np.ndarray) -> tuple:
    k = int(mask.sum())
    pos = int(c_label[mask].sum())
    return (k, len(set(c_prof[mask].tolist())), pos, k - pos)


is_tr, is_te = c_part == "train", c_part == "test"
chk("F7 views", "cleaned file and split assignment list the same 520 rows in the same order", aligned and len(cleaned) == 520)
chk("F7 views", "primary view (all 520 records): train / test (rows, profiles, positive, negative) = (416, 202, 256, 160) / (104, 49, 64, 40)",
    cnt(is_tr) == (416, 202, 256, 160) and cnt(is_te) == (104, 49, 64, 40), f"{cnt(is_tr)} / {cnt(is_te)}")
chk("F7 views", "sensitivity view (251 unique profiles): train / test = (202, 202, 139, 63) / (49, 49, 34, 15)",
    cnt(is_tr & (c_fcp == 1)) == (202, 202, 139, 63) and cnt(is_te & (c_fcp == 1)) == (49, 49, 34, 15),
    f"{cnt(is_tr & (c_fcp == 1))} / {cnt(is_te & (c_fcp == 1))}")
by_prof = defaultdict(set)
for p_, l_ in zip(c_prof.tolist(), c_label.tolist()):
    by_prof[p_].add(l_)
chk("F7 views", "251 unique records = 251 unique profiles, 0 conflicting profiles, and the sensitivity view holds one record per profile",
    len(set(c_rec.tolist())) == 251 == len(set(c_prof.tolist())) and sum(len(s) > 1 for s in by_prof.values()) == 0
    and int(c_fcp.sum()) == 251 == len(set(c_prof[c_fcp == 1].tolist())))
chk("F7 views", "neither view was deleted: both remain in cleaned_all_rows.csv (520 rows) and train_rows.csv (416 rows, 202 flagged is_first_copy)",
    len(cleaned) == 520 and n == 416 and int(fcp.sum()) == 202)

# --- SQLite (read-only, immutable) -----------------------------------------------------------------
con = sqlite3.connect(f"file:{ROOT / 'data/processed/train_only.sqlite'}?mode=ro&immutable=1", uri=True)
db_ids = {r[0] for r in con.execute("select row_id from train_rows")}
db_tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
con.close()
chk("F4 SQL/EDA", "the SQLite database holds exactly the 416 training rows (table train_rows) and no test row",
    db_ids == train_ids and db_tables == ["train_rows"] and not (db_ids & test_ids))

# --- 5. Wording consistency ---------------------------------------------------------------------------
live_docs = [ROOT / "docs/PROJECT_PLAN.md", ROOT / "docs/DATA_SOURCE.md"] + sorted(
    p for p in (ROOT / "reports").glob("*.md") if p.name != "05_eda_training.md" and not p.name.startswith("07_"))
FORBIDDEN = ["Primary analysis: exact duplicates removed", "Sensitivity analysis: all rows kept",
             "keep one copy of each identical record (251 rows)", "primary analysis on deduplicated data",
             "needs approval before any modelling", "ROC-AUC (n/a for the baseline)", "Proposed duplicate policy"]
stale = [f"{p.relative_to(ROOT)}: {ph}" for p in live_docs for ph in FORBIDDEN if ph in p.read_text(encoding="utf-8")]
chk("F7 views", "no live plan or report (05 excepted, see plan §8) still states the 6 Nov policy or the 'n/a' baseline wording", not stale, stale)
need = {
    "docs/PROJECT_PLAN.md": [r"Primary analysis: all 520[^\n]*", r"Sensitivity analysis: the 251 unique[^\n]*", r"Baseline ROC-AUC is 0\.5"],
    "reports/02_duplicate_audit.md": [r"Primary analysis:\*\* all 520[^\n]*", r"Sensitivity analysis:\*\* the 251 unique[^\n]*"],
    "reports/DAY1_SUMMARY.md": [r"Primary analysis:\*\* all 520[^\n]*", r"Sensitivity analysis:\*\* the 251 unique[^\n]*"],
}
missing = [f"{f}: {pat}" for f, pats in need.items() for pat in pats if not re.search(pat, (ROOT / f).read_text(encoding="utf-8"))]
chk("F7 views", "plan §3.5, audit §5 and the Day 1 summary all state primary = all 520 records and sensitivity = 251 unique profiles", not missing, missing)

# --- 6. No models ---------------------------------------------------------------------------------------
pat = re.compile(r"\.fit\(|fit_transform|\.predict|predict_proba|cross_val|GridSearch|RandomizedSearch|Classifier|Regression|\bSVC\b|"
                 r"RandomForest|NaiveBayes|CategoricalNB|\btorch\b|joblib\.dump|\bpickle\b|train_test_split")
hits = []
for p in sorted((ROOT / "src").glob("*.py")):
    if p.name == "s07_verify_correction.py":      # it defines the patterns, so it is excluded from its own scan
        continue
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if pat.search(line):
            hits.append(f"{p.name}:{i}")
MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".h5", ".keras", ".ipynb"}
model_files = [r for r in cur if Path(r).suffix in MODEL_SUFFIXES]
model_dirs = [d for d in ("models", "notebooks", "app") if (ROOT / d).exists()]
chk("F10 models", "no estimator is fitted anywhere in src/ (only allowed token: the smoke test's in-memory joblib.dump of a toy dict, s00:89)",
    set(hits) <= {"s00_smoke_test.py:89"}, hits)
chk("F10 models", "no model file, notebook, or models/notebooks/app folder exists; requirements.txt is unchanged",
    not model_files and not model_dirs and "requirements.txt" in unchanged, model_files + model_dirs)

# --- 7. Git (file-level checks; no git command is run) -------------------------------------------------
git = ROOT / ".git"
gcfg = (git / "config").read_text(encoding="utf-8") if git.exists() else ""
objects = [p for p in (git / "objects").rglob("*") if p.is_file()] if git.exists() else []
heads = [p for p in (git / "refs" / "heads").rglob("*") if p.is_file()] if git.exists() else []
gi = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
chk("F9 git", "a local repository exists on branch main with no remote configured",
    git.is_dir() and (git / "HEAD").read_text().strip() == "ref: refs/heads/main" and "[remote" not in gcfg and "[branch" not in gcfg)
chk("F9 git", "no commit exists (no objects, no branch ref) and nothing is staged (no index)",
    not objects and not heads and not (git / "index").exists(), f"objects={len(objects)}, heads={len(heads)}")
chk("F9 git", ".gitignore excludes data/raw/ and generated *.sqlite, and does not exclude the frozen split, reports or archive",
    "/data/raw/" in gi and "*.sqlite" in gi and not re.search(r"^/?(data/splits|reports|archive|data/processed)/?$", gi, re.M))

# --- Report ---------------------------------------------------------------------------------------------------
n_ok = sum(1 for c in checks if c[3])


def table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    return out


L = [
    "# 07 · Correction verification (7 Nov 2026)",
    "",
    f"Overall: **{'ALL PASS' if n_ok == len(checks) else 'FAILURES PRESENT'}** ({n_ok}/{len(checks)}).",
    "",
    "Independent re-implementation: standard library, numpy and scikit-learn's `StratifiedGroupKFold` splitter only "
    f"(python {sys.version.split()[0]}, numpy {np.__version__}, scikit-learn {sklearn.__version__}). "
    "No project module is imported and no estimator is fitted. Read-only apart from this report.",
    "The baseline for 'unchanged' is the archived 6 Nov manifest (`" + ARCH_REL + "/reports/sha256_manifest.txt`).",
    "The final manifest check and the Git command checks are run after this report is written and are reported in "
    "`07_day1_correction_report.md`.",
    "",
    "## Checks",
    "",
] + table(["#", "area", "check", "result", "detail"],
          [[c[0], c[1], c[2], "PASS" if c[3] else "**FAIL**", c[4]] for c in checks]) + [
    "",
    "## Files whose content changed since the 6 Nov manifest",
    "",
] + table(["file", "6 Nov SHA-256", "7 Nov SHA-256"], [[r, old_man[r], cur[r]] for r in changed]) + [
    "",
    f"## New files outside the archive ({len([r for r in new if not r.startswith(ARCH_REL + '/')])})",
    "",
] + table(["file", "SHA-256"], [[r, cur[r]] for r in new if not r.startswith(ARCH_REL + "/")]) + [
    "",
    "Reports `07_correction_verification.md` (this file) and `07_day1_correction_report.md` are written after this scan and are listed in the regenerated manifest.",
    "",
    f"## Archived files ({len(arch_files)})",
    "",
] + table(["file under `" + ARCH_REL + "/`", "SHA-256"], [[r, sha(ARCH / r)] for r in arch_files]) + [
    "",
    f"## Unchanged since 6 Nov ({len(unchanged)} files)",
    "",
    ", ".join(f"`{r}`" for r in unchanged),
]
OUT.write_text("\n".join(L) + "\n")

print(f"{n_ok}/{len(checks)} checks pass")
for c in checks:
    if not c[3]:
        print("FAIL:", c[0], c[1], c[2], c[4])
print("changed:", changed)
print("new outside archive:", [r for r in new if not r.startswith(ARCH_REL + "/")])
print("unchanged count:", len(unchanged))
sys.exit(0 if n_ok == len(checks) else 1)
