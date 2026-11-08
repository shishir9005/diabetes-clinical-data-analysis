"""Day 1 correction (7 Nov 2026): regenerate ONLY the cross-validation folds, with seeds [0, 1, 2].

Owner decision of 7 Nov 2026: the three CV repeats use StratifiedGroupKFold seeds 0, 1 and 2.
The original freeze (6 Nov 2026, src/s03_freeze_split.py) used 100, 101 and 102.

This script does NOT regenerate or change the train/test split. It reads the frozen training
partition (data/processed/train_rows.csv) and the frozen split_assignment.csv, and it refuses to
run unless both still match the SHA-256 values recorded at the original freeze. Test rows are
touched only through identifiers and the permitted split-count report (profile id, label and
view flag; no test-row features are read).

Safety rules enforced in code:
  * the pre-correction evidence must already be archived and hash-verified
    (archive/day1_pre_correction_2026-11-07/), so nothing is silently overwritten;
  * the live cv_folds.csv and split_config.json must already have been moved away;
  * if any acceptance check fails, nothing is written and the script exits non-zero.

Views: "all_rows" is the primary analysis (all 520 original records); "deduplicated" is the
sensitivity analysis (the 251 unique questionnaire profiles, flagged by is_first_copy == 1).
Folds are assigned per profile_id, so one assignment serves both views.

Run with:  <env python> -I -B src/s03b_regenerate_cv_folds.py
Writes (only if every check passes):
  data/splits/cv_folds.csv, data/splits/split_config.json   (made read-only)
  reports/03_split_and_cv.md, reports/03_split_and_cv.json
"""
from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from prep import CLEANED_ALL, PROJECT_ROOT, REPORTS_DIR, SPLITS_DIR, TRAIN_ROWS, md_table, sha256_file  # noqa: E402

CV_SEEDS, CV_N_SPLITS = [0, 1, 2], 5
OLD_CV_SEEDS = [100, 101, 102]
ARCHIVE_REL = "archive/day1_pre_correction_2026-11-07"
ARCHIVE = PROJECT_ROOT / ARCHIVE_REL

OUT_ASSIGN = SPLITS_DIR / "split_assignment.csv"
OUT_FOLDS = SPLITS_DIR / "cv_folds.csv"
OUT_CONFIG = SPLITS_DIR / "split_config.json"
REPORT_MD = REPORTS_DIR / "03_split_and_cv.md"
REPORT_JSON = REPORTS_DIR / "03_split_and_cv.json"

# SHA-256 values recorded at the original freeze (reports/sha256_manifest.txt, 6 Nov 2026).
MUST_BE_UNCHANGED = {
    OUT_ASSIGN: "8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e",
    TRAIN_ROWS: "05dfd30e3ff96f102d95e4f5198037071f5e4484b87f695f8635a55a6e1370e8",
    CLEANED_ALL: "2678581eedfbffae3d1d2a4668b3cbfffca78bd5369bcbc2a6981ac4cd810ff8",
}
ARCHIVED_6NOV = {
    "data/splits/cv_folds.csv": "9bbd563722eddc46552161b90def120291225206eed96ffe307a6e1d60551bef",
    "data/splits/split_config.json": "4a0b51c457b461e4d943818fac9fa891e76fe324ff3d4c7d96f55d7daa1df5aa",
    "reports/03_split_and_cv.md": "defc8361dbea5119e759c88a844a562033b0e200603580f74a0c83b484500232",
    "reports/03_split_and_cv.json": "2a90f9fa56ae5e59262d6f4a018a73e770ab8b666868a97b7edb0e184adec802",
}

# --- Guards: never overwrite evidence, never touch the frozen split ---------------------
problems = []
for path, expected in MUST_BE_UNCHANGED.items():
    if not path.exists() or sha256_file(path) != expected:
        problems.append(f"{path.relative_to(PROJECT_ROOT)} does not match its recorded SHA-256")
for rel, expected in ARCHIVED_6NOV.items():
    arch = ARCHIVE / rel
    if not arch.exists() or sha256_file(arch) != expected:
        problems.append(f"archived 6 Nov copy missing or altered: {ARCHIVE_REL}/{rel}")
for path in (OUT_FOLDS, OUT_CONFIG):
    if path.exists():
        problems.append(f"{path.relative_to(PROJECT_ROOT)} still exists; move it to the archive first (no silent overwrite)")
if problems:
    sys.exit("Refusing to run:\n  " + "\n  ".join(problems))

old_cfg = json.loads((ARCHIVE / "data/splits/split_config.json").read_text())
old_report = json.loads((ARCHIVE / "reports/03_split_and_cv.json").read_text())
if old_cfg["cv"]["random_state_per_repeat"] != OLD_CV_SEEDS:
    sys.exit("Archived config does not record the expected 6 Nov seeds.")

train = pd.read_csv(TRAIN_ROWS)                      # frozen training partition, original row order
assign = pd.read_csv(OUT_ASSIGN)                     # frozen split (identifiers only are used)
ids_only = pd.read_csv(CLEANED_ALL, usecols=["row_id", "profile_id", "is_first_copy", "label"])
checks: list[dict] = []


def add_check(scope: str, name: str, ok: bool, detail="") -> None:
    checks.append({"scope": scope, "check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)})


def counts(frame: pd.DataFrame) -> dict:
    n, pos = len(frame), int(frame["label"].sum())
    return {"rows": n, "groups": int(frame["profile_id"].nunique()), "positive": pos,
            "negative": n - pos, "pct_positive": round(100 * pos / n, 2) if n else float("nan")}


VIEWS = {  # key -> (role in the analysis plan, row filter)
    "all_rows": ("primary", lambda d: d),
    "deduplicated": ("sensitivity", lambda d: d[d["is_first_copy"] == 1]),
}

# --- Frozen inputs ------------------------------------------------------------------------
train_ids = set(assign.loc[assign["partition"] == "train", "row_id"])
test_ids = set(assign.loc[assign["partition"] == "test", "row_id"])
add_check("frozen inputs", "split_assignment.csv, train_rows.csv and cleaned_all_rows.csv match their recorded SHA-256",
          True, "verified before any work")
add_check("frozen inputs", "train_rows.csv holds exactly the frozen training partition",
          set(train["row_id"]) == train_ids and len(train) == len(train_ids) == 416, f"{len(train)} rows")
add_check("frozen inputs", "training rows are in original file order", bool((train["row_id"].diff().dropna() > 0).all()))
add_check("frozen inputs", "train and test ids are disjoint and together cover all 520 rows",
          not (train_ids & test_ids) and len(train_ids) + len(test_ids) == len(assign) == len(ids_only) == 520)
ref = ids_only.set_index("row_id").loc[train["row_id"]]
add_check("frozen inputs", "train_rows.csv profile ids, view flags and labels agree with the cleaned file",
          bool((ref["profile_id"].to_numpy() == train["profile_id"].to_numpy()).all()
               and (ref["is_first_copy"].to_numpy() == train["is_first_copy"].to_numpy()).all()
               and (ref["label"].to_numpy() == train["label"].to_numpy()).all()))

# --- Test split: re-checked, not regenerated ------------------------------------------------
cl = ids_only.merge(assign[["row_id", "partition"]], on="row_id", validate="one_to_one")
test_rows_report = []
for view, (role, f) in VIEWS.items():
    tr, te = f(cl[cl["partition"] == "train"]), f(cl[cl["partition"] == "test"])
    test_rows_report.append({"view": view, "role": role, "partition": "train", **counts(tr)})
    test_rows_report.append({"view": view, "role": role, "partition": "test", **counts(te)})
    add_check(f"test split [{view}]", "both classes in training partition", bool(tr["label"].nunique() == 2))
    add_check(f"test split [{view}]", "both classes in test partition", bool(te["label"].nunique() == 2))
    overlap = set(tr["profile_id"]) & set(te["profile_id"])
    add_check(f"test split [{view}]", "no profile group in both partitions", not overlap, f"overlap={len(overlap)}")
new_counts = [{k: v for k, v in d.items() if k != "role"} for d in test_rows_report]
add_check("test split", "split counts equal the original 6 Nov report (split unchanged)", new_counts == old_report["test_split_counts"])

# --- 15 CV folds on the frozen training partition ---------------------------------------------
fold_rows = []
partitions_by_repeat = []
for r, seed in enumerate(CV_SEEDS):
    col = f"cv_r{r}_fold"
    assign_fold = np.full(len(train), -1)
    times_validated = np.zeros(len(train), dtype=int)
    splits = list(StratifiedGroupKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=seed)
                  .split(np.zeros(len(train)), train["label"], train["profile_id"]))
    for k, (_, va) in enumerate(splits):
        assign_fold[va] = k
        times_validated[va] += 1
    train[col] = assign_fold
    add_check(f"repeat {r} (seed {seed})", "every training row in exactly one validation fold",
              bool((times_validated == 1).all()), f"min={times_validated.min()}, max={times_validated.max()}")
    add_check(f"repeat {r} (seed {seed})", "all 5 folds are non-empty",
              len(splits) == CV_N_SPLITS and all(len(va) > 0 for _, va in splits))
    partitions_by_repeat.append(frozenset(frozenset(train.loc[train[col] == k, "row_id"]) for k in range(CV_N_SPLITS)))
    for k in range(CV_N_SPLITS):
        for view, (role, f) in VIEWS.items():
            v = f(train)
            tr_part, va_part = v[v[col] != k], v[v[col] == k]
            both_tr = bool(tr_part["label"].nunique() == 2)
            both_va = bool(va_part["label"].nunique() == 2)
            overlap = set(tr_part["profile_id"]) & set(va_part["profile_id"])
            ok = both_tr and both_va and not overlap
            ctr, cva = counts(tr_part), counts(va_part)
            fold_rows.append({
                "repeat": r, "seed": seed, "fold": k, "view": view, "role": role,
                **{f"train_{a}": b for a, b in ctr.items()}, **{f"val_{a}": b for a, b in cva.items()},
                "both_classes_train": both_tr, "both_classes_val": both_va,
                "profile_overlap": len(overlap), "result": "PASS" if ok else "FAIL",
            })

fold_df = pd.DataFrame(fold_rows)
n_combos = len(fold_df)
n_fail = int((fold_df["result"] == "FAIL").sum())
add_check("CV folds", f"all {n_combos // 2} folds x 2 views pass all three checks "
                      "(both classes in training and validation parts; no shared profile)",
          n_fail == 0, f"{n_combos - n_fail}/{n_combos} fold-view combinations passed")
add_check("CV folds", "no test row appears in any fold (CV rows are the training rows only)",
          set(train["row_id"]) == train_ids and set(train["row_id"]).isdisjoint(test_ids))
distinct = len(set(partitions_by_repeat))
add_check("CV folds", "the three repeats produce different fold partitions (informational)",
          distinct == len(CV_SEEDS), f"{distinct} distinct partitions out of {len(CV_SEEDS)}")
old_folds = pd.read_csv(ARCHIVE / "data/splits/cv_folds.csv").set_index("row_id").loc[train["row_id"]]
same_as_old = [bool((old_folds[f"cv_r{r}_fold"].to_numpy() == train[f"cv_r{r}_fold"].to_numpy()).all())
               for r in range(len(CV_SEEDS))]
add_check("CV folds", "new folds differ from the archived 6 Nov folds in every repeat (informational)",
          not any(same_as_old), f"identical to the old repeat: {same_as_old}")

all_pass = all(c["result"] == "PASS" for c in checks)
if not all_pass:
    print(pd.DataFrame(checks).to_string(index=False))
    sys.exit("CHECK FAILURE: nothing was written. The method and rules were not changed.")

# --- Freeze (only because every check passed) ------------------------------------------------
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
cfg = json.loads(json.dumps(old_cfg))                # deep copy; the test-split block stays as frozen on 6 Nov
cfg["cv"]["random_state_per_repeat"] = CV_SEEDS
cfg["cv"]["regenerated_utc"] = now
cfg["cv"]["regenerated_by"] = "src/s03b_regenerate_cv_folds.py"
cfg["cv"]["reason"] = ("Owner decision, 7 Nov 2026: CV seeds fixed at [0, 1, 2]. Only the CV folds were regenerated; "
                       "the train/test split was not.")
cfg["cv"]["supersedes"] = {
    "archive": ARCHIVE_REL,
    "random_state_per_repeat": OLD_CV_SEEDS,
    "cv_folds_sha256": ARCHIVED_6NOV["data/splits/cv_folds.csv"],
    "split_config_sha256": ARCHIVED_6NOV["data/splits/split_config.json"],
}
cfg["view_roles"] = {"all_rows": "primary analysis: all 520 original records",
                     "deduplicated": "sensitivity analysis: the 251 unique questionnaire profiles (is_first_copy == 1)"}
cfg["cv_regeneration_versions"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                                   "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
cfg["acceptance_checks_passed"] = True

fold_cols = ["row_id", "profile_id", "is_first_copy"] + [f"cv_r{r}_fold" for r in range(len(CV_SEEDS))]
train[fold_cols].to_csv(OUT_FOLDS, index=False)
OUT_CONFIG.write_text(json.dumps(cfg, indent=2))
for p in (OUT_FOLDS, OUT_CONFIG):                    # integrity safeguard against accidental edits; not immutability
    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
FROZEN = [OUT_ASSIGN, OUT_FOLDS, OUT_CONFIG, TRAIN_ROWS]
frozen_sha = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in FROZEN}
cfg_report = dict(cfg)
cfg_report["frozen_file_sha256"] = frozen_sha

REPORT_JSON.write_text(json.dumps(
    {"config": cfg_report, "checks": checks, "test_split_counts": test_rows_report, "cv_folds": fold_rows},
    indent=2, default=str))

# --- Report -------------------------------------------------------------------------------------
change_table = pd.DataFrame([
    {"item": "CV seeds (3 repeats x 5 folds)", "6 Nov 2026": str(OLD_CV_SEEDS), "7 Nov 2026": str(CV_SEEDS)},
    {"item": "train/test split (seed 42, fold 0)", "6 Nov 2026": "frozen", "7 Nov 2026": "unchanged; not regenerated"},
    {"item": "split_assignment.csv SHA-256", "6 Nov 2026": MUST_BE_UNCHANGED[OUT_ASSIGN],
     "7 Nov 2026": frozen_sha["data/splits/split_assignment.csv"] + " (identical)"},
    {"item": "train_rows.csv SHA-256", "6 Nov 2026": MUST_BE_UNCHANGED[TRAIN_ROWS],
     "7 Nov 2026": frozen_sha["data/processed/train_rows.csv"] + " (identical)"},
    {"item": "cv_folds.csv SHA-256", "6 Nov 2026": ARCHIVED_6NOV["data/splits/cv_folds.csv"],
     "7 Nov 2026": frozen_sha["data/splits/cv_folds.csv"]},
    {"item": "split_config.json SHA-256", "6 Nov 2026": ARCHIVED_6NOV["data/splits/split_config.json"],
     "7 Nov 2026": frozen_sha["data/splits/split_config.json"]},
])
show_cols = ["repeat", "seed", "fold", "train_rows", "train_groups", "train_positive", "train_negative", "train_pct_positive",
             "val_rows", "val_groups", "val_positive", "val_negative", "val_pct_positive",
             "both_classes_train", "both_classes_val", "profile_overlap", "result"]
L = [
    "# 03 · Frozen grouped split and cross-validation folds",
    "",
    f"Test split frozen (UTC): {cfg['created_utc']} (unchanged). CV folds regenerated (UTC): {now}, seeds {CV_SEEDS}. "
    "Overall status: **ALL CHECKS PASSED — folds frozen**.",
    "",
    f"This report replaces the 6 Nov 2026 version (CV seeds {OLD_CV_SEEDS}). The old report, folds and config are "
    f"archived in `{ARCHIVE_REL}/` (see its `ARCHIVE_INDEX.md`).",
    "",
    "Views: **primary** = all 520 original records (`all_rows`); **sensitivity** = the 251 unique questionnaire "
    "profiles, one record per profile (`deduplicated`, `is_first_copy == 1`). The split and the folds are assigned per "
    "`profile_id`, so one assignment serves both views.",
    "",
    "## What changed on 7 Nov 2026",
    "",
    md_table(change_table),
    "",
    "## Configuration",
    "",
    f"- Test split (unchanged): `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on all 520 rows, "
    "fold 0 predeclared as the test set. It was **not** regenerated.",
    f"- CV: {len(CV_SEEDS)} repeats × `StratifiedGroupKFold(n_splits={CV_N_SPLITS}, shuffle=True)` on the frozen training "
    f"partition (`train_rows.csv`, y = `label`, groups = `profile_id`), with `random_state` = {CV_SEEDS}. "
    f"That gives **{CV_N_SPLITS * len(CV_SEEDS)} folds**.",
    "- Sensitivity view: each row keeps its profile's fold, and only `is_first_copy == 1` rows are used.",
    f"- Input: `cleaned_all_rows.csv`, SHA-256 `{cfg['input_sha256']}`.",
    f"- Versions at the original freeze: {cfg['versions']}.",
    f"- Versions used for this regeneration: {cfg['cv_regeneration_versions']}.",
    "",
    "## Acceptance checks",
    "",
    md_table(pd.DataFrame(checks)),
    "",
    "## Test split counts (permitted split-count report; identical to the 6 Nov report)",
    "",
    md_table(pd.DataFrame(test_rows_report)),
    "",
    "## All 15 CV folds: primary view (all rows)",
    "",
    md_table(fold_df[fold_df["view"] == "all_rows"][show_cols]),
    "",
    "## All 15 CV folds: sensitivity view (unique profiles)",
    "",
    md_table(fold_df[fold_df["view"] == "deduplicated"][show_cols]),
    "",
    "## Frozen files",
    "",
    md_table(pd.DataFrame([{"file": k, "sha256": v} for k, v in frozen_sha.items()])),
    "",
    "`cv_folds.csv` and `split_config.json` were written a second time on 7 Nov 2026 under the controlled correction; "
    "the replaced versions are archived with their hashes. All four files are read-only. That is an "
    "integrity safeguard, not immutability.",
]
REPORT_MD.write_text("\n".join(L) + "\n")

print(pd.DataFrame(checks).to_string(index=False))
print(pd.DataFrame(test_rows_report).to_string(index=False))
print(fold_df[["repeat", "seed", "fold", "view", "train_rows", "train_positive", "train_negative", "val_rows",
               "val_positive", "val_negative", "profile_overlap", "result"]].to_string(index=False))
print("frozen file hashes:", json.dumps(frozen_sha, indent=2))
