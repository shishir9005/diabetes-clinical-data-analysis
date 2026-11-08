# 03 · Frozen grouped split and cross-validation folds

Test split frozen (UTC): 2026-11-06T13:27:12Z (unchanged). CV folds regenerated (UTC): 2026-11-07T13:35:42Z, seeds [0, 1, 2]. Overall status: **ALL CHECKS PASSED — folds frozen**.

This report replaces the 6 Nov 2026 version (CV seeds [100, 101, 102]). The old report, folds and config are archived in `archive/day1_pre_correction_2026-11-07/` (see its `ARCHIVE_INDEX.md`).

Views: **primary** = all 520 original records (`all_rows`); **sensitivity** = the 251 unique questionnaire profiles, one record per profile (`deduplicated`, `is_first_copy == 1`). The split and the folds are assigned per `profile_id`, so one assignment serves both views.

## What changed on 7 Nov 2026

| item | 6 Nov 2026 | 7 Nov 2026 |
|---|---|---|
| CV seeds (3 repeats x 5 folds) | [100, 101, 102] | [0, 1, 2] |
| train/test split (seed 42, fold 0) | frozen | unchanged; not regenerated |
| split_assignment.csv SHA-256 | 8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e | 8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e (identical) |
| train_rows.csv SHA-256 | 05dfd30e3ff96f102d95e4f5198037071f5e4484b87f695f8635a55a6e1370e8 | 05dfd30e3ff96f102d95e4f5198037071f5e4484b87f695f8635a55a6e1370e8 (identical) |
| cv_folds.csv SHA-256 | 9bbd563722eddc46552161b90def120291225206eed96ffe307a6e1d60551bef | f71888b1ba497c57524de495bdd9d93133bc7ff94541228850b6773fa58bd696 |
| split_config.json SHA-256 | 4a0b51c457b461e4d943818fac9fa891e76fe324ff3d4c7d96f55d7daa1df5aa | 721a9d99859bdcce5fe138450297dbbdb1b2cc2e30d9ce08800bea993f015f3b |

## Configuration

- Test split (unchanged): `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on all 520 rows, fold 0 predeclared as the test set. It was **not** regenerated.
- CV: 3 repeats × `StratifiedGroupKFold(n_splits=5, shuffle=True)` on the frozen training partition (`train_rows.csv`, y = `label`, groups = `profile_id`), with `random_state` = [0, 1, 2]. That gives **15 folds**.
- Sensitivity view: each row keeps its profile's fold, and only `is_first_copy == 1` rows are used.
- Input: `cleaned_all_rows.csv`, SHA-256 `2678581eedfbffae3d1d2a4668b3cbfffca78bd5369bcbc2a6981ac4cd810ff8`.
- Versions at the original freeze: {'python': '3.12.16', 'numpy': '2.5.4', 'pandas': '3.1.1', 'scikit-learn': '1.9.1'}.
- Versions used for this regeneration: {'python': '3.12.16', 'numpy': '2.5.4', 'pandas': '3.1.1', 'scikit-learn': '1.9.1'}.

## Acceptance checks

| scope | check | result | detail |
|---|---|---|---|
| frozen inputs | split_assignment.csv, train_rows.csv and cleaned_all_rows.csv match their recorded SHA-256 | PASS | verified before any work |
| frozen inputs | train_rows.csv holds exactly the frozen training partition | PASS | 416 rows |
| frozen inputs | training rows are in original file order | PASS |  |
| frozen inputs | train and test ids are disjoint and together cover all 520 rows | PASS |  |
| frozen inputs | train_rows.csv profile ids, view flags and labels agree with the cleaned file | PASS |  |
| test split [all_rows] | both classes in training partition | PASS |  |
| test split [all_rows] | both classes in test partition | PASS |  |
| test split [all_rows] | no profile group in both partitions | PASS | overlap=0 |
| test split [deduplicated] | both classes in training partition | PASS |  |
| test split [deduplicated] | both classes in test partition | PASS |  |
| test split [deduplicated] | no profile group in both partitions | PASS | overlap=0 |
| test split | split counts equal the original 6 Nov report (split unchanged) | PASS |  |
| repeat 0 (seed 0) | every training row in exactly one validation fold | PASS | min=1, max=1 |
| repeat 0 (seed 0) | all 5 folds are non-empty | PASS |  |
| repeat 1 (seed 1) | every training row in exactly one validation fold | PASS | min=1, max=1 |
| repeat 1 (seed 1) | all 5 folds are non-empty | PASS |  |
| repeat 2 (seed 2) | every training row in exactly one validation fold | PASS | min=1, max=1 |
| repeat 2 (seed 2) | all 5 folds are non-empty | PASS |  |
| CV folds | all 15 folds x 2 views pass all three checks (both classes in training and validation parts; no shared profile) | PASS | 30/30 fold-view combinations passed |
| CV folds | no test row appears in any fold (CV rows are the training rows only) | PASS |  |
| CV folds | the three repeats produce different fold partitions (informational) | PASS | 3 distinct partitions out of 3 |
| CV folds | new folds differ from the archived 6 Nov folds in every repeat (informational) | PASS | identical to the old repeat: [False, False, False] |

## Test split counts (permitted split-count report; identical to the 6 Nov report)

| view | role | partition | rows | groups | positive | negative | pct_positive |
|---|---|---|---|---|---|---|---|
| all_rows | primary | train | 416 | 202 | 256 | 160 | 61.54 |
| all_rows | primary | test | 104 | 49 | 64 | 40 | 61.54 |
| deduplicated | sensitivity | train | 202 | 202 | 139 | 63 | 68.81 |
| deduplicated | sensitivity | test | 49 | 49 | 34 | 15 | 69.39 |

## All 15 CV folds: primary view (all rows)

| repeat | seed | fold | train_rows | train_groups | train_positive | train_negative | train_pct_positive | val_rows | val_groups | val_positive | val_negative | val_pct_positive | both_classes_train | both_classes_val | profile_overlap | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 332 | 162 | 204 | 128 | 61.45 | 84 | 40 | 52 | 32 | 61.9 | True | True | 0 | PASS |
| 0 | 0 | 1 | 333 | 162 | 205 | 128 | 61.56 | 83 | 40 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 0 | 0 | 2 | 333 | 161 | 205 | 128 | 61.56 | 83 | 41 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 0 | 0 | 3 | 333 | 162 | 205 | 128 | 61.56 | 83 | 40 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 0 | 0 | 4 | 333 | 161 | 205 | 128 | 61.56 | 83 | 41 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 1 | 1 | 0 | 332 | 162 | 204 | 128 | 61.45 | 84 | 40 | 52 | 32 | 61.9 | True | True | 0 | PASS |
| 1 | 1 | 1 | 333 | 160 | 205 | 128 | 61.56 | 83 | 42 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 1 | 1 | 2 | 333 | 162 | 205 | 128 | 61.56 | 83 | 40 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 1 | 1 | 3 | 333 | 162 | 205 | 128 | 61.56 | 83 | 40 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 1 | 1 | 4 | 333 | 162 | 205 | 128 | 61.56 | 83 | 40 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 2 | 2 | 0 | 332 | 161 | 204 | 128 | 61.45 | 84 | 41 | 52 | 32 | 61.9 | True | True | 0 | PASS |
| 2 | 2 | 1 | 333 | 162 | 205 | 128 | 61.56 | 83 | 40 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 2 | 2 | 2 | 333 | 162 | 205 | 128 | 61.56 | 83 | 40 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 2 | 2 | 3 | 333 | 161 | 205 | 128 | 61.56 | 83 | 41 | 51 | 32 | 61.45 | True | True | 0 | PASS |
| 2 | 2 | 4 | 333 | 162 | 205 | 128 | 61.56 | 83 | 40 | 51 | 32 | 61.45 | True | True | 0 | PASS |

## All 15 CV folds: sensitivity view (unique profiles)

| repeat | seed | fold | train_rows | train_groups | train_positive | train_negative | train_pct_positive | val_rows | val_groups | val_positive | val_negative | val_pct_positive | both_classes_train | both_classes_val | profile_overlap | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |
| 0 | 0 | 1 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |
| 0 | 0 | 2 | 161 | 161 | 112 | 49 | 69.57 | 41 | 41 | 27 | 14 | 65.85 | True | True | 0 | PASS |
| 0 | 0 | 3 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |
| 0 | 0 | 4 | 161 | 161 | 111 | 50 | 68.94 | 41 | 41 | 28 | 13 | 68.29 | True | True | 0 | PASS |
| 1 | 1 | 0 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |
| 1 | 1 | 1 | 160 | 160 | 111 | 49 | 69.38 | 42 | 42 | 28 | 14 | 66.67 | True | True | 0 | PASS |
| 1 | 1 | 2 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |
| 1 | 1 | 3 | 162 | 162 | 112 | 50 | 69.14 | 40 | 40 | 27 | 13 | 67.5 | True | True | 0 | PASS |
| 1 | 1 | 4 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |
| 2 | 2 | 0 | 161 | 161 | 111 | 50 | 68.94 | 41 | 41 | 28 | 13 | 68.29 | True | True | 0 | PASS |
| 2 | 2 | 1 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |
| 2 | 2 | 2 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |
| 2 | 2 | 3 | 161 | 161 | 112 | 49 | 69.57 | 41 | 41 | 27 | 14 | 65.85 | True | True | 0 | PASS |
| 2 | 2 | 4 | 162 | 162 | 111 | 51 | 68.52 | 40 | 40 | 28 | 12 | 70.0 | True | True | 0 | PASS |

## Frozen files

| file | sha256 |
|---|---|
| data/splits/split_assignment.csv | 8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e |
| data/splits/cv_folds.csv | f71888b1ba497c57524de495bdd9d93133bc7ff94541228850b6773fa58bd696 |
| data/splits/split_config.json | 721a9d99859bdcce5fe138450297dbbdb1b2cc2e30d9ce08800bea993f015f3b |
| data/processed/train_rows.csv | 05dfd30e3ff96f102d95e4f5198037071f5e4484b87f695f8635a55a6e1370e8 |

`cv_folds.csv` and `split_config.json` were written a second time on 7 Nov 2026 under the controlled correction; the replaced versions are archived with their hashes. All four files are read-only. That is an integrity safeguard, not immutability.
