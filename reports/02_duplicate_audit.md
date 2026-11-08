# 02 · Duplicate and conflicting-profile audit

Scope: all 520 rows of the cleaned data. Full-file access is permitted here because auditing
and profile identification happen before the split is established. **No rows were removed.**

Definitions:

- **Identical questionnaire record**: two or more rows with the same values in all 16 features and the label.
- **Feature profile**: the 16 feature values. Rows with the same profile are always kept on the same side of any split.
- **Conflicting profile**: one feature profile that appears with both labels.

## 1. Identical questionnaire records versus repeated patients

The file has no patient identifier, visit date or record identifier. Identical rows therefore
cannot be attributed to the same person. Each one could be the same patient entered more than
once, a copy made while compiling the data, or different people who gave the same answers.

| Category | Count |
|---|---|
| Identical questionnaire records (rows in duplicate groups) | 376 |
| Confirmed repeated patients | **0 confirmed.** Cannot be determined from this data. |

## 2. Exact duplicate records (all 17 columns identical)

| Measure | Count | % of 520 |
|---|---|---|
| Total rows | 520 | 100.00 |
| Unique records | 251 | 48.27 |
| Redundant copies (rows − unique records) | 269 | 51.73 |
| Duplicate groups (records appearing ≥ 2 times) | 107 | – |
| Rows belonging to a duplicate group (incl. first copy) | 376 | 72.31 |

Group-size distribution (copies per duplicate group → number of groups):

| copies_in_group | number_of_groups |
|---|---|
| 2 | 38 |
| 3 | 34 |
| 4 | 8 |
| 5 | 9 |
| 6 | 14 |
| 7 | 3 |
| 16 | 1 |

Class split:

| subset | rows | positive | negative | pct_positive |
|---|---|---|---|---|
| all rows | 520 | 320 | 200 | 61.54 |
| deduplicated (one copy per record) | 251 | 173 | 78 | 68.92 |
| rows in duplicate groups | 376 | 220 | 156 | 58.51 |
| redundant copies only | 269 | 147 | 122 | 54.65 |

The ten largest duplicate groups are shown below. `n_symptoms_yes` counts the 14 symptoms
answered Yes. This is descriptive only and does not identify anyone.

| record_id | copies | age | gender_male | n_symptoms_yes | label |
|---|---|---|---|---|---|
| 14 | 16 | 30 | 1 | 0 | 0 |
| 56 | 7 | 38 | 1 | 0 | 0 |
| 109 | 7 | 46 | 1 | 4 | 0 |
| 153 | 7 | 53 | 1 | 8 | 0 |
| 5 | 6 | 27 | 1 | 0 | 0 |
| 65 | 6 | 39 | 0 | 9 | 1 |
| 75 | 6 | 40 | 1 | 2 | 0 |
| 86 | 6 | 43 | 0 | 9 | 1 |
| 90 | 6 | 43 | 1 | 5 | 0 |
| 103 | 6 | 45 | 1 | 2 | 0 |

## 3. Conflicting profiles (same 16 features, different labels)

| measure | all rows | deduplicated |
|---|---|---|
| rows_in_view | 520 | 251 |
| unique_profiles | 251 | 251 |
| profiles_with_more_than_one_row | 107 | 0 |
| conflicting_profiles | 0 | 0 |
| rows_in_conflicting_profiles | 0 | 0 |
| pct_rows_in_conflicting_profiles | 0.0 | 0.0 |
| positive_rows_in_conflicting_profiles | 0 | 0 |
| negative_rows_in_conflicting_profiles | 0 | 0 |
| minimum_rows_any_deterministic_classifier_must_misclassify | 0 | 0 |
| theoretical_max_accuracy_from_conflicts_only | 1.0 | 1.0 |

Conflicting profiles in detail (all-rows view):

_None found._

### Accuracy limit attributable to conflicting profiles

For identical feature profiles, a deterministic classifier produces the same prediction for every
row. It must therefore misclassify at least the number of rows belonging to the minority label
within each conflicting profile. Which label is misclassified depends on the model's prediction.

- All rows: at least **0** of 520 rows must be misclassified, so the theoretical maximum accuracy from this source alone is **1.0000**.
- Deduplicated: at least **0** of 251 rows, giving a ceiling of **1.0000**.

These ceilings are properties of the data. They are **not observed model results**, and they
ignore every other source of error. No model has been trained.

## 4. SQL cross-check

The counts were recomputed with SQL `GROUP BY … HAVING` in a temporary in-memory SQLite database,
once on the cleaned values and once on the literal raw strings.

| query | result |
|---|---|
| clean_exact_dups(groups, rows_in_groups, redundant) | (107, 376, 269) |
| raw_text_exact_dups(groups, rows_in_groups, redundant) | (107, 376, 269) |
| clean_conflicts(profiles, rows, minority_rows) | (0, 0, 0) |
| raw_text_conflicts(profiles, rows, minority_rows) | (0, 0, 0) |

Pandas result: duplicates (107, 376, 269), conflicts (0, 0, 0). **SQL matches pandas: True.**

## 5. Duplicate policy (approved 7 Nov 2026; see docs/PROJECT_PLAN.md §3.5)

This section was corrected on 7 Nov 2026. The 6 Nov version proposed the opposite assignment (deduplicated data as the primary analysis). It is archived in `archive/day1_pre_correction_2026-11-07/reports/02_duplicate_audit.md`. The audit counts above are unchanged.

- **Primary analysis:** all 520 original records are retained.
- **Sensitivity analysis:** the 251 unique questionnaire profiles, one record per profile. No profile is conflicting, so the 251 unique records and the 251 unique profiles are the same set.
- **Both analyses:** use the same frozen split by feature profile, and keep conflicting profiles in (none exist).
- No percentage threshold is used to decide removal. The earlier planning rule that would have made deduplicated data primary when duplicates exceeded 5% is withdrawn.
