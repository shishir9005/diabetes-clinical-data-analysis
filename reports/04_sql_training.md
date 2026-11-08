# 04 · SQLite queries (training rows only)

Database: `data/processed/train_only.sqlite`, built only from `data/processed/train_rows.csv` (416 training rows). **No test rows were loaded.**
Queries: `sql/train_queries.sql`. Each query reports two views: all training rows, and
deduplicated training rows (one copy per identical record).

## row_counts

| view | n_rows | n_profiles |
|---|---|---|
| all_rows | 416 | 202 |
| deduplicated | 202 | 202 |

## class_counts

| view | class | n | pct_of_view |
|---|---|---|---|
| all_rows | Positive | 256 | 61.54 |
| all_rows | Negative | 160 | 38.46 |
| deduplicated | Positive | 139 | 68.81 |
| deduplicated | Negative | 63 | 31.19 |

## symptom_prevalence_by_class

Percentage answering Yes (transposed for readability):

| index | all_rows · Positive (% Yes) | all_rows · Negative (% Yes) | deduplicated · Positive (% Yes) | deduplicated · Negative (% Yes) |
|---|---|---|---|---|
| polyuria | 78.9 | 5.0 | 77.7 | 4.8 |
| polydipsia | 71.9 | 4.4 | 71.2 | 4.8 |
| sudden_weight_loss | 57.0 | 15.6 | 53.2 | 15.9 |
| weakness | 69.5 | 43.1 | 69.8 | 49.2 |
| polyphagia | 56.6 | 23.8 | 56.8 | 20.6 |
| genital_thrush | 26.2 | 16.9 | 32.4 | 12.7 |
| visual_blurring | 55.5 | 30.6 | 51.1 | 30.2 |
| itching | 46.5 | 51.2 | 49.6 | 50.8 |
| irritability | 32.0 | 5.0 | 34.5 | 7.9 |
| delayed_healing | 42.2 | 43.8 | 46.0 | 44.4 |
| partial_paresis | 60.9 | 18.1 | 58.3 | 17.5 |
| muscle_stiffness | 40.2 | 31.9 | 42.4 | 31.7 |
| alopecia | 23.8 | 53.8 | 30.9 | 50.8 |
| obesity | 19.1 | 11.3 | 18.7 | 9.5 |

Group sizes: all_rows Positive n=256; all_rows Negative n=160; deduplicated Positive n=139; deduplicated Negative n=63

## gender_by_class

| view | gender | n | positive | negative | pct_positive |
|---|---|---|---|---|---|
| all_rows | Female | 153 | 136 | 17 | 88.9 |
| all_rows | Male | 263 | 120 | 143 | 45.6 |
| deduplicated | Female | 71 | 62 | 9 | 87.3 |
| deduplicated | Male | 131 | 77 | 54 | 58.8 |

## age_band_by_class

| view | age_band | n | positive | negative | pct_positive |
|---|---|---|---|---|---|
| all_rows | 1: <30 | 19 | 8 | 11 | 42.1 |
| all_rows | 2: 30-39 | 92 | 65 | 27 | 70.7 |
| all_rows | 3: 40-49 | 123 | 67 | 56 | 54.5 |
| all_rows | 4: 50-59 | 112 | 65 | 47 | 58.0 |
| all_rows | 5: >=60 | 70 | 51 | 19 | 72.9 |
| deduplicated | 1: <30 | 8 | 4 | 4 | 50.0 |
| deduplicated | 2: 30-39 | 48 | 33 | 15 | 68.8 |
| deduplicated | 3: 40-49 | 49 | 33 | 16 | 67.3 |
| deduplicated | 4: 50-59 | 55 | 39 | 16 | 70.9 |
| deduplicated | 5: >=60 | 42 | 30 | 12 | 71.4 |

## age_summary_by_class

| view | class | n | min_age | max_age | mean_age |
|---|---|---|---|---|---|
| all_rows | Positive | 256 | 16 | 90 | 48.8 |
| all_rows | Negative | 160 | 26 | 72 | 47.2 |
| deduplicated | Positive | 139 | 16 | 90 | 49.3 |
| deduplicated | Negative | 63 | 26 | 72 | 47.6 |

## duplicate_check_within_training

| duplicate_groups | rows_in_duplicate_groups | redundant_copies |
|---|---|---|
| 87 | 301 | 214 |

## conflicting_profiles_within_training

| conflicting_profiles |
|---|
| 0 |

Notes:

- These are descriptive counts within the training partition.
- They do not show causation and do not describe the general population. The data come from
  a single hospital's questionnaire sample.
