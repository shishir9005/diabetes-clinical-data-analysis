-- Training-only SQL for Day 1.
-- The database holds the training partition only (table train_rows); test rows were never loaded.
-- Each query starts with a "-- name:" line so src/s04_sql_train.py can run and report it.

-- name: setup_views
CREATE VIEW train_dedup AS
    SELECT * FROM train_rows WHERE is_first_copy = 1;
CREATE VIEW train_both AS
    SELECT 'all_rows' AS view, * FROM train_rows
    UNION ALL
    SELECT 'deduplicated' AS view, * FROM train_dedup;

-- name: row_counts
SELECT view, COUNT(*) AS n_rows, COUNT(DISTINCT profile_id) AS n_profiles
FROM train_both
GROUP BY view
ORDER BY view;

-- name: class_counts
SELECT view,
       CASE label WHEN 1 THEN 'Positive' ELSE 'Negative' END AS class,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY view), 2) AS pct_of_view
FROM train_both
GROUP BY view, label
ORDER BY view, label DESC;

-- name: symptom_prevalence_by_class
-- Percentage answering Yes, by class.
SELECT view,
       CASE label WHEN 1 THEN 'Positive' ELSE 'Negative' END AS class,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(polyuria), 1)           AS polyuria,
       ROUND(100.0 * AVG(polydipsia), 1)         AS polydipsia,
       ROUND(100.0 * AVG(sudden_weight_loss), 1) AS sudden_weight_loss,
       ROUND(100.0 * AVG(weakness), 1)           AS weakness,
       ROUND(100.0 * AVG(polyphagia), 1)         AS polyphagia,
       ROUND(100.0 * AVG(genital_thrush), 1)     AS genital_thrush,
       ROUND(100.0 * AVG(visual_blurring), 1)    AS visual_blurring,
       ROUND(100.0 * AVG(itching), 1)            AS itching,
       ROUND(100.0 * AVG(irritability), 1)       AS irritability,
       ROUND(100.0 * AVG(delayed_healing), 1)    AS delayed_healing,
       ROUND(100.0 * AVG(partial_paresis), 1)    AS partial_paresis,
       ROUND(100.0 * AVG(muscle_stiffness), 1)   AS muscle_stiffness,
       ROUND(100.0 * AVG(alopecia), 1)           AS alopecia,
       ROUND(100.0 * AVG(obesity), 1)            AS obesity
FROM train_both
GROUP BY view, label
ORDER BY view, label DESC;

-- name: gender_by_class
SELECT view,
       CASE gender_male WHEN 1 THEN 'Male' ELSE 'Female' END AS gender,
       COUNT(*) AS n,
       SUM(label) AS positive,
       COUNT(*) - SUM(label) AS negative,
       ROUND(100.0 * AVG(label), 1) AS pct_positive
FROM train_both
GROUP BY view, gender_male
ORDER BY view, gender;

-- name: age_band_by_class
SELECT view,
       CASE WHEN age < 30 THEN '1: <30'
            WHEN age < 40 THEN '2: 30-39'
            WHEN age < 50 THEN '3: 40-49'
            WHEN age < 60 THEN '4: 50-59'
            ELSE '5: >=60' END AS age_band,
       COUNT(*) AS n,
       SUM(label) AS positive,
       COUNT(*) - SUM(label) AS negative,
       ROUND(100.0 * AVG(label), 1) AS pct_positive
FROM train_both
GROUP BY view, age_band
ORDER BY view, age_band;

-- name: age_summary_by_class
SELECT view,
       CASE label WHEN 1 THEN 'Positive' ELSE 'Negative' END AS class,
       COUNT(*) AS n,
       MIN(age) AS min_age,
       MAX(age) AS max_age,
       ROUND(AVG(age), 1) AS mean_age
FROM train_both
GROUP BY view, label
ORDER BY view, label DESC;

-- name: duplicate_check_within_training
SELECT COUNT(*) AS duplicate_groups,
       COALESCE(SUM(n), 0) AS rows_in_duplicate_groups,
       COALESCE(SUM(n - 1), 0) AS redundant_copies
FROM (
    SELECT COUNT(*) AS n
    FROM train_rows
    GROUP BY age, gender_male, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia,
             genital_thrush, visual_blurring, itching, irritability, delayed_healing,
             partial_paresis, muscle_stiffness, alopecia, obesity, label
    HAVING COUNT(*) > 1
);

-- name: conflicting_profiles_within_training
SELECT COUNT(*) AS conflicting_profiles
FROM (
    SELECT 1
    FROM train_rows
    GROUP BY age, gender_male, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia,
             genital_thrush, visual_blurring, itching, irritability, delayed_healing,
             partial_paresis, muscle_stiffness, alopecia, obesity
    HAVING COUNT(DISTINCT label) > 1
);
