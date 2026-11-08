# Project Plan — Diabetes Clinical Data Analysis

Version 1.8 · 9 November 2026 (earlier versions of this plan are kept locally in an unpublished `archive/` folder)

| Item | Status |
|---|---|
| Day 1 (setup, data validation, audits, frozen split, training-only SQL/EDA) | Approved and executed 6 Nov 2026. **Corrected and documented 7 Nov 2026 (§7)** |
| Duplicate policy (§3.5) | **Approved 7 Nov 2026**: primary = all 520 records, sensitivity = 251 unique profiles |
| Streamlit disclaimer and prediction-output rule (§3.13) | **Approved wording recorded 7 Nov 2026**; shown by the app |
| Model training and evaluation (§3.10–§3.12) | **Approved 7 Nov 2026** by the project owner; day-based milestones dropped |
| Notebook and README | **Approved and done 8 Nov 2026** (`notebooks/clinical_diabetes_analysis.ipynb`, `README.md`; §7 v1.5) |
| Streamlit app (local) | **Approved and built 8 Nov 2026** (`app/`, `tests/`; §3.13, §7 v1.6) |
| Release preparation | **Decided 8 Nov 2026**: MIT for the code; row-level data, split and fold files, databases, private reports and the archive stay unpublished (§8 items 3 and 6). The first commit is the owner's step |
| Demo model for the app and publication identity | **Decided 9 Nov 2026**: the app uses a separate, row-free Logistic Regression; the research-selected RBF SVM stays unpublished (§3.13, §7 v1.8) |
| Deployment | **Not approved yet** — needs explicit approval |

---

## 0. Day 1 stop conditions

On Day 1 the project will not:

- train any machine-learning or neural-network model;
- implement the FNN;
- create the Streamlit application;
- deploy or publish anything;
- connect a remote Git repository, push, or commit;
- install packages beyond the approved Day 1 list (numpy, pandas, scikit-learn, matplotlib, seaborn, jupyterlab, streamlit, joblib, plus whatever pip installs as their dependencies);
- alter any existing project, environment or notebook.

No step is reported as complete unless it was executed and verified.

---

## 1. Workspace boundary

- All project files and generated reports stay inside the project folder.
- The only planned exception is a dedicated Conda environment, `diabetes-ml` (Python 3.12), created for this project. Conda also records every new environment in its own environment list. This is reported, not hidden.
- Tool caches are kept out of shared locations:
  - Conda's package-download cache goes to the session's temporary scratch directory (via `CONDA_PKGS_DIRS`), not to the shared base environment.
  - pip runs with `--no-cache-dir`.
  - Matplotlib's font cache goes to the scratch directory (via `MPLCONFIGDIR`).
- Not modified: other Conda environments, existing repositories, notebooks, research folders, and personal folders and settings.
- These were fingerprinted before Day 1 started and are re-checked at the end: the package lists of the existing environments, the shared base environment's listings, the Conda environment registry, the Matplotlib, Streamlit and Jupyter configuration folders, the pip cache, and the listing of a neighbouring project folder. The fingerprints are in a private environment report that is not published, because it contains local machine details.
- Every Python command calls the new environment's interpreter by its full path, so nothing can install into or run from `base` by accident.

### Disk-space preflight

- Before the project was created or any package installed, available space was checked again. The rule is to stop below 10 GiB.
- The external SSD is not assumed to be mounted.
- The result is recorded in the private environment report.

---

## 2. Dataset

- **UCI Machine Learning Repository, "Early Stage Diabetes Risk Prediction" (dataset id 529).**
- Only the official UCI archive is downloaded.
- The source URL, download timestamp, licence and citation are copied from the UCI page into `docs/DATA_SOURCE.md` at download time.
- The UCI documentation's claims (rows, columns, values, missingness) are **treated as unverified** until checked against the actual CSV (§3.2).

---

## 3. Methodology

### 3.1 Integrity safeguards

- The downloaded archive and the extracted CSV are never edited.
- SHA-256 hashes of the archive and CSV are recorded straight after download. They are **verified again after all processing**.
- Raw files are made read-only (`chmod a-w`). This is an integrity safeguard that prevents accidental edits. It does **not** make files immutable: the owner can still change permissions.
- Python is run with `-I -B` (isolated mode, no bytecode files) when reading downloaded data. This is import hygiene, so modules can't be loaded from a data folder by accident. It does **not** protect files from modification.
- The frozen split files (§3.7) are hashed and made read-only in the same way.

### 3.2 Validation of the original file

The following are checked against the actual file:

- dimensions and exact column names, including stray whitespace;
- the set of distinct values in each column;
- Age data type and range;
- missingness: real NaN, empty strings, and placeholder tokens such as `?`, `NA` or `-`;
- spelling and case variants;
- the target values.

Any mismatch with the documentation is reported. It is not silently fixed.

### 3.3 Cleaning

- Cleaning is deterministic recoding only, with no learned parameters: Yes/No → 1/0, Male → 1 (Female → 0, column `gender_male`), Positive → 1, and snake_case column names.
- The recoding refuses any value outside the expected set, rather than guessing.
- No rows are removed.
- Each row gets three identifiers:
  - `row_id`: the original file order;
  - `record_id`: identical across all 17 columns;
  - `profile_id`: identical across the 16 features.

### 3.4 Duplicate and conflicting-profile audit

This runs on the full dataset, before the split is established (permitted; see §3.6).

| Check | Definition | Reported |
|---|---|---|
| Exact duplicate records | All 16 features and the label identical | Rows in duplicate groups; redundant copies (rows − unique records); unique records; % of total; group-size distribution; class split |
| Conflicting profiles | 16 features identical, labels differ | Number of profiles; rows involved; per-label counts; % |

- The counts are cross-checked with SQL `GROUP BY … HAVING COUNT(*) > 1` in a temporary in-memory SQLite database.
- **Wording.** The dataset has no patient identifier, visit date or record identifier. Identical rows are reported as *identical questionnaire records*. They are never called repeated patients, because there is no evidence either way. An identical row could be:
  - the same person entered more than once;
  - a copy made while compiling the data;
  - different people who gave the same answers.
- **Accuracy limit from conflicting profiles.** For identical feature profiles, a deterministic classifier produces the same prediction for every row. It must therefore misclassify at least the number of rows belonging to the minority label within each conflicting profile. Which label is misclassified depends on the model's prediction.
- **Theoretical ceiling.**
  - The maximum achievable accuracy attributable to conflicting profiles is 1 − (Σ over conflicting profiles of minority-label rows) / N.
  - It is computed for the all-rows view and the deduplicated view.
  - It is a property of the data, not an observed model result, and is never presented as model performance.
  - It ignores every other source of error, so real model accuracy can be well below it.

### 3.5 Duplicate policy — approved 7 Nov 2026

This is the project owner's decision of 7 Nov 2026. It replaces the v1.1 proposal, which made the deduplicated data the primary analysis.

- **Primary analysis: all 520 original records retained.** The data has no patient identifier, so identical questionnaire records cannot be assumed to be accidental copies (§3.4).
- **Sensitivity analysis: the 251 unique questionnaire profiles.** One record is kept for each distinct 16-feature profile. The audit found no conflicting profiles, so the 251 unique records and the 251 unique profiles are the same set.
- **Split by profile (both analyses).** Identical 16-feature profiles never appear on both sides of a train/test or train/validation boundary. Both analyses use the same frozen split and the same model settings.
- **Conflicting profiles are kept in both analyses.** They are not dropped or relabelled. None were found in this file.
- **Nothing is deleted or removed automatically.** Both views stay available in `cleaned_all_rows.csv` and `train_rows.csv`; `is_first_copy == 1` marks the sensitivity view. Identical records are never dropped from the primary analysis automatically, and no percentage threshold triggers removal.
- **Withdrawn:** the earlier planning rule that would have made the deduplicated data primary when duplicates exceeded 5%. (v1.1 of this plan already stated that no percentage threshold decides removal.)
- **Terminology.** In the Day 1 reports and figures, "all rows" is the primary view and "deduplicated" is the sensitivity view.
- **Wording.** Identical questionnaire records and profiles are not described as confirmed repeated patients: the file has no patient identifier (§3.4).

The frozen split (§3.7) is defined at profile level on the full data, so it serves both analyses. Generating it on Day 1 did not prejudge this policy.

### 3.6 Test-set isolation

Before the split is established, the following may read the full dataset:

- full-file integrity validation;
- duplicate auditing;
- profile identification;
- frozen split generation.

After the split is frozen, test records must not be used for:

- EDA or feature exploration;
- feature engineering or feature selection;
- preprocessing decisions;
- hyperparameter tuning or model selection.

The test set is accessed for the predeclared final evaluation only. Its required integrity and split-count reports are permitted.

In practice:

- The split script writes a training-only file, `data/processed/train_rows.csv`.
- The SQL and EDA scripts read only that file. They never open the full cleaned file or the split assignment.
- The SQLite database contains training rows only.

### 3.7 Frozen grouped split and CV folds

**Test split**

- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on the full cleaned data (all rows), with `groups = profile_id` and `y = label`.
- Fold 0 is predeclared as the test set (about 20%). The rest is training.

**CV folds: 15 in total**

- 3 repeats × 5-fold `StratifiedGroupKFold` on the training partition, with `random_state` = 0, 1 and 2 (owner decision, 7 Nov 2026).
- The original freeze on 6 Nov used seeds 100, 101 and 102. Those folds were archived and replaced on 7 Nov. Only the CV folds changed: the train/test split was not regenerated, and `split_assignment.csv` is byte-identical (SHA-256 `8f0077e5…`).
- Fold membership is assigned per `profile_id`, so the same folds apply to the sensitivity (deduplicated) view.
- The "15 grouped CV folds" count comes from the 6 Nov review. It is implemented as 3 × 5 repeated grouped CV. The 5 Nov draft said a single 5-fold CV.

**Acceptance checks**

| Partition | Checks |
|---|---|
| Each of the 15 CV folds, in both the all-rows and deduplicated views | Both classes present in the training partition; both classes present in the validation partition; no profile group in both partitions |
| Each repeat | Every training row appears in exactly one validation fold |
| Test split | Both classes present on each side; no profile group shared; no test row inside any CV fold |

- All fold counts and class distributions are reported.
- **If any check fails, the work stops and reports the failure.** The splitting method and rules are not silently changed.
- Freezing means:
  - `split_assignment.csv`, `cv_folds.csv` and `split_config.json` are written once;
  - they are hashed into `reports/sha256_manifest.txt`;
  - they are made read-only.
- Exception: `cv_folds.csv` and `split_config.json` were written a second time on 7 Nov 2026 under the controlled correction (§7). The replaced versions are archived with their hashes.

### 3.8 SQL (training rows only)

Python's built-in `sqlite3`. The database `data/processed/train_only.sqlite` holds training rows only.

Queries:

- class counts;
- symptom rates by class;
- age band × class;
- gender × class;
- a duplicate check within training.

### 3.9 EDA (training rows only)

- Class balance, age by class, symptom rates by class, and gender × class.
- Each plot is shown for the all-rows (primary) and deduplicated (sensitivity) views of the training partition.

### 3.10 Models and evaluation — approved 7 Nov 2026

| Model | Search | Combinations |
|---|---|---|
| Majority-class baseline | `DummyClassifier(most_frequent)` | – |
| Logistic Regression | C ∈ {0.01, 0.1, 1, 10} | 4 |
| Categorical NB | alpha ∈ {0.1, 0.5, 1.0}; Age in fixed bands (<30, 30–39, 40–49, 50–59, ≥60); `min_categories` set | 3 |
| RBF SVM | C ∈ {0.1, 1, 10} × gamma ∈ {scale, 0.01, 0.1} | 9 |
| Random Forest | 300 trees; max_depth ∈ {None, 5, 10} × min_samples_leaf ∈ {1, 3} | 6 |

**Preprocessing**

- Done inside scikit-learn pipelines, fitted on training folds only.
- Age is standardised for Logistic Regression and SVM. Binary features pass through unchanged.
- Random Forest uses raw features.

**Tuning and selection**

- Tuning uses the frozen 15 folds and is scored by ROC-AUC.
- The model used for the demo is the one with the best mean CV ROC-AUC. A simpler model is preferred if it is within one CV standard deviation. The test score is never used for selection. *(Update 9 Nov 2026: this selects the research model. The app's demo model is a separate Logistic Regression, §3.13.)*

**Test-set metrics**

- Reported once, after all choices are fixed: accuracy, precision, recall, specificity, F1, ROC-AUC and the confusion matrix.
- **Baseline ROC-AUC is 0.5, not n/a.** `DummyClassifier(most_frequent)` gives every row the same score, so every Positive–Negative pair is tied and the area under the ROC curve is 0.5 (ties count as half). ROC-AUC would be undefined only if the evaluated set contained a single class. The acceptance checks (§3.7) guarantee both classes in every validation fold and in the test set, in both views. This is a wording correction only: the baseline and the metric definitions are unchanged.
- The positive class is the dataset label "Positive".
- The threshold is 0.5, or decision value 0 for SVM, with no threshold tuning.
- CV mean ± SD is reported alongside.

**Saved pipeline**

- One verified scikit-learn pipeline is saved with joblib.
- The check: the reloaded pipeline gives identical test-set probabilities.
- The scikit-learn version is pinned.
- *(Update 9 Nov 2026: this pipeline is the research-selected RBF SVM. It stores training records as support vectors, so it is kept on the owner's computer and is not published. The app uses a separate demo-only Logistic Regression, §3.13.)*

### 3.11 Bootstrap intervals — optional

If they are reported:

- **Method:** group bootstrap on the frozen test set, resampling profile groups with replacement. The number of repetitions is stated.
- **What they cover:** uncertainty conditional on the fitted model and the frozen test split. They do not capture all training-split or model-selection uncertainty.
- **Undefined repetitions:** for each metric, the number of repetitions where it is undefined is recorded. Examples: ROC-AUC when a resample contains only one class; precision when there are no positive predictions; specificity when there are no negatives. Undefined repetitions are not silently discarded, and no interval is reported without disclosing this count.

### 3.12 Optional FNN — approved 7 Nov 2026 as an optional script

- It is a script (`src/s10_fnn.py`), not a notebook. PyTorch is installed only if this is attempted.
- It uses the same frozen split and the training-fitted scaler.
- It is averaged over several seeds.
- It is added to the comparison table but never deployed.

### 3.13 Streamlit — approved 8 Nov 2026 (local app; deployment not approved)

**Layout**

- A single lightweight app with five pages (`st.navigation`): Overview, Exploratory analysis, Model comparison, Evaluation, and Educational prediction. This replaces the earlier five-tab sketch.
- It only reads saved aggregate files (`reports/`, `reports/figures/`, `models/`). No backend, no database, no retraining, no row-level data, no external calls, and it writes nothing.
- Streamlit's usage statistics, first-run email prompt and developer menu are switched off in `.streamlit/config.toml`, and the server listens on `localhost` only (Streamlit's default is every network interface).

**Demo**

- It uses one verified scikit-learn pipeline: a separate, demo-only Logistic Regression (`models/demo_logistic_regression.joblib`, decided 9 Nov 2026), **not** the research-selected RBF SVM. An RBF SVM stores its support vectors (127 of the 416 training rows) inside its saved file, so `models/primary_demo_pipeline.joblib` is kept on the owner's computer and is not published. The demo model is the Logistic Regression configuration that cross-validation chose for the primary view (`C = 0.1`), refitted to the 416 training rows by `src/s12_demo_model.py`: no new tuning, no test row, and no change to any research result. The serialised bytes are inspected before the file is written (the classes they refer to, every array and attribute, and a search for every training row in every stored array), and the tests repeat that inspection on the published file.
- The app compares the file with its recorded SHA-256 before loading it, refuses a model that is not a Logistic Regression or that has attributes storing training records, and refuses to predict if the saved input columns differ from the ones it sends.
- Inputs are validated (all 16 answers are required; age 16 to 90, the range in the dataset) and are clearly labelled as made-up examples. The app tells visitors not to enter personal health information. Nothing is stored or logged.
- Only the predicted class is displayed.

**Disclaimer (approved wording, recorded 7 Nov 2026)**

The project owner approved this exact wording after reviewing the Day 1 correction report. The app must display it as written, without paraphrase:

> This application is for educational purposes only. It is not a medical device and does not provide a diagnosis or medical advice. Its predictions should not be used to make healthcare decisions. Please consult a qualified healthcare professional if you have concerns about diabetes.

This replaces the v1.1 placeholder wording, which is kept in the archived v1.1 plan. The sentence from the original project brief could not be recovered from the project files, so the approved wording above is the one to use.

**Prediction-output rule**

- The educational app may display the predicted class.
- Do not display numeric prediction probabilities unless a probability-calibration procedure has been implemented and independently evaluated.
- Never describe the output as a medical diagnosis.
- The reload-consistency check in §3.10 compares probabilities internally. That is a test of the saved pipeline, not something the app displays.
- The demo Logistic Regression could produce probabilities, but the app only calls `predict()`. A test checks that the app code never uses `predict_proba`, `predict_log_proba` or `decision_function`.

---

## 4. Project structure

The working copy is laid out as below. Items marked "not published" stay on the owner's computer (§8 item 3).

```
Diabetes_Clinical_Data_Analysis/
├── README.md                    # overview, results, how to run the app and reproduce
├── LICENSE                      # MIT licence for the code and documentation (not the dataset)
├── requirements.txt             # app packages, pinned
├── requirements-analysis.txt    # adds the notebook and analysis scripts
├── requirements-fnn.txt         # adds PyTorch for the optional neural network
├── app/                         # Streamlit app: streamlit_app.py, views/ (five pages), app_common.py, demo_model.py
├── .streamlit/config.toml       # telemetry off, error details off, localhost only, theme
├── tests/test_app.py            # tests for the app, the demo model and the release rules
├── docs/
│   ├── PROJECT_PLAN.md          # this file
│   └── DATA_SOURCE.md           # URL, timestamp, licence, citation
├── data/                        # not published (row-level)
│   ├── raw/                     # official UCI zip and CSV (read-only)
│   ├── processed/               # cleaned_all_rows.csv, train_rows.csv, train_only.sqlite
│   └── splits/                  # split_assignment.csv, cv_folds.csv, split_config.json (frozen)
├── sql/train_queries.sql
├── src/                         # prep.py (shared), Day 1 stage scripts, correction scripts (s03b, s07), modelling scripts (mlcommon, s08–s11), s12_demo_model.py (the app's demo model), rebuild_processed_data.py
├── reports/                     # numbered reports, model result tables, sha256_manifest.txt, figures/ (the environment reports and Day 1 working notes are not published)
├── archive/                     # not published: earlier versions of replaced files
├── notebooks/                   # clinical_diabetes_analysis.ipynb (executed)
├── models/                      # demo_logistic_regression.joblib and .txt (published); primary_demo_pipeline.joblib and .txt (research RBF SVM, not published)
└── .gitignore                   # what is published and what is not
```

---

## 5. Schedule

**Superseded 7 Nov 2026:** the project owner dropped day-based milestones (§7 v1.4). The table is kept as a record of the original plan.

| Day | Essential | Optional (not guaranteed) | Gate |
|---|---|---|---|
| 1 | Environment and smoke tests; download; validation; duplicate and conflicting-profile audit; frozen split and 15 CV folds; training-only SQL and EDA; reports | – | Final Day 1 review after the 7 Nov correction (duplicate policy approved 7 Nov) |
| 2 | Baseline + 4 classical models on the frozen folds; one test evaluation; primary analysis (all 520 records) and sensitivity analysis (251 unique profiles); saved, verified pipeline | Leakage-prone split illustration; bootstrap intervals | Approval |
| 3 | Complete notebook and README; local Streamlit app | PyTorch FNN (time-boxed) | Approval |
| 4 | Buffer; clean re-run; final review | Public deployment (user pushes and deploys); model-interpretation extras; extra plots | – |

---

## 6. Day 1 sequence (approved)

1. Set up the project directory and the new Conda environment.
2. Run the environment smoke tests: imports and versions only, with no model fitting.
3. Download only the official UCI dataset archive.
4. Validate and audit the original data.
5. Produce the duplicate and conflicting-profile reports.
6. Generate and freeze the grouped train/test split and CV folds.
7. Run SQLite queries on training rows only.
8. Perform training-only EDA.
9. Prepare the Day 1 reports. Re-verify raw-file hashes and environment fingerprints. List every file created.

Then stop for review. (Executed 6 Nov 2026; corrected 7 Nov 2026, see §7.)

---

## 7. Change log

**v1.1 (6 Nov 2026), from the user's review:**

1. The earlier conflicting-profile wording was replaced with the reviewer's corrected statement. A separate theoretical-ceiling calculation was added (§3.4).
2. CV acceptance checks were added for all 15 grouped folds, with stop-on-failure (§3.7).
3. Rules were added for interpreting bootstrap intervals and disclosing undefined repetitions (§3.11).
4. Test-set isolation wording was clarified (§3.6). SQL and EDA were moved to training rows only.
5. The workspace boundary was set to this folder. The new Conda environment is the only planned exception (§1).
6. The disk-space preflight was added, with a stop below 10 GiB and no assumption that the SSD is mounted. Integrity safeguards are described without claims of immutability (§1, §3.1).
7. Split generation moved from Day 2 to Day 1, so SQL and EDA can be restricted to training rows.
8. The project folder was renamed from `diabetes-questionnaire-ml` to `Diabetes_Clinical_Data_Analysis`.

**v1.2 (7 Nov 2026), controlled Day 1 correction:**

1. **CV seeds.** The 15 CV folds were regenerated with seeds 0, 1 and 2 (6 Nov: 100, 101, 102). The train/test split was not regenerated or changed (`split_assignment.csv` SHA-256 `8f0077e5…`; `train_rows.csv` unchanged). The old folds, config and reports are kept locally in the unpublished archive (§3.7).
2. **Duplicate policy approved (§3.5).** Primary analysis: all 520 original records. Sensitivity analysis: the 251 unique questionnaire profiles. This reverses the v1.1 proposal, and the earlier 5% planning rule is withdrawn. `reports/02_duplicate_audit.md` §5 and the Day 1 summary were corrected to match.
3. **Baseline ROC-AUC wording (§3.10).** "n/a" was replaced by 0.5, with the reason and the one condition under which it would be undefined.
4. **Git.** A local repository was initialised with a `.gitignore`. No remote, no push, no commit.
5. **Streamlit disclaimer.** The original verbatim sentence cannot be recovered from the project files. Documented as unresolved (§3.13, §8).
6. **Reports and manifest.** Reports 03 and 06 were regenerated, report 07 was added, the Day 1 summary was rewritten and `reports/sha256_manifest.txt` was regenerated. Every replaced file is archived locally, with an index.

**v1.3 (7 Nov 2026), final Day 1 documentation pass:**

1. **Streamlit disclaimer (§3.13).** The owner's approved wording is recorded verbatim. It replaces the v1.1 placeholder.
2. **Prediction-output rule (§3.13).** The app may display the predicted class; no numeric probability unless calibration has been implemented and independently evaluated; the output is never described as a diagnosis.
3. **Approval wording.** `src/s02_clean_and_audit.py`, `src/s05_eda_train.py` and `reports/05_eda_training.md` now state the approved duplicate policy instead of the 6 Nov approval-pending wording. The scripts were edited, not re-run: no data, split, fold, database or figure changed.
4. **§3.5.** Explicit statements added: no automatic removal from the primary analysis, and identical questionnaire records are not described as confirmed repeated patients.
5. **Git review (read-only).** A first-commit list was proposed in a private working note. Nothing was staged or committed.
6. **§8** updated: items 1, 2 and 5 are resolved; items 3 and 4 remain open. The pre-edit files are archived locally.

**v1.4 (7 Nov 2026), model training and evaluation:**

1. **Approval.** The project owner approved model training and evaluation (§3.10–§3.12) and dropped day-based milestones. The notebook, README, Streamlit app and deployment are not approved yet.
2. **Scripts.** `src/mlcommon.py` (shared helpers), `src/s08_cv_select.py` (selection on training data only), `src/s09_test_eval.py` (one test evaluation and the demo pipeline) and `src/s11_build_report.py` (report and figures).
3. **Selection rule, declared before any result existed.** It is applied separately inside each view, with identical grids, folds and rule. Within a model, the configuration with the highest mean CV ROC-AUC is taken (exact ties: the first listed). Across the four models, every model whose mean CV ROC-AUC is at least the best mean minus the SD of the best model's 15 fold values is eligible, and the simplest eligible model in the fixed order Logistic Regression < Categorical NB < RBF SVM < Random Forest is chosen. The primary view's choice was the demo model (superseded for the app by v1.8: the app now uses a separate, demo-only Logistic Regression); the sensitivity view's choice is for comparison only. The test set was evaluated once per view and changed nothing.
4. **Outputs.** `reports/model_cv_results.csv`, `reports/model_selection.json` and `.sha256`, `reports/model_test_results.json` and `.csv`, `reports/09_model_results.md`, `reports/figures/fig05_cv_roc_auc_by_model.png`, `reports/figures/fig06_test_roc_curves.png`, `models/primary_demo_pipeline.joblib` and `.txt`.
5. **Result.** The rule chose the RBF SVM in both views (details in `reports/09_model_results.md`).
6. **Housekeeping.** `.gitignore` now excludes the private environment reports.
7. **Optional FNN (§3.12).** Run as `src/s10_fnn.py` after the classical results were final: a small feed-forward network, seeds 0 to 4. The epoch count is chosen by early stopping on a group-separated inner split, then a fresh network is retrained on the whole training data for that fixed count. Runs are seeded, CPU-only and deterministic. PyTorch 2.15.0 (CPU) was installed into `diabetes-ml` for this step only, with its dependencies (filelock, fsspec, mpmath, networkx, sympy); numpy, pandas, scikit-learn, scipy, joblib and matplotlib kept their versions. The FNN took no part in model selection and is not deployed. Outputs: `reports/fnn_cv_results.csv` and `reports/fnn_results.json`.

**v1.5 (8 Nov 2026), notebook and README:**

1. **Approval.** The project owner approved the notebook and README. The Streamlit app and deployment are not approved yet.
2. **Notebook.** `notebooks/clinical_diabetes_analysis.ipynb`, saved with its outputs. It uses the training rows only, displays the saved test and FNN results without reading any test row, recomputes the cross-validation scores of the four chosen configurations and checks that they equal the saved values, and contains no local paths. It was executed with temporary Jupyter, IPython and matplotlib directories.
3. **README.** `README.md`: headline results copied from `reports/09_model_results.md`, the evaluation safeguards, reproduce steps, data citation, limitations and status.
4. **Requirements.** `requirements.txt` was replaced by a clean pinned list. The previous file was a `pip freeze` that contained a line that cannot be installed on another machine; it is archived unchanged in the unpublished archive. Torch stays optional.
5. **Rebuild helper.** `src/rebuild_processed_data.py` recreates `data/processed/` from the raw download and the frozen split assignment, checks the SHA-256 of its inputs and outputs, and refuses to overwrite. A test run into a temporary folder gave byte-identical copies of the frozen files.
6. **Source record.** `docs/DATA_SOURCE.md` now states what UCI does not say (how the label was determined, how patients were selected) and carries the full citation of the introductory paper, taken from the Crossref record for its DOI.
7. **Unchanged.** Raw data, frozen inputs, the cross-validation and test results, the selection file and its hash, the figures and the demo pipeline.

**v1.6 (8 Nov 2026), Streamlit app and release preparation:**

1. **Approval.** The project owner approved the local Streamlit app and the preparation of the project for public release. Deployment, committing and publishing are not approved.
2. **App.** `app/` (entry `streamlit_app.py`, five pages in `views/`, helpers `app_common.py` and `demo_model.py`), `.streamlit/config.toml` and `tests/test_app.py`. The app shows the saved tables and figures, and an educational prediction form for the saved RBF SVM pipeline (§3.13; replaced by a demo-only Logistic Regression in v1.8). Nothing was retrained.
3. **Requirements.** `requirements.txt` now lists only what the app needs (numpy, pandas, scipy, scikit-learn, joblib and streamlit, all pinned). `requirements-analysis.txt` adds Matplotlib, JupyterLab and ipykernel for the notebook and scripts, and `requirements-fnn.txt` adds PyTorch for the optional neural-network experiment.
4. **What is published (§8 item 3).** `.gitignore` now keeps row-level data (`data/`), databases, private environment reports, Day 1 working notes and the archive out of Git. This plan was cleaned of local paths and machine details, and its references to private records were reworded. The unpublished files stay on the owner's computer and are still listed in `reports/sha256_manifest.txt`.
5. **Consequence for reproducibility.** `src/rebuild_processed_data.py`, the analysis scripts and the notebook need `data/processed/` and `data/splits/`, which are not published. They run in the working copy. A clone can run the app, read the executed notebook and the reports, and inspect the code (see the README).
6. **Unchanged.** Raw data, frozen inputs, the cross-validation and test results, the selection file and its hash, the figures and the demo pipeline.

**v1.7 (8 Nov 2026), publication review:**

1. **Decisions.** The project owner chose the MIT licence for the code and confirmed that raw and processed row-level data, split and fold assignment files, databases, private environment reports and local working notes are not published.
2. **Changes.** `LICENSE` (MIT) and a Licence section in the README that keeps the UCI dataset's CC BY 4.0 attribution separate from the code licence; the notebook's reproduce note names `requirements-analysis.txt`; the README, this plan and `.gitignore` no longer describe these choices as open; tests were added for the licence, for the app's imports against `requirements.txt`, and for the README's numbers against the report.
3. **Unchanged.** Frozen data, folds, the test assignment, model results and the saved model.

**v1.8 (9 Nov 2026), publication identity and a row-free demo model:**

1. **Decisions.** The project owner chose the GitHub identity (username `shishir9005`, commit author `SHISHIR MD SHIFAT HOSSAIN`, and the GitHub no-reply address, set in this repository's local Git configuration only) and kept the rule that no row-level data is published, which includes records stored inside a model file. The RBF SVM stores 127 of the 416 training rows as support vectors, so the app needs a different model.
2. **Demo model.** `src/s12_demo_model.py` fits the Logistic Regression configuration that cross-validation chose in the primary view (`C = 0.1`) on the 416 training rows, inspects the serialised bytes, and writes `models/demo_logistic_regression.joblib` (3,505 bytes) and `.txt` only if the inspection passes. The app (`app/demo_model.py`, `app/views/prediction.py`) loads it and refuses any model that is not a Logistic Regression or that carries attributes storing training records. The research-selected RBF SVM (`models/primary_demo_pipeline.*`) is unchanged, stays on the owner's computer and is excluded from Git (`.gitignore`).
3. **Documentation.** The README (a "Models in this project" section and the author link `https://github.com/shishir9005`), the notebook's note on the saved pipeline, §3.10, §3.13, §4 and §8 item 3 of this plan, and the app's Overview, Model comparison and Educational prediction pages now say that the app's model is the demo-only Logistic Regression and not the selected SVM. `reports/09_model_results.md`, the rule text in `reports/model_selection.json`, the `demo_pipeline` entry in `reports/model_test_results.json` and the scripts that write them are hash-pinned and were not changed. They still say "demo model" and "demo pipeline" for the research-selected RBF SVM. The README explains the wording (the RBF SVM is the research-selected model, Logistic Regression is used separately for the public Streamlit demonstration, and the SVM file stays local because it contains training records), and the Model comparison page carries the same note next to the saved rule.
4. **Tests.** New tests inspect the published model file (no support vectors, no training row in any stored array, only scikit-learn, NumPy and joblib classes), check that an SVM, or a Logistic Regression that carries records, is refused, that only the demo model file is in the publication set, that Git ignores the SVM files, that no e-mail address other than the GitHub no-reply address is published, and that the app never asks the model for probabilities.
5. **Unchanged.** Raw data, frozen inputs, splits and folds, the cross-validation and test results, the selection file and its hash, the figures, and the research-selected RBF SVM with its description.

---

## 8. Open issues (updated 9 Nov 2026)

1. **Streamlit disclaimer wording — resolved 7 Nov 2026.** The owner approved the exact wording, which is recorded in §3.13. The sentence from the original project brief still cannot be recovered from the project files; the approved wording supersedes it.
2. **Outdated approval wording — resolved 7 Nov 2026.** `src/s02_clean_and_audit.py`, `src/s05_eda_train.py` and `reports/05_eda_training.md` were edited to state the approved duplicate policy (§3.5). The scripts were not re-run, and no data, split, fold, database or figure changed.
3. **What goes into Git — decided 8 Nov 2026.** Published: code, tests, the app, the executed notebook, aggregate reports and figures, the demo-only Logistic Regression (`models/demo_logistic_regression.*`), the docs and the licence. Not published, by the owner's decision: everything in `data/` (the raw download, the processed CSVs, and the split and fold assignment files), databases, private environment reports, Day 1 working notes and the archive of earlier versions. A clone can therefore run the app and read the notebook, reports and code, but the notebook and the analysis scripts can be re-run only in the working copy. The saved RBF SVM (`models/primary_demo_pipeline.*`) stores 127 support vectors, which are training rows in standardised form. The 8 Nov decision to publish it because the app needed it was reversed on 9 Nov 2026: the file stays on the owner's computer and is excluded from Git, and the app uses the row-free demo Logistic Regression instead (§3.13, §7 v1.8).
4. **A one-off timestamp change on an unrelated local folder (reported 6 Nov) — open, unresolved.** Day 1 reported a single change to the modification time of a folder outside the project, with its contents unchanged and the cause unidentified. The folder is outside the project and was not investigated or modified. The details are in the private environment report.
5. **App output rule — resolved 7 Nov 2026.** Recorded in §3.13: the app may display the predicted class; no numeric probabilities unless calibration has been implemented and independently evaluated; never described as a diagnosis.
6. **Code licence — decided 8 Nov 2026: MIT,** for the code and the documentation written for this project (`LICENSE`; the copyright holder is SHISHIR MD SHIFAT HOSSAIN). The dataset is not covered: it keeps its CC BY 4.0 licence and attribution (`docs/DATA_SOURCE.md`).
