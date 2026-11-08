"""Tests for the Streamlit app.  Run them from the project folder:

    python -m unittest discover -s tests -v

They check the demo-only Logistic Regression and its inputs (and that its saved file holds no training record),
the validation rules, that the pages show the numbers in the saved reports, that the app has no code for writing
files or reaching the network, and that the files Git would publish contain no local paths, secrets, e-mail
addresses or row-level data. Nothing here writes to the project folder.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

logging.disable(logging.WARNING)  # Streamlit logs a line per cache reset outside a running server; keep the output readable

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
REPORTS = ROOT / "reports"
sys.path[:0] = [str(APP), str(ROOT / "src")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import app_common  # noqa: E402
import demo_model as dm  # noqa: E402
import mlcommon  # noqa: E402  (src/: the project's own feature list)
import prep  # noqa: E402  (src/: the project's own recoding)
import s12_demo_model as s12  # noqa: E402  (src/: fits the demo model and inspects what its saved file contains)

APPROVED_DISCLAIMER = (
    "This application is for educational purposes only. It is not a medical device and does not provide a diagnosis "
    "or medical advice. Its predictions should not be used to make healthcare decisions. Please consult a qualified "
    "healthcare professional if you have concerns about diabetes."
)
PAGES = ["overview", "exploration", "model_comparison", "evaluation", "prediction"]
MODELS = ["Majority-class baseline", "Logistic Regression", "Categorical NB", "RBF SVM", "Random Forest"]
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
ANALYSES = {"Primary: all 520 records": "primary", "Sensitivity: 251 unique profiles": "sensitivity"}

EXAMPLE_A = {"age": 45, "gender_male": "Male", **dict.fromkeys(dm.SYMPTOMS, "No")}
EXAMPLE_B = {**EXAMPLE_A, "polyuria": "Yes", "polydipsia": "Yes"}


# --- helpers -----------------------------------------------------------------------------------------------
def read_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def open_page(page: str) -> AppTest:
    """The app started from its entry file, switched to one page."""
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=90).run()
    return at.switch_page(f"views/{page}.py").run()


def report_tables(at: AppTest) -> list[pd.DataFrame]:
    """The static tables on a page, with the row-label column turned back into an ordinary first column."""
    return [t.value.reset_index() for t in at.table]


def all_text(at: AppTest) -> list[str]:
    """Every piece of text the page shows: markdown, captions, labels, help texts, metric values and tables."""
    found: list[str] = []

    def walk(node) -> None:
        for child in node.children.values():
            if hasattr(child, "children"):
                walk(child)
                continue
            value = getattr(child, "value", None)
            found.append(value.to_string() if isinstance(value, pd.DataFrame) else str(value))
            found.extend(str(getattr(child, attr)) for attr in ("label", "help") if getattr(child, attr, None))

    walk(at.main)
    walk(at.sidebar)
    return found


def fill(at: AppTest, answers: dict, generation: int = 0) -> None:
    at.number_input(key=f"answer_{generation}_age").set_value(answers["age"])
    for feature in dm.FEATURES[1:]:
        at.radio(key=f"answer_{generation}_{feature}").set_value(answers[feature])


def submit(at: AppTest) -> AppTest:
    next(button for button in at.button if button.label == "Show predicted class").click()
    return at.run()


def snapshot(folder: Path) -> dict:
    """Relative path -> (size, modification time) of every file under folder, ignoring Git and bytecode caches."""
    state = {}
    for current, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for name in files:
            path = Path(current) / name
            info = path.stat()
            state[path.relative_to(folder).as_posix()] = (info.st_size, info.st_mtime_ns)
    return state


def invented_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """40 invented rows in the 16 input columns (random numbers, nothing from the dataset) and alternating labels."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.integers(0, 2, size=(40, 16)), columns=dm.FEATURES)
    frame["age"] = rng.integers(20, 70, size=40)
    return frame, np.arange(40) % 2


def git_publishable_files() -> list[str] | None:
    """Files Git would add: tracked, plus untracked and not ignored. None if Git is unavailable."""
    try:
        result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(set(result.stdout.splitlines()))


# Saved results, formatted the way the reports format them, computed here from the JSON/CSV files and not from
# the Markdown report that the app reads.
def expected_cv(view: str) -> list[list[str]]:
    with open(REPORTS / "model_cv_results.csv", newline="") as handle:
        chosen = {r["model"]: r for r in csv.DictReader(handle) if r["view"] == view and r["selected_within_model"] == "True"}
    rows = [[f"{float(chosen[m][k + '_mean']):.3f} ± {float(chosen[m][k + '_sd']):.3f}" for k in METRICS] for m in MODELS]
    fnn = read_json("fnn_results.json")["views"][view]["cv"]
    rows.append([f"{fnn['metrics_mean'][k]:.3f} ± {fnn['metrics_sd'][k]:.3f}" for k in METRICS])
    return rows


def _results_block(view: str, model: str | None) -> dict:
    if model is None:
        return read_json("fnn_results.json")["views"][view]["test"]["seed_averaged"]
    return read_json("model_test_results.json")["views"][view]["models"][model]


def expected_test(view: str) -> list[list[str]]:
    def cell(block: dict, k: str) -> str:
        low, high = block["ci95"][k]
        return f"{block['metrics'][k]:.3f} [{low:.3f}, {high:.3f}]"
    return [[cell(_results_block(view, m), k) for k in METRICS] for m in [*MODELS, None]]


# --- the saved model -------------------------------------------------------------------------------------
class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = dm.load_pipeline()

    def test_loads_and_matches_the_recorded_hash(self):
        recorded = dm.recorded_sha256()
        self.assertEqual(dm.sha256_file(dm.MODEL_PATH), recorded)
        manifest = (REPORTS / "sha256_manifest.txt").read_text()
        self.assertIn(f"{recorded}  models/demo_logistic_regression.joblib", manifest)

    def test_model_is_the_reports_logistic_regression_configuration(self):
        classifier = self.pipeline[-1]
        params = read_json("model_selection.json")["views"]["primary"]["models"]["Logistic Regression"]["params"]
        self.assertEqual(type(classifier).__name__, "LogisticRegression")
        self.assertEqual({name: getattr(classifier, name) for name in params}, params)
        self.assertEqual(classifier.classes_.tolist(), [0, 1])
        self.assertEqual(classifier.coef_.shape, (1, 16))
        self.assertEqual([name for name in dm.RECORD_ATTRIBUTES if hasattr(classifier, name)], [])

    def test_coefficients_equal_a_fresh_fit_on_the_training_rows_only(self):
        if not (ROOT / "data" / "processed" / "train_rows.csv").is_file():
            self.skipTest("the training rows are kept on the owner's computer, not in the published repository")
        params = read_json("model_selection.json")["views"]["primary"]["models"]["Logistic Regression"]["params"]
        view = mlcommon.view_frame(mlcommon.load_training(), "primary")  # the training rows; no test row is read
        self.assertEqual(len(view), read_json("model_test_results.json")["views"]["primary"]["n_train_rows"])
        fresh = mlcommon.build_estimator("Logistic Regression", params).fit(view[dm.FEATURES], view["label"].to_numpy())
        np.testing.assert_allclose(self.pipeline[-1].coef_, fresh[-1].coef_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.pipeline[-1].intercept_, fresh[-1].intercept_, rtol=0, atol=1e-12)

    def test_input_feature_order_is_the_saved_order(self):
        saved = list(self.pipeline.feature_names_in_)
        self.assertEqual(len(saved), 16)
        self.assertEqual(dm.FEATURES, saved)
        self.assertEqual(dm.FEATURES, mlcommon.FEATURES)
        self.assertEqual(dm.FEATURES, prep.FEATURE_COLUMNS)
        listed = re.search(r"^\s*age, gender_male.*$", dm.MODEL_NOTE.read_text(), re.M).group(0)
        self.assertEqual(dm.FEATURES, [name.strip() for name in listed.split(",")])
        self.assertEqual(list(dm.encode_answers(EXAMPLE_B).columns), saved)

    def test_encoding_is_the_projects_recoding(self):
        self.assertEqual(dm.GENDER_CODES, prep.GENDER)
        self.assertEqual(dm.YES_NO_CODES, prep.YES_NO)
        self.assertEqual(dm.SYMPTOMS, prep.SYMPTOM_COLUMNS)
        self.assertEqual(dm.CLASS_NAMES, {code: name for name, code in prep.LABEL.items()})
        frame = dm.encode_answers(EXAMPLE_B)
        self.assertEqual(frame.iloc[0].tolist(), [45, 1, 1, 1] + [0] * 12)
        self.assertTrue((frame.dtypes == "int64").all())

    def test_age_range_is_the_range_in_the_dataset(self):
        age = read_json("01_data_validation.json")["age_summary"]
        self.assertEqual((dm.AGE_MIN, dm.AGE_MAX), (age["min"], age["max"]))

    def test_the_two_synthetic_examples(self):
        self.assertEqual(dm.predict_class(self.pipeline, dm.encode_answers(EXAMPLE_A)), "Negative")
        self.assertEqual(dm.predict_class(self.pipeline, dm.encode_answers(EXAMPLE_B)), "Positive")

    def test_prediction_returns_only_a_class_and_uses_only_predict(self):
        class OnlyPredict:
            def predict(self, frame):
                return [1]

            def __getattr__(self, name):  # decision_function, predict_proba, ... must never be touched
                raise AssertionError(f"the app must not use {name}")

        self.assertEqual(dm.predict_class(OnlyPredict(), dm.encode_answers(EXAMPLE_A)), "Positive")
        for bad in ([2], [0, 1], []):
            probe = type("Probe", (), {"predict": lambda self, frame, bad=bad: bad})()
            with self.assertRaises(RuntimeError):
                dm.predict_class(probe, dm.encode_answers(EXAMPLE_A))
        with self.assertRaises(ValueError):  # wrong column order is refused before the model is called
            dm.predict_class(self.pipeline, dm.encode_answers(EXAMPLE_A)[dm.FEATURES[::-1]])


class InputValidationTests(unittest.TestCase):
    def test_valid_answers_are_accepted(self):
        for age in (16, 45, 90, 45.0, "45"):
            with self.subTest(age=age):
                self.assertEqual(dm.validate_answers({**EXAMPLE_A, "age": age}), [])
        for gender in dm.GENDER_CODES:
            self.assertEqual(dm.validate_answers({**EXAMPLE_A, "gender_male": gender}), [])

    def test_invalid_answers_are_rejected_and_never_encoded(self):
        without_obesity = {k: v for k, v in EXAMPLE_A.items() if k != "obesity"}
        cases = {
            "age missing": {**EXAMPLE_A, "age": None}, "age empty text": {**EXAMPLE_A, "age": " "},
            "age below range": {**EXAMPLE_A, "age": 15}, "age above range": {**EXAMPLE_A, "age": 91},
            "age fractional": {**EXAMPLE_A, "age": 45.5}, "age is text": {**EXAMPLE_A, "age": "forty"},
            "age is boolean": {**EXAMPLE_A, "age": True}, "age is nan": {**EXAMPLE_A, "age": float("nan")},
            "age is infinite": {**EXAMPLE_A, "age": float("inf")}, "age negative": {**EXAMPLE_A, "age": -45},
            "gender missing": {**EXAMPLE_A, "gender_male": None}, "gender unknown": {**EXAMPLE_A, "gender_male": "Other"},
            "gender as a number": {**EXAMPLE_A, "gender_male": 1},
            "symptom missing": {**EXAMPLE_A, "polyuria": None}, "symptom unknown": {**EXAMPLE_A, "polyuria": "Maybe"},
            "symptom lower case": {**EXAMPLE_A, "polyuria": "yes"}, "symptom as a number": {**EXAMPLE_A, "polyuria": 1},
            "symptom key absent": without_obesity, "extra input": {**EXAMPLE_A, "bmi": 30}, "nothing at all": {},
        }
        for name, answers in cases.items():
            with self.subTest(name):
                self.assertTrue(dm.validate_answers(answers))
                with self.assertRaises(ValueError):
                    dm.encode_answers(answers)

    def test_messages_use_display_names_when_given(self):
        problems = dm.validate_answers({**EXAMPLE_A, "sudden_weight_loss": None}, labels={"sudden_weight_loss": "Sudden weight loss"})
        self.assertEqual(problems, ["Answer Yes or No for: Sudden weight loss."])


class DemoArtifactTests(unittest.TestCase):
    """The published model file holds fitted numbers only: no training record, support vector or row-level value.

    The claim rests on looking inside the actual file, not on the model's type alone: src/s12_demo_model.py loads the
    bytes, records every class the file refers to, and walks every array and attribute of the loaded object.
    """

    @classmethod
    def setUpClass(cls):
        cls.audit = s12.audit_artifact(dm.MODEL_PATH)

    def test_the_inspection_finds_nothing_that_could_be_a_record(self):
        self.assertEqual(s12.rejection_reasons(self.audit), [])
        self.assertEqual(sorted(set(self.audit["attributes"]) & s12.FORBIDDEN_ATTRIBUTES), [])
        self.assertTrue(all(name.startswith(s12.ALLOWED_PREFIXES) for name in self.audit["classes"]), self.audit["classes"])

    def test_the_file_is_small(self):
        # The 416 training rows alone would take about 53 kB as 64-bit numbers (416 x 16 x 8 bytes).
        self.assertEqual(self.audit["bytes"], dm.MODEL_PATH.stat().st_size)
        self.assertLess(self.audit["bytes"], 10_000)
        self.assertLess(self.audit["values_in_arrays"], 100)
        self.assertEqual(self.audit["largest_array"], 16, "the largest array is the 16 coefficients")

    def test_the_only_numbers_stored_are_coefficients_the_age_scaling_and_bookkeeping(self):
        numeric = {where.rsplit(".", 1)[1]: (shape, size) for where, shape, dtype, size in self.audit["arrays"] if dtype != "object"}
        self.assertEqual(numeric, {"coef_": ((1, 16), 16), "intercept_": ((1,), 1), "mean_": ((1,), 1), "var_": ((1,), 1),
                                   "scale_": ((1,), 1), "classes_": ((2,), 2), "n_iter_": ((1,), 1)})

    def test_the_only_text_stored_in_arrays_is_input_column_names(self):
        text_arrays = [array for array in self.audit["array_objects"] if array.dtype == object]
        self.assertEqual(len(text_arrays), 3)
        for array in text_arrays:
            self.assertLessEqual(set(array.tolist()), set(dm.FEATURES))

    def test_no_training_row_appears_in_any_stored_array(self):
        if not (ROOT / "data" / "processed" / "train_rows.csv").is_file():
            self.skipTest("the training rows are kept on the owner's computer, not in the published repository")
        X = mlcommon.view_frame(mlcommon.load_training(), "primary")[dm.FEATURES]
        self.assertEqual(len(X), read_json("model_test_results.json")["views"]["primary"]["n_train_rows"])
        scaled = self.audit["model"].named_steps["prep"].transform(X)
        self.assertEqual(s12.rows_found_in_arrays(self.audit["array_objects"], X.to_numpy(dtype=float)), 0)
        self.assertEqual(s12.rows_found_in_arrays(self.audit["array_objects"], scaled), 0)

    def test_the_search_would_notice_stored_records(self):
        # A control for the checks above: the same tools do flag an SVM, which stores its support vectors.
        import joblib
        from sklearn.svm import SVC
        rows = np.arange(32, dtype=float).reshape(2, 16)
        self.assertEqual(s12.rows_found_in_arrays([rows.copy()], rows), 2)
        self.assertEqual(s12.rows_found_in_arrays([rows.ravel()], rows), 2)
        self.assertEqual(s12.rows_found_in_arrays([rows + 0.5], rows), 0)
        frame, labels = invented_training_data()
        buffer = io.BytesIO()
        joblib.dump(SVC(kernel="rbf").fit(frame, labels), buffer)
        svm = s12.audit_bytes(buffer.getvalue())
        self.assertIn("support_vectors_", " ".join(s12.rejection_reasons(svm)))
        self.assertGreater(s12.rows_found_in_arrays(svm["array_objects"], frame.to_numpy(dtype=float)), 0)


class ModelFileProblemTests(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.tmp, self.model, self.note = tmp, tmp / "copy.joblib", tmp / "copy.txt"
        shutil.copy(dm.MODEL_PATH, self.model)
        shutil.copy(dm.MODEL_NOTE, self.note)

    def assertRefused(self, pattern: str):
        with self.assertRaisesRegex(dm.ModelUnavailableError, pattern) as caught:
            dm.load_pipeline(self.model, self.note)
        self.assertNotIn(str(self.tmp), str(caught.exception), "messages must not contain absolute paths")

    def test_an_unchanged_copy_loads(self):
        self.assertEqual(list(dm.load_pipeline(self.model, self.note).feature_names_in_), dm.FEATURES)

    def test_missing_model_file(self):
        self.model.unlink()
        self.assertRefused("was not found")

    def test_changed_model_file_is_not_loaded(self):
        data = bytearray(self.model.read_bytes())
        data[len(data) // 2] ^= 1
        self.model.write_bytes(bytes(data))
        self.assertRefused("SHA-256")

    def test_missing_or_hashless_description_file(self):
        self.note.write_text("no hash recorded here")
        self.assertRefused("does not record a SHA-256")
        self.note.unlink()
        self.assertRefused("was not found")

    def save_with_matching_hash(self, estimator):
        """Replace the copy with this estimator and record its real hash, so that only what it contains can be refused."""
        import joblib
        joblib.dump(estimator, self.model)
        self.note.write_text(f"joblib file SHA-256: {dm.sha256_file(self.model)}\n")

    def test_an_rbf_svm_is_refused_even_when_its_hash_matches(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        frame, labels = invented_training_data()
        for estimator in (SVC(kernel="rbf"), make_pipeline(StandardScaler(), SVC(kernel="rbf"))):
            with self.subTest(type(estimator).__name__):
                self.save_with_matching_hash(estimator.fit(frame, labels))
                self.assertRefused("not the demo-only Logistic Regression")

    def test_a_logistic_regression_that_carries_records_is_refused(self):
        from sklearn.linear_model import LogisticRegression
        frame, labels = invented_training_data()
        model = LogisticRegression(max_iter=5000).fit(frame, labels)
        model.support_vectors_ = frame.to_numpy()  # what an SVM would add
        self.save_with_matching_hash(model)
        self.assertRefused("not the demo-only Logistic Regression")

    def test_a_model_for_other_input_columns_is_refused(self):
        from sklearn.linear_model import LogisticRegression
        frame, labels = invented_training_data()
        self.save_with_matching_hash(LogisticRegression(max_iter=5000).fit(frame[dm.FEATURES[::-1]], labels))
        self.assertRefused("different input columns")


# --- the pages ---------------------------------------------------------------------------------------------
class PageTests(unittest.TestCase):
    def test_disclaimer_in_the_app_is_the_approved_wording(self):
        self.assertEqual(app_common.DISCLAIMER, APPROVED_DISCLAIMER)

    def test_every_page_renders_cleanly_with_the_disclaimer(self):
        for page in PAGES:
            with self.subTest(page):
                at = open_page(page)
                self.assertEqual(len(at.exception), 0)
                warnings = [w.value for w in at.warning]
                self.assertIn(APPROVED_DISCLAIMER, warnings)
                self.assertFalse([w for w in warnings if "unavailable" in w], "a saved file could not be read")
                text = "\n".join(all_text(at))
                self.assertNotIn("Traceback", text)
                self.assertNotIn("/" + "Users" + "/", text)

    def test_model_comparison_tables_match_the_saved_results(self):
        at = open_page("model_comparison")
        for label, view in ANALYSES.items():
            at.radio(key="comparison_analysis").set_value(label).run()
            tables = report_tables(at)
            cv = next(t for t in tables if "Selected configuration" in t.columns)
            test = next(t for t in tables if "ROC-AUC [95% interval]" in t.columns)
            with self.subTest(view=view):
                self.assertEqual(cv["Model"].tolist()[:5], MODELS)
                self.assertEqual(cv.iloc[:, 2:].values.tolist(), expected_cv(view))
                self.assertEqual(test["Model"].tolist()[:5], MODELS)
                self.assertEqual(test.iloc[:, 1:].values.tolist(), expected_test(view))

    def test_confusion_counts_match_the_saved_results(self):
        at = open_page("evaluation")
        for label, view in ANALYSES.items():
            at.radio(key="evaluation_analysis").set_value(label).run()
            for model, key in [*zip(MODELS, MODELS), ("FNN (secondary experiment)", None)]:
                at.selectbox(key="evaluation_model").set_value(model).run()
                counts = _results_block(view, key if key else None)["confusion"]
                shown = {m.label: m.value for m in at.metric}
                with self.subTest(view=view, model=model):
                    self.assertEqual(shown, {"True negatives": str(counts["tn"]), "False positives": str(counts["fp"]),
                                             "False negatives": str(counts["fn"]), "True positives": str(counts["tp"])})

    def test_primary_versus_sensitivity_table_matches_the_saved_results(self):
        shown = next(t for t in report_tables(open_page("evaluation")) if "CV diff (S - P)" in t.columns)
        self.assertEqual(shown["Model"].tolist(), [*MODELS, "FNN (secondary experiment)"])
        for row, model in zip(shown.itertuples(index=False), [*MODELS, None]):
            cv_p, cv_s = (expected_cv(v)[([*MODELS, None].index(model))][5].split(" ")[0] for v in ("primary", "sensitivity"))
            test_p, test_s = (expected_test(v)[([*MODELS, None].index(model))][5].split(" ")[0] for v in ("primary", "sensitivity"))
            with self.subTest(model=model):
                self.assertEqual((row[1], row[2], row[4], row[5]), (cv_p, cv_s, test_p, test_s))

    def test_cv_scores_are_above_test_scores_as_the_pages_say(self):
        for view in ANALYSES.values():
            for model, cv, test in zip(MODELS[1:], expected_cv(view)[1:], expected_test(view)[1:]):
                with self.subTest(view=view, model=model):
                    self.assertGreater(float(cv[5].split(" ")[0]), float(test[5].split(" ")[0]))

    def test_the_figures_shown_are_the_saved_ones(self):
        shown = set(re.findall(r"fig\d\d_\w+\.png", "".join(p.read_text() for p in (APP / "views").glob("*.py"))))
        self.assertEqual(shown, {p.name for p in (REPORTS / "figures").glob("*.png")}, "every saved figure is used")
        manifest = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (REPORTS / "sha256_manifest.txt").read_text().splitlines()}
        embedded = (REPORTS / "09_model_results.md").read_text() + (REPORTS / "05_eda_training.md").read_text()
        for name in shown:
            with self.subTest(figure=name):
                self.assertEqual(dm.sha256_file(REPORTS / "figures" / name), manifest[f"reports/figures/{name}"])
                self.assertIn(name, embedded, "the report embeds the same file")


class PredictionPageTests(unittest.TestCase):
    def test_blank_form_gives_a_list_of_problems_and_no_class(self):
        at = submit(open_page("prediction"))
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(len(at.error), 1)
        for expected in ("Age is missing", "Gender is missing", "Polyuria", "Obesity"):
            self.assertIn(expected, at.error[0].value)

    def test_the_two_examples_show_only_the_class(self):
        for example, expected in (("A", "Negative"), ("B", "Positive")):
            with self.subTest(example=example):
                at = open_page("prediction")
                next(b for b in at.button if b.label == f"Fill example {example}").click()
                at.run()
                before = all_text(at)
                at = submit(at)
                self.assertEqual(len(at.error), 0)
                self.assertEqual([(m.label, m.value) for m in at.metric], [("Predicted dataset class", expected)])
                new = [t for t in all_text(at) if t not in before and t != APPROVED_DISCLAIMER]
                joined = " ".join(new).lower()
                for banned in ("%", "probab", "confiden", "score", "risk", "likelihood", "chance", "diagnos", "advice", "decision"):
                    self.assertNotIn(banned, joined, f"'{banned}' must not appear next to the result")
                shown = [d.value for d in at.dataframe]
                self.assertEqual((len(shown), len(at.table)), (1, 0), "the only table is the encoded model input")
                self.assertEqual(shown[0].columns.tolist(), dm.FEATURES)
                self.assertEqual(shown[0].shape, (1, 16))
                self.assertIn(APPROVED_DISCLAIMER, [c.value for c in at.caption], "disclaimer repeated beside the result")

    # Values typed into a form are only sent when the form is submitted, so these tests fill the widgets and
    # submit in one step (a rerun in between would drop the staged values).
    def test_a_complete_manual_entry_is_predicted_and_matches_the_direct_call(self):
        for answers in (EXAMPLE_A, EXAMPLE_B, {**EXAMPLE_B, "age": 16, "gender_male": "Female"}):
            with self.subTest(age=answers["age"], polyuria=answers["polyuria"]):
                at = open_page("prediction")
                fill(at, answers)
                at = submit(at)
                self.assertEqual(len(at.error), 0)
                expected = dm.predict_class(dm.load_pipeline(), dm.encode_answers(answers))
                self.assertEqual([m.value for m in at.metric], [expected])

    def test_out_of_range_age_is_reported_by_name_and_not_predicted(self):
        for age in (15, 91, 200):
            with self.subTest(age=age):
                at = open_page("prediction")
                fill(at, EXAMPLE_A)
                at.number_input(key="answer_0_age").set_value(age)
                at = submit(at)
                self.assertEqual(len(at.metric), 0)
                self.assertEqual(len(at.error), 1)
                self.assertIn(f"Age {age} is outside the range of ages in the dataset (16 to 90)", at.error[0].value)

    def test_age_input_has_no_bounds_so_every_typed_value_reaches_the_validation(self):
        # With min/max set, a real browser keeps the previous valid number when an out-of-range one is typed, and the
        # form is then checked with an age the visitor no longer sees. Found by trying 200 in a browser.
        age = open_page("prediction").number_input(key="answer_0_age")
        self.assertIsNone(age.min)
        self.assertIsNone(age.max)

    def test_one_unanswered_question_blocks_the_prediction(self):
        at = open_page("prediction")
        fill(at, EXAMPLE_A)
        at.radio(key="answer_0_alopecia").set_value(None)
        at = submit(at)
        self.assertEqual(len(at.metric), 0)
        message = at.error[0].value
        self.assertIn("Alopecia", message)
        for answered in ("Age", "Gender", "Polyuria", "Obesity"):
            self.assertNotIn(answered, message, "only the unanswered question is reported")

    def test_clear_button_rebuilds_every_answer_blank_and_removes_the_result(self):
        at = open_page("prediction")
        next(b for b in at.button if b.label == "Fill example B").click()
        at.run()
        self.assertEqual(at.radio(key="answer_0_polyuria").value, "Yes")
        at = submit(at)
        self.assertEqual(len(at.metric), 1)
        next(b for b in at.button if b.label == "Clear all answers").click()
        at.run()
        self.assertEqual(len(at.metric), 0, "the result does not outlive the answers")
        # New widget keys (generation 1): the old ones, with the old answers, are gone.
        self.assertIsNone(at.number_input(key="answer_1_age").value)
        self.assertTrue(all(at.radio(key=f"answer_1_{f}").value is None for f in dm.FEATURES[1:]))
        self.assertFalse([f for f in dm.FEATURES if f"answer_0_{f}" in at.session_state],
                         "no answer from before the clear is kept")
        # The blank form is validated like a new one, and examples can be filled again.
        at = submit(at)
        self.assertEqual(len(at.metric), 0)
        self.assertIn("Age is missing", at.error[0].value)
        next(b for b in at.button if b.label == "Fill example A").click()
        at.run()
        at = submit(at)
        self.assertEqual([m.value for m in at.metric], ["Negative"])

    def test_the_page_says_this_is_a_demo_only_logistic_regression_and_not_the_selected_svm(self):
        text = "\n".join(all_text(open_page("prediction")))
        for phrase in ("demo-only Logistic Regression", "It is **not** the model the project selected", "RBF SVM",
                       "stores some of its training records", "holds only coefficients", "Not the selected model"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_overview_and_model_comparison_say_the_prediction_page_uses_a_different_model(self):
        for page in ("overview", "model_comparison"):
            with self.subTest(page):
                text = "\n".join(all_text(open_page(page)))
                self.assertIn("demo-only Logistic Regression", text)
                if page == "model_comparison":  # the saved selection rule shown here calls the chosen SVM "the demo model"
                    self.assertIn("means the research-selected model", text)


class MissingFileTests(unittest.TestCase):
    """Run the app from a copy with some saved files removed or changed; it must say so instead of crashing."""

    SCRIPT = r"""
import json, sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
copy = Path(sys.argv[1])
sys.path.insert(0, str(copy / "app"))
at = AppTest.from_file(str(copy / "app" / "streamlit_app.py"), default_timeout=90).run()
out = {}
for page in ("overview", "exploration", "model_comparison", "evaluation", "prediction"):
    at.switch_page(f"views/{page}.py").run()
    out[page] = {"exceptions": len(at.exception), "errors": [e.value for e in at.error],
                 "warnings": [w.value for w in at.warning][1:], "metrics": len(at.metric)}
print(json.dumps(out))
"""

    def run_copy(self, change) -> dict:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for name in ("app", "reports", "models"):
            shutil.copytree(ROOT / name, tmp / name, ignore=shutil.ignore_patterns("__pycache__"))
        change(tmp)
        result = subprocess.run([sys.executable, "-B", "-c", self.SCRIPT, str(tmp)], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_missing_model_file_shows_a_message(self):
        out = self.run_copy(lambda tmp: (tmp / "models" / "demo_logistic_regression.joblib").unlink())
        self.assertTrue(all(page["exceptions"] == 0 for page in out.values()))
        self.assertIn("was not found", out["prediction"]["errors"][0])
        self.assertEqual(out["prediction"]["metrics"], 0)
        self.assertNotIn("Users", json.dumps(out))

    def test_changed_model_file_is_refused(self):
        def tamper(tmp):
            path = tmp / "models" / "demo_logistic_regression.joblib"
            data = bytearray(path.read_bytes())
            data[100] ^= 1
            path.write_bytes(bytes(data))
        out = self.run_copy(tamper)
        self.assertIn("SHA-256", out["prediction"]["errors"][0])
        self.assertEqual(out["prediction"]["metrics"], 0)

    def test_missing_reports_and_figures_show_warnings_on_every_page(self):
        def remove(tmp):
            shutil.rmtree(tmp / "reports")
            (tmp / "reports").mkdir()
        out = self.run_copy(remove)
        for page, state in out.items():
            with self.subTest(page):
                self.assertEqual(state["exceptions"], 0)
                self.assertTrue(any("unavailable" in w for w in state["warnings"]), state)


# --- what the app is allowed to do, and what Git would publish --------------------------------------------
class PrivacyAndHygieneTests(unittest.TestCase):
    def test_app_code_cannot_write_files_or_reach_the_network(self):
        banned = [r"\brequests\b", r"\burllib", r"http\.client", r"\bsocket\b", r"\bsmtplib\b", r"\bsubprocess\b",
                  r"os\.system", r"\.write_text\(", r"\.write_bytes\(", r"\.to_csv\(", r"\.to_json\(", r"\.to_pickle\(",
                  r"joblib\.dump", r"pickle\.dump", r"\bsqlite3\b", r"open\([^)]*['\"][wax]\+?b?['\"]", r"file_uploader",
                  r"\bst\.secrets\b", r"os\.environ", r"^\s*(?:import|from)\s+(?:torch|matplotlib|seaborn)\b"]
        for path in sorted(APP.rglob("*.py")):
            source = path.read_text()
            for pattern in banned:
                with self.subTest(file=path.name, pattern=pattern):
                    self.assertIsNone(re.search(pattern, source, re.M), f"{path.name} matches {pattern}")

    def test_the_app_never_asks_the_model_for_probabilities_or_scores(self):
        # A Logistic Regression can produce probabilities (the research SVM's saved pipeline could not), so the rule that
        # only the predicted class is ever shown is enforced here, in the code, and not by the choice of model.
        for path in sorted(APP.rglob("*.py")):
            with self.subTest(file=path.name):
                self.assertIsNone(re.search(r"predict_proba|predict_log_proba|decision_function", path.read_text()))

    def test_the_only_web_links_in_the_app_are_citations(self):
        hosts = set()
        for path in APP.rglob("*.py"):
            hosts |= set(re.findall(r"https?://([^/\s)\"']+)", path.read_text()))
        self.assertEqual(hosts, {"doi.org"})

    def test_streamlit_settings_are_local_and_private(self):
        config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
        self.assertIs(config["browser"]["gatherUsageStats"], False)
        self.assertIs(config["server"]["showEmailPrompt"], False)
        self.assertEqual(config["server"]["address"], "localhost", "by default Streamlit listens on every network interface")
        self.assertEqual(config["client"]["showErrorDetails"], "none")

    def test_streamlit_agent_skill_links_cannot_be_published_by_accident(self):
        # Streamlit's "Install" pop-up creates symlinks into the local Python environment, inside the project folder.
        for path in (".agents/skills/developing-with-streamlit", ".claude/skills/developing-with-streamlit"):
            with self.subTest(path=path):
                try:
                    code = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode
                except OSError:
                    self.skipTest("git is not available")
                if code == 128:
                    self.skipTest("not a git repository")
                self.assertEqual(code, 0, f"{path} would be picked up by `git add .`")

    def test_requirements_are_pinned_and_split(self):
        def packages(name):
            lines = [l.split("#")[0].strip() for l in (ROOT / name).read_text().splitlines()]
            return [l for l in lines if l and not l.startswith("-r")], [l for l in lines if l.startswith("-r")]
        core, core_includes = packages("requirements.txt")
        analysis, analysis_includes = packages("requirements-analysis.txt")
        fnn, fnn_includes = packages("requirements-fnn.txt")
        self.assertEqual(core_includes, [])
        self.assertEqual((analysis_includes, fnn_includes), (["-r requirements.txt"], ["-r requirements-analysis.txt"]))
        for line in core + analysis + fnn:
            self.assertRegex(line, r"^[A-Za-z0-9_.-]+==\d+(\.\d+)*$", f"not pinned: {line}")
        names = {line.split("==")[0] for line in core}
        self.assertEqual(names, {"numpy", "pandas", "scipy", "scikit-learn", "joblib", "streamlit"})
        self.assertEqual(fnn, ["torch==2.15.0"], "PyTorch is optional and not part of the app")
        recorded = read_json("model_test_results.json")["software"]
        pinned = dict(line.split("==") for line in core)
        for name in ("numpy", "pandas", "scikit-learn", "joblib"):
            self.assertEqual(pinned[name], recorded[name], f"{name} must match the version the model was fitted with")

    def test_app_imports_are_all_listed_in_requirements(self):
        listed = {line.split("==")[0] for line in (ROOT / "requirements.txt").read_text().splitlines()
                  if line.strip() and not line.startswith(("#", "-r"))}
        package_of = {"sklearn": "scikit-learn"}  # import name -> package name
        local = {path.stem for path in APP.rglob("*.py")}
        imported = set()
        for path in APP.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imported |= {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
        needed = {package_of.get(name, name) for name in imported if name not in sys.stdlib_module_names and name not in local}
        self.assertLessEqual(needed, listed, f"imported by the app but missing from requirements.txt: {sorted(needed - listed)}")
        self.assertTrue({"streamlit", "pandas", "scikit-learn", "joblib"} <= needed, "the app's real dependencies were not detected")

    def test_mit_licence_covers_the_code_and_not_the_dataset(self):
        licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(licence.startswith("MIT License\n"))
        self.assertRegex(licence, r"(?m)^Copyright \(c\) 2026 SHISHIR MD SHIFAT HOSSAIN$")
        self.assertNotRegex(licence, r"[\[\]<>]", "no placeholder may remain in the licence")
        for phrase in ("Permission is hereby granted, free of charge", 'THE SOFTWARE IS PROVIDED "AS IS"'):
            self.assertIn(phrase, licence)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("## Licence", 1)[1].split("\n## ", 1)[0]
        for phrase in ("MIT licence", "does not cover the dataset", "CC BY 4.0", "makes no claim to it", "attribution"):
            self.assertIn(phrase, section)
        source = (ROOT / "docs" / "DATA_SOURCE.md").read_text(encoding="utf-8")
        self.assertIn("CC BY 4.0", source)
        self.assertIn("not the data", source)

    def test_running_the_pages_writes_nothing_to_the_project_folder(self):
        before = snapshot(ROOT)
        for page in PAGES:
            open_page(page)
        at = open_page("prediction")
        next(b for b in at.button if b.label == "Fill example B").click()
        at.run()
        submit(at)
        self.assertEqual(snapshot(ROOT), before)

    def test_files_git_would_publish_contain_no_local_paths_or_secrets(self):
        files = git_publishable_files()
        if files is None:
            self.skipTest("git is not available")
        private_prefixes = ("data/", "archive/")
        private_files = {"reports/00_environment.md", "reports/00_environment_fingerprints.txt", "reports/00_smoke_test.json",
                         "reports/07_correction_verification.md", "reports/07_day1_correction_report.md",
                         "reports/08_day1_final_review.md", "reports/DAY1_SUMMARY.md",
                         "models/primary_demo_pipeline.joblib", "models/primary_demo_pipeline.txt"}  # the SVM stores records
        needles = ["/" + "Users" + "/", "/private/", "/opt/" + "homebrew", "miniconda", "Desktop", "iCloud", ".conda/",
                   "/home/", "Major Project"]
        secret = re.compile(r"ghp_[A-Za-z0-9]{20}|AKIA[0-9A-Z]{12}|sk-[A-Za-z0-9]{20}|xox[bap]-|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY")
        self.assertTrue(files, "no files found")
        for name in files:
            if name == "tests/test_app.py":
                continue  # this file lists the forbidden strings, so it would always match itself
            with self.subTest(file=name):
                self.assertFalse(name.startswith(private_prefixes), "row-level data or archive would be published")
                self.assertNotIn(name, private_files)
                self.assertFalse(name.endswith((".sqlite", ".sqlite3", ".env", ".DS_Store")))
                if name.endswith((".png", ".joblib")):
                    raw = (ROOT / name).read_bytes()
                    self.assertNotIn(("/" + "Users" + "/").encode(), raw)
                    continue
                text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
                for needle in needles:
                    self.assertNotIn(needle, text, f"{name} contains {needle!r}")
                self.assertIsNone(secret.search(text), f"{name} looks like it contains a secret")

    def test_no_email_address_is_published_except_the_github_noreply_one(self):
        files = git_publishable_files()
        if files is None:
            self.skipTest("git is not available")
        email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
        checked = 0
        for name in files:
            if name.endswith((".png", ".joblib")):
                continue
            checked += 1
            found = [m for m in email.findall((ROOT / name).read_text(encoding="utf-8", errors="replace"))
                     if not m.endswith("@users.noreply.github.com")]
            with self.subTest(file=name):
                self.assertEqual(found, [], f"{name} contains an e-mail address")
        self.assertTrue(checked, "no text files were scanned")

    def test_only_the_row_free_demo_model_is_published(self):
        files = git_publishable_files()
        if files is None:
            self.skipTest("git is not available")
        self.assertEqual([f for f in files if f.startswith("models/")],
                         ["models/demo_logistic_regression.joblib", "models/demo_logistic_regression.txt"])
        self.assertEqual([f for f in files if f.endswith((".joblib", ".pkl", ".pickle", ".sav", ".pt", ".pth", ".onnx"))],
                         ["models/demo_logistic_regression.joblib"])
        for path in ("models/primary_demo_pipeline.joblib", "models/primary_demo_pipeline.txt"):
            with self.subTest(path=path):  # an RBF SVM stores training records, so Git must ignore it
                code = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode
                self.assertEqual(code, 0, f"{path} would be picked up by `git add .`")


class DocumentationTests(unittest.TestCase):
    """The README, the notebook summary, the report and the saved results must tell the same story."""

    VIEWS = {"primary": "Primary analysis (all 520", "sensitivity": "Sensitivity analysis (251"}
    MODEL = {"Majority-class baseline": "Majority-class baseline", "Logistic Regression": "Logistic Regression",
             "Categorical Naive Bayes": "Categorical NB", "RBF SVM (chosen)": "RBF SVM", "Random Forest": "Random Forest",
             "Small neural network (secondary)": "FNN (secondary experiment)"}

    @staticmethod
    def readme() -> str:
        return (ROOT / "README.md").read_text(encoding="utf-8")

    def readme_tables(self) -> dict[str, list[list[str]]]:
        tables, block = {}, []
        for line in self.readme().splitlines() + [""]:
            if line.startswith("|"):
                block.append([cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")])
            elif block:
                tables[block[0][0]] = block[2:]  # the first header cell names the table; drop the header and the rule line
                block = []
        return tables

    @staticmethod
    def report(*headings: str, column: str) -> pd.DataFrame:
        return app_common.report_table(app_common.RESULTS_REPORT, *headings, column=column).set_index("Model")

    def test_readme_results_table_matches_the_report(self):
        rows = self.readme_tables()["Model"]
        self.assertEqual([row[0] for row in rows], list(self.MODEL))
        for row in rows:
            model = self.MODEL[row[0]]
            for column, (view, kind) in enumerate([("primary", "cv"), ("primary", "test"), ("sensitivity", "cv"), ("sensitivity", "test")], start=1):
                if kind == "cv":
                    expected = self.report("Cross-validation results", self.VIEWS[view], column="Selected configuration").loc[model, "ROC-AUC"]
                else:
                    expected = self.report("Test results", self.VIEWS[view], column="ROC-AUC [95% interval]").loc[model, "ROC-AUC [95% interval]"]
                    if model == "Majority-class baseline":
                        expected = expected.split(" ")[0]  # the README shows the baseline without its zero-width interval
                with self.subTest(model=model, view=view, kind=kind):
                    self.assertEqual(row[column], expected)

    def test_readme_chosen_model_table_matches_the_report(self):
        rows = self.readme_tables()["Metric"]
        self.assertEqual([row[0] for row in rows],
                         ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "Confusion matrix (TN / FP / FN / TP)"])
        for row in rows:
            for column, view in ((1, "primary"), (2, "sensitivity")):
                if row[0].startswith("Confusion"):
                    counts = self.report("Test results", self.VIEWS[view], column="True Negative").loc["RBF SVM"]
                    expected = " / ".join(counts[name] for name in ("True Negative", "False Positive", "False Negative", "True Positive"))
                else:
                    expected = self.report("Test results", self.VIEWS[view], column="ROC-AUC [95% interval]").loc["RBF SVM", f"{row[0]} [95% interval]"]
                with self.subTest(metric=row[0], view=view):
                    self.assertEqual(row[column], expected)

    def test_readme_counts_and_claims_match_the_saved_results(self):
        readme = self.readme()
        p, s = (read_json("model_test_results.json")["views"][v] for v in ("primary", "sensitivity"))
        self.assertIn(f"{p['n_train_rows']} training rows; {p['n_test_rows']} test rows from {p['n_test_groups']} profile groups, "
                      f"{p['test_positive']} Positive and {p['test_negative']} Negative", readme)
        self.assertIn(f"{s['n_train_rows']} training rows; {s['n_test_rows']} test rows, {s['test_positive']} Positive and "
                      f"{s['test_negative']} Negative", readme)
        audit = read_json("02_duplicate_audit.json")["exact_duplicates"]
        self.assertIn(f"There are {audit['total_rows']} rows but only {audit['unique_records']} distinct", readme)
        age = read_json("01_data_validation.json")["age_summary"]
        self.assertIn(f"{age['rows_outside_documented_20_65']} rows fall outside it (ages {age['min']} to {age['max']})", readme)
        accuracy = {v: self.report("Test results", self.VIEWS[v], column="ROC-AUC [95% interval]").loc["Majority-class baseline", "Accuracy [95% interval]"].split(" ")[0]
                    for v in self.VIEWS}
        self.assertIn(f"{accuracy['primary']} accuracy on the primary test set ({accuracy['sensitivity']} on the sensitivity test set)", readme)
        stored = re.search(r"stores (\d+) of the (\d+) training records", dm.MODEL_NOTE.read_text())
        self.assertEqual(int(stored.group(2)), p["n_train_rows"])
        self.assertIn(f"{stored.group(1)} of the {p['n_train_rows']} training rows", readme)
        svm_file = ROOT / "models" / "primary_demo_pipeline.joblib"
        if svm_file.is_file():  # the research SVM stays on the owner's computer; it is not in the published repository
            import joblib
            self.assertIn(f"{dm.sha256_file(svm_file)}  models/primary_demo_pipeline.joblib", (REPORTS / "sha256_manifest.txt").read_text())
            self.assertEqual(int(joblib.load(svm_file)[-1].n_support_.sum()), int(stored.group(1)))
        chosen = {v: read_json("model_selection.json")["views"][v]["chosen_model"] for v in self.VIEWS}
        self.assertEqual(chosen, {"primary": "RBF SVM", "sensitivity": "RBF SVM"})
        self.assertEqual(read_json("model_test_results.json")["demo_pipeline"]["model"], "RBF SVM")
        self.assertIn("primary = **RBF SVM**; sensitivity = **RBF SVM**", (REPORTS / "09_model_results.md").read_text(encoding="utf-8"))

    def test_readme_explains_the_two_models_and_what_is_published(self):
        section = self.readme().split("## Models in this project", 1)[1].split("\n## ", 1)[0]
        for phrase in ("RBF SVM", "pre-declared", "demo-only Logistic Regression", "support vectors",
                       "models/demo_logistic_regression.joblib", "models/primary_demo_pipeline.joblib", "not published",
                       "reports/09_model_results.md", "reports/model_selection.json",
                       "because it contains training records"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, section)

    def test_readme_links_to_the_authors_github_profile(self):
        readme = self.readme()
        self.assertIn("[SHISHIR MD SHIFAT HOSSAIN](https://github.com/shishir9005)", readme)
        for url in re.findall(r"https://github\.com/[^\s)>\]\"']+", readme):
            self.assertTrue(url.startswith("https://github.com/shishir9005"), url)

    def test_notebook_summary_uses_numbers_from_the_report_and_names_the_right_requirements(self):
        notebook = json.loads((ROOT / "notebooks" / "clinical_diabetes_analysis.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(cell["source"]) for cell in notebook["cells"]]
        summary = next(text for text in cells if text.startswith("## Key results"))
        report = (REPORTS / "09_model_results.md").read_text(encoding="utf-8")
        numbers = set(re.findall(r"\b\d\.\d{3}\b", summary))
        self.assertGreaterEqual(len(numbers), 8)
        self.assertEqual({n for n in numbers if n not in report}, set(), "numbers in the notebook summary that the report lacks")
        self.assertIn("requirements-analysis.txt", next(text for text in cells if text.startswith("## 12. How to reproduce")))
        # The cell that loads the research SVM must say that its file is local-only (an SVM stores training records).
        loader = next(i for i, cell in enumerate(notebook["cells"])
                      if cell["cell_type"] == "code" and "primary_demo_pipeline.joblib" in cells[i])
        self.assertIn("LOCAL-ONLY FILE", cells[loader])
        self.assertIn("not in the public repository", cells[loader])
        self.assertIn("cannot be fully reproduced from the public repository", cells[loader - 1])
        self.assertFalse([o for cell in notebook["cells"] for o in cell.get("outputs", []) if o.get("output_type") == "error"])


if __name__ == "__main__":
    unittest.main()
