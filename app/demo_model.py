"""The demo-only Logistic Regression: loading, input validation, encoding and prediction.

There is no Streamlit code in this file, so all of it can be tested without a browser
(tests/test_app.py). The page that uses it is app/views/prediction.py.

Only the predicted class is ever produced. Nothing here computes probabilities, scores or risk,
and nothing is written to disk.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import warnings
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "demo_logistic_regression.joblib"
MODEL_NOTE = ROOT / "models" / "demo_logistic_regression.txt"

# The app only ever loads the demo-only Logistic Regression, never a model that stores training records. An RBF SVM
# keeps its support vectors, with their classes, inside the saved file. That is why the research-selected SVM is not
# used here and is not published (see the README).
DEMO_ESTIMATOR = "LogisticRegression"
RECORD_ATTRIBUTES = ("support_vectors_", "support_", "dual_coef_", "shape_fit_", "_fit_X")

# The 16 inputs, in the order the saved pipeline was fitted on. load_pipeline() compares this list with the
# pipeline's own feature_names_in_, so the two cannot drift apart without the app refusing to predict.
FEATURES = [
    "age", "gender_male", "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
    "genital_thrush", "visual_blurring", "itching", "irritability", "delayed_healing", "partial_paresis",
    "muscle_stiffness", "alopecia", "obesity",
]
SYMPTOMS = FEATURES[2:]

# Ages in the dataset run from 16 to 90 (reports/01_data_validation.json). UCI documents 20 to 65, but 52 of the
# 520 rows are outside that range. The model has only seen ages 16 to 90, so that is the range the app accepts.
AGE_MIN, AGE_MAX = 16, 90

GENDER_CODES = {"Female": 0, "Male": 1}       # the project's recoding (src/prep.py): gender_male
YES_NO_CODES = {"No": 0, "Yes": 1}
CLASS_NAMES = {0: "Negative", 1: "Positive"}  # predict() returns 1 = Positive, 0 = Negative


class ModelUnavailableError(Exception):
    """The saved model is missing, was changed, or does not fit this app. The message is safe to show."""


def _shown(path: Path) -> str:
    """A path for messages: relative to the project, never an absolute path."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_sha256(note_path: Path = MODEL_NOTE) -> str:
    """The SHA-256 of the joblib file as recorded when the model was saved (models/demo_logistic_regression.txt)."""
    try:
        text = note_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelUnavailableError(f"{_shown(note_path)} was not found, so the model cannot be checked") from exc
    match = re.search(r"joblib file SHA-256:\s*([0-9a-f]{64})", text)
    if not match:
        raise ModelUnavailableError(f"{_shown(note_path)} does not record a SHA-256 for the model")
    return match.group(1)


def load_pipeline(path: Path = MODEL_PATH, note_path: Path = MODEL_NOTE):
    """Load the demo model after checking that it is the recorded file and expects the app's inputs.

    A joblib file is a pickle and can run code when it is loaded, so the file is compared with the recorded
    SHA-256 first, and a scikit-learn version mismatch is treated as an error rather than a warning. The loaded
    object must be a Logistic Regression without any attribute that stores training records, so the app can never
    serve a model such as an SVM that carries part of the training data.
    """
    if not path.is_file():
        raise ModelUnavailableError(f"{_shown(path)} was not found")
    if sha256_file(path) != recorded_sha256(note_path):
        raise ModelUnavailableError(f"{_shown(path)} does not match its recorded SHA-256, so it was not loaded")
    try:
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = joblib.load(path)
    except ImportError as exc:
        raise ModelUnavailableError("scikit-learn or joblib is not installed (python -m pip install -r requirements.txt)") from exc
    except Exception as exc:  # a version mismatch or an unreadable pickle
        LOG.warning("Could not load %s", _shown(path), exc_info=True)
        raise ModelUnavailableError(
            f"{_shown(path)} could not be loaded ({type(exc).__name__}); install the versions in requirements.txt"
        ) from exc
    if list(getattr(pipeline, "feature_names_in_", [])) != FEATURES:
        raise ModelUnavailableError("the saved model expects different input columns than this app sends")
    if not hasattr(pipeline, "predict"):
        raise ModelUnavailableError("the saved object is not a fitted pipeline")
    final = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    if type(final).__name__ != DEMO_ESTIMATOR or any(hasattr(final, name) for name in RECORD_ATTRIBUTES):
        raise ModelUnavailableError("the saved file is not the demo-only Logistic Regression, so it was not loaded")
    return pipeline


def _number(value: object) -> float | None:
    """The value as a finite float, or None if it is not a usable number (booleans are not numbers here)."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_answers(answers: Mapping[str, object], labels: Mapping[str, str] | None = None) -> list[str]:
    """Problems with a set of answers (an empty list means the answers can be sent to the model).

    `answers` maps each of the 16 feature names to the person's answer: a whole-number age, "Female" or "Male"
    for gender_male, and "Yes" or "No" for each symptom. `labels` optionally maps feature names to display names.
    """
    name = (lambda feature: labels.get(feature, feature)) if labels else (lambda feature: feature)
    problems: list[str] = []

    unexpected = sorted(set(answers) - set(FEATURES))
    if unexpected:
        problems.append("Unexpected inputs: " + ", ".join(unexpected) + ".")

    age = answers.get("age")
    if age is None or (isinstance(age, str) and not age.strip()):
        problems.append(f"Age is missing: enter a whole number from {AGE_MIN} to {AGE_MAX}.")
    else:
        value = _number(age)
        if value is None:
            problems.append("Age must be a number.")
        elif value != int(value):
            problems.append("Age must be a whole number of years.")
        elif not AGE_MIN <= value <= AGE_MAX:
            problems.append(f"Age {int(value)} is outside the range of ages in the dataset ({AGE_MIN} to {AGE_MAX}).")

    gender = answers.get("gender_male")
    if gender is None:
        problems.append("Gender is missing: choose Female or Male.")
    elif gender not in GENDER_CODES:
        problems.append("Gender must be Female or Male.")

    unanswered = [name(s) for s in SYMPTOMS if answers.get(s) is None]
    invalid = [name(s) for s in SYMPTOMS if answers.get(s) is not None and answers.get(s) not in YES_NO_CODES]
    if unanswered:
        problems.append("Answer Yes or No for: " + ", ".join(unanswered) + ".")
    if invalid:
        problems.append("The answer must be Yes or No for: " + ", ".join(invalid) + ".")
    return problems


def encode_answers(answers: Mapping[str, object]) -> pd.DataFrame:
    """One row with the 16 columns in the saved order, coded as the model was trained (age in years, others 0/1)."""
    problems = validate_answers(answers)
    if problems:
        raise ValueError(" ".join(problems))
    row = {"age": int(float(answers["age"])), "gender_male": GENDER_CODES[answers["gender_male"]]}
    row.update({symptom: YES_NO_CODES[answers[symptom]] for symptom in SYMPTOMS})
    return pd.DataFrame([row], columns=FEATURES).astype("int64")


def predict_class(pipeline, frame: pd.DataFrame) -> str:
    """The predicted dataset class, "Positive" or "Negative". Only predict() is called: no scores or probabilities."""
    if list(frame.columns) != FEATURES:
        raise ValueError("the input columns are not the saved feature order")
    prediction = pipeline.predict(frame)
    if len(prediction) != 1 or int(prediction[0]) not in CLASS_NAMES:
        raise RuntimeError("the model returned something other than one class")
    return CLASS_NAMES[int(prediction[0])]
