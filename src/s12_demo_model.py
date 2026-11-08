"""Demo-only Logistic Regression for the Streamlit app, and the inspection that shows its file holds no records.

Why this exists: the research-selected model is the RBF SVM, but an RBF SVM stores some of its training records
(its support vectors, with their classes) inside the saved file, so that file is kept out of the published
repository. The app uses this separate, much smaller model instead. It is not the selected model and was not
chosen because it performs better.

What it does:
  1. Verifies reports/model_selection.json against its recorded SHA-256 and reads the Logistic Regression
     configuration that cross-validation chose for the primary analysis. No new tuning is done.
  2. Fits that configuration on the full primary TRAINING set (data/processed/train_rows.csv, hash-verified by
     load_training). This script never opens cleaned_all_rows.csv or split_assignment.csv, so no test row is read.
  3. Serialises the model in memory and inspects what the file would contain (audit_bytes). It refuses to write
     the model if the file holds anything that could be a table of records: a large array, long lists, support
     vectors or row indices, classes other than scikit-learn / numpy / joblib, or any training row stored in an array.
  4. Only then writes models/demo_logistic_regression.joblib and models/demo_logistic_regression.txt (which records
     the file's SHA-256, checked by the app before it loads the file).

Run with:  <env python> -I -B src/s12_demo_model.py       (refuses to overwrite existing files)
Writes:    models/demo_logistic_regression.joblib, models/demo_logistic_regression.txt

The helpers audit_bytes / audit_artifact / rejection_reasons are also used by tests/test_app.py, which runs them
on the published file.
"""
from __future__ import annotations

import io
import json
import pickle
import platform
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from mlcommon import FEATURES, MODELS_DIR, REPORTS, build_estimator, load_training, sha256_file, view_frame  # noqa: E402

MODEL_FILE = MODELS_DIR / "demo_logistic_regression.joblib"
NOTE_FILE = MODELS_DIR / "demo_logistic_regression.txt"

# Limits for "this file holds no table of records". The training partition alone is 416 rows x 17 values, about
# 56,000 bytes as 64-bit floats. A Logistic Regression pipeline is a few thousand bytes: 16 coefficients, an
# intercept, the age scaling and the column names.
MAX_FILE_BYTES = 10_000
MAX_VALUES_IN_ARRAYS = 100
MAX_CONTAINER_LENGTH = 32
FORBIDDEN_ATTRIBUTES = {"support_vectors_", "support_", "dual_coef_", "n_support_", "shape_fit_", "_fit_X", "tree_",
                        "estimators_"}
ALLOWED_PREFIXES = ("sklearn.", "numpy.", "joblib.", "builtins.")


def walk(root) -> dict:
    """Every NumPy array, instance attribute name and container length reachable from a loaded model."""
    arrays: list[tuple] = []
    objects: list[np.ndarray] = []
    attributes: set[str] = set()
    longest = [0]
    seen: set[int] = set()

    def visit(obj, where: str) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            arrays.append((where, tuple(obj.shape), str(obj.dtype), int(obj.size)))
            objects.append(obj)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            longest[0] = max(longest[0], len(obj))
            for i, item in enumerate(obj):
                visit(item, f"{where}[{i}]")
        elif isinstance(obj, dict):
            longest[0] = max(longest[0], len(obj))
            for key, value in obj.items():
                visit(value, f"{where}[{key!r}]")
        elif hasattr(obj, "__dict__") and not isinstance(obj, (type, str, bytes)):
            for name, value in vars(obj).items():
                attributes.add(name)
                visit(value, f"{where}.{name}")

    visit(root, "model")
    return {"arrays": arrays, "array_objects": objects, "attributes": attributes, "longest_container": longest[0]}


def audit_bytes(raw: bytes) -> dict:
    """What a serialised model really contains.

    The bytes are loaded while every class the file refers to is recorded, then the loaded object is walked for
    arrays, attributes and containers, and the readable strings in the raw bytes are listed.
    """
    referenced: set[str] = set()
    original = pickle._Unpickler.find_class

    def spy(self, module, name):
        referenced.add(f"{module}.{name}")
        return original(self, module, name)

    pickle._Unpickler.find_class = spy
    try:
        model = joblib.load(io.BytesIO(raw))
    finally:
        pickle._Unpickler.find_class = original
    found = walk(model)
    sizes = [a[3] for a in found["arrays"]]
    return {
        "model": model, "bytes": len(raw), "classes": sorted(referenced), "arrays": found["arrays"],
        "array_objects": found["array_objects"], "values_in_arrays": sum(sizes), "largest_array": max(sizes, default=0),
        "attributes": sorted(found["attributes"]), "longest_container": found["longest_container"],
        "strings": sorted({m.decode("ascii") for m in re.findall(rb"[\x20-\x7e]{4,}", raw)}),
    }


def audit_artifact(path) -> dict:
    return audit_bytes(Path(path).read_bytes())


def rejection_reasons(audit: dict) -> list[str]:
    """Reasons the file must not be published as a row-free model. An empty list means none was found."""
    problems = []
    if audit["bytes"] > MAX_FILE_BYTES:
        problems.append(f"the file is {audit['bytes']} bytes (limit {MAX_FILE_BYTES})")
    if audit["values_in_arrays"] > MAX_VALUES_IN_ARRAYS:
        problems.append(f"arrays hold {audit['values_in_arrays']} values (limit {MAX_VALUES_IN_ARRAYS})")
    tables = [a for a in audit["arrays"] if len(a[1]) >= 2 and a[1][0] > 1]
    if tables:
        problems.append(f"arrays with more than one row: {[(a[0], a[1]) for a in tables]}")
    if audit["longest_container"] > MAX_CONTAINER_LENGTH:
        problems.append(f"a list or dict has {audit['longest_container']} entries (limit {MAX_CONTAINER_LENGTH})")
    forbidden = sorted(set(audit["attributes"]) & FORBIDDEN_ATTRIBUTES)
    if forbidden:
        problems.append(f"attributes that store training records: {forbidden}")
    outside = [c for c in audit["classes"] if not c.startswith(ALLOWED_PREFIXES)]
    if outside:
        problems.append(f"the file refers to classes outside scikit-learn / numpy / joblib: {outside}")
    return problems


def rows_found_in_arrays(arrays: list[np.ndarray], rows: np.ndarray) -> int:
    """How many of the given 16-value rows appear, exactly, as a row of some stored numeric array."""
    width = rows.shape[1]
    found = 0
    for array in arrays:
        if array.dtype.kind not in "fiu" or array.size < width or array.size % width:
            continue
        table = array.reshape(-1, width).astype(float)
        found += sum(bool(np.isclose(table, row, atol=1e-9).all(axis=1).any()) for row in rows)
    return found


def main() -> None:
    existing = [p.name for p in (MODEL_FILE, NOTE_FILE) if p.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite {existing}. Use a copy of the project to run this script again.")

    selection_path = REPORTS / "model_selection.json"
    if sha256_file(selection_path) != (REPORTS / "model_selection.sha256").read_text().split()[0]:
        raise SystemExit("ABORT: reports/model_selection.json does not match reports/model_selection.sha256")
    params = json.loads(selection_path.read_text())["views"]["primary"]["models"]["Logistic Regression"]["params"]

    frame = load_training()                         # verifies the SHA-256 of train_rows.csv and cv_folds.csv
    view = view_frame(frame, "primary")
    X, y = view[FEATURES], view["label"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # a convergence warning would be an error
        model = build_estimator("Logistic Regression", params).fit(X, y)
        refit = build_estimator("Logistic Regression", params).fit(X, y)
    assert np.array_equal(model[-1].coef_, refit[-1].coef_), "the fit is not deterministic"

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    raw = buffer.getvalue()
    audit = audit_bytes(raw)
    problems = rejection_reasons(audit)
    raw_rows = X.to_numpy(dtype=float)
    scaled_rows = model.named_steps["prep"].transform(X)
    leaked = rows_found_in_arrays(audit["array_objects"], raw_rows) + rows_found_in_arrays(audit["array_objects"], scaled_rows)
    if leaked:
        problems.append(f"{leaked} training row(s) appear inside stored arrays")
    if problems:
        raise SystemExit("REJECTED, nothing was written: " + "; ".join(problems))

    MODELS_DIR.mkdir(exist_ok=True)
    MODEL_FILE.write_bytes(raw)
    reloaded = joblib.load(MODEL_FILE)
    assert np.array_equal(reloaded.predict(X), model.predict(X)), "the reloaded model predicts differently"
    assert np.array_equal(reloaded.decision_function(X), model.decision_function(X)), "the reloaded model scores differently"

    n_arrays, n_values = len(audit["arrays"]), audit["values_in_arrays"]
    NOTE_FILE.write_text("\n".join([
        f"Demo-only model for the Streamlit app: Logistic Regression, parameters {json.dumps(params)}",
        "It is NOT the model the project's selection rule chose. That is the RBF SVM, whose saved file stores 127 of the "
        "416 training records as support vectors and is therefore not published (see the README).",
        "Configuration: the Logistic Regression configuration that cross-validation chose for the primary analysis "
        "(reports/model_selection.json). No new tuning was done.",
        f"Fitted on: data/processed/train_rows.csv, all {len(X)} training rows of the primary analysis; no test row was read.",
        f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}, python {platform.python_version()}",
        "Input: a pandas DataFrame with exactly these 16 columns in this order (age in years; every other column 0/1):",
        "  " + ", ".join(FEATURES),
        "predict() returns the predicted dataset class: 1 = Positive, 0 = Negative.",
        "The app shows the predicted class only. Probabilities from this model were not calibrated or evaluated, so none "
        "is shown (plan section 3.13).",
        "This file was not evaluated separately. The report's Logistic Regression row (primary analysis) evaluated this "
        "configuration once on the test set.",
        f"Inspection of the saved file (src/s12_demo_model.py, run before the file was written): {audit['bytes']:,} bytes; "
        f"{n_arrays} arrays holding {n_values} values in total (largest: {audit['largest_array']}, the coefficients); "
        "it refers only to scikit-learn, numpy and joblib classes; it has no support vectors, row indices or per-row "
        "values, and no training row appears in any stored array.",
        f"joblib file SHA-256: {sha256_file(MODEL_FILE)}",
        "Built only from scikit-learn components.", ""]))

    a = pd.DataFrame([{**dict.fromkeys(FEATURES, 0), "age": 45, "gender_male": 1}])[FEATURES]
    b = a.assign(polyuria=1, polydipsia=1)
    label = {0: "Negative", 1: "Positive"}
    print(f"demo model saved: Logistic Regression {json.dumps(params)}, fitted on {len(X)} training rows, no test row read")
    print(f"file {audit['bytes']:,} bytes | {n_arrays} arrays, {n_values} values, largest {audit['largest_array']} | "
          f"references only: {sorted({c.split('.')[0] for c in audit['classes']})} | training rows found in arrays: {leaked}")
    print(f"synthetic example A -> {label[int(reloaded.predict(a)[0])]}, B -> {label[int(reloaded.predict(b)[0])]} (class only)")


if __name__ == "__main__":
    main()
