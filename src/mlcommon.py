"""Shared helpers for the model scripts: s08 (CV selection), s09 (test evaluation), s10 (FNN) and s11 (report).

Nothing in this module reads the test rows, the split assignment or the full cleaned file. The test
loaders live in s09 and s10 and run only after the frozen selection file has been verified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"
TRAIN_ROWS = ROOT / "data" / "processed" / "train_rows.csv"
CV_FOLDS = ROOT / "data" / "splits" / "cv_folds.csv"
EXPECTED_SHA256 = {
    TRAIN_ROWS: "05dfd30e3ff96f102d95e4f5198037071f5e4484b87f695f8635a55a6e1370e8",
    CV_FOLDS: "f71888b1ba497c57524de495bdd9d93133bc7ff94541228850b6773fa58bd696",
}

FEATURES = [
    "age", "gender_male", "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
    "genital_thrush", "visual_blurring", "itching", "irritability", "delayed_healing", "partial_paresis",
    "muscle_stiffness", "alopecia", "obesity",
]
VIEWS = ("primary", "sensitivity")
VIEW_TITLE = {"primary": "Primary analysis (all 520 records)",
              "sensitivity": "Sensitivity analysis (251 unique profiles)"}
METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")
METRIC_TITLE = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "specificity": "Specificity",
                "f1": "F1", "roc_auc": "ROC-AUC"}
BASELINE = "Majority-class baseline"
FNN_NAME = "FNN (secondary experiment)"
# Fixed simplicity order, declared before any result existed (simplest first).
MODEL_ORDER = ["Logistic Regression", "Categorical NB", "RBF SVM", "Random Forest"]
GRIDS = {
    BASELINE: [{}],
    "Logistic Regression": [{"C": c} for c in (0.01, 0.1, 1, 10)],
    "Categorical NB": [{"alpha": a} for a in (0.1, 0.5, 1.0)],
    "RBF SVM": [{"C": c, "gamma": g} for c in (0.1, 1, 10) for g in ("scale", 0.01, 0.1)],
    "Random Forest": [{"max_depth": d, "min_samples_leaf": leaf} for d in (None, 5, 10) for leaf in (1, 3)],
}
BOOTSTRAP_REPS, BOOTSTRAP_SEED = 1000, 2026


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(expected: dict) -> None:
    bad = [p.name for p, h in expected.items() if sha256_file(p) != h]
    if bad:
        raise SystemExit("Frozen input changed or missing (SHA-256 mismatch): " + ", ".join(bad))


def jsonable(obj):
    """Make nested results JSON-safe: numpy scalars to Python, NaN to null."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# --- training data, views and frozen folds ---------------------------------------------------------
def load_training() -> pd.DataFrame:
    """Training rows plus the three frozen fold columns. Verifies the SHA-256 of both inputs first."""
    verify_inputs(EXPECTED_SHA256)
    tr = pd.read_csv(TRAIN_ROWS)
    cv = pd.read_csv(CV_FOLDS)
    assert tr.columns.tolist()[4:20] == FEATURES, "feature column order changed"
    assert len(tr) == 416 and len(cv) == 416
    assert tr["row_id"].tolist() == cv["row_id"].tolist(), "cv_folds.csv rows differ from train_rows.csv"
    assert tr["profile_id"].tolist() == cv["profile_id"].tolist()
    assert tr["is_first_copy"].tolist() == cv["is_first_copy"].tolist()
    return pd.concat([tr, cv[["cv_r0_fold", "cv_r1_fold", "cv_r2_fold"]]], axis=1)


def view_frame(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    if view == "primary":
        vf = frame.reset_index(drop=True)
        assert len(vf) == 416 and vf["profile_id"].nunique() == 202
    elif view == "sensitivity":
        vf = frame[frame["is_first_copy"] == 1].reset_index(drop=True)
        assert len(vf) == 202 and vf["profile_id"].nunique() == 202
    else:
        raise ValueError(view)
    return vf


def make_splits(vf: pd.DataFrame) -> list:
    """The 15 frozen (repeat, fold, train_idx, validation_idx) splits of a view, with group-separation asserts."""
    y, g = vf["label"].to_numpy(), vf["profile_id"].to_numpy()
    splits = []
    for r in range(3):
        fold = vf[f"cv_r{r}_fold"].to_numpy()
        for k in range(5):
            va, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
            assert len(set(y[tr])) == 2 and len(set(y[va])) == 2, f"a class is missing (repeat {r}, fold {k})"
            assert not (set(g[tr]) & set(g[va])), f"a profile is on both sides (repeat {r}, fold {k})"
            splits.append((r, k, tr, va))
    assert len(splits) == 15
    return splits


# --- models ----------------------------------------------------------------------------------------------
def age_bands(X):
    """Fixed left-closed bands <30, 30-39, 40-49, 50-59, >=60 as codes 0-4. Nothing is learned."""
    out = X.copy()
    out["age"] = np.digitize(X["age"].to_numpy(), [30, 40, 50, 60])
    return out.to_numpy(dtype=int)


def _age_scaler():
    return ColumnTransformer([("age", StandardScaler(), ["age"])], remainder="passthrough",
                             verbose_feature_names_out=False)


def build_estimator(model: str, params: dict):
    if model == BASELINE:
        return DummyClassifier(strategy="most_frequent")
    if model == "Logistic Regression":
        return Pipeline([("prep", _age_scaler()),
                         ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, **params))])
    if model == "Categorical NB":
        return Pipeline([("bands", FunctionTransformer(age_bands, validate=False)),
                         ("clf", CategoricalNB(min_categories=[5] + [2] * 15, **params))])
    if model == "RBF SVM":
        return Pipeline([("prep", _age_scaler()), ("clf", SVC(kernel="rbf", **params))])
    if model == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1, **params)
    raise ValueError(model)


def positive_score(est, X) -> np.ndarray:
    """Score for the Positive class: predicted probability, or the decision value for the SVM."""
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(X))[:, 1]
    return np.asarray(est.decision_function(X))


# --- metrics ---------------------------------------------------------------------------------------------
def confusion_counts(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return tn, fp, fn, tp


def metrics_from_counts(tn, fp, fn, tp) -> dict:
    """Accuracy, precision, recall, specificity and F1. A 0/0 metric is NaN (undefined), never 0.
    F1 is computed as 2TP / (2TP + FP + FN)."""
    def div(a, b):
        return float(a / b) if b > 0 else float("nan")
    return {"accuracy": div(tp + tn, tn + fp + fn + tp), "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "specificity": div(tn, tn + fp), "f1": div(2 * tp, 2 * tp + fp + fn)}


def auc_score(y, score) -> float:
    return float(roc_auc_score(y, score)) if len(set(np.asarray(y).tolist())) == 2 else float("nan")


def evaluate_estimator(est, X, y):
    """Metrics, confusion counts, scores and predictions of a fitted estimator on (X, y)."""
    pred = np.asarray(est.predict(X)).astype(int)
    score = positive_score(est, X)
    counts = confusion_counts(y, pred)
    m = metrics_from_counts(*counts)
    m["roc_auc"] = auc_score(y, score)
    return m, counts, score, pred


def summarize(values) -> tuple:
    """(mean, SD with ddof=1, number undefined) over the defined values."""
    a = np.asarray(values, dtype=float)
    ok = ~np.isnan(a)
    mean = float(a[ok].mean()) if ok.any() else float("nan")
    sd = float(a[ok].std(ddof=1)) if ok.sum() > 1 else float("nan")
    return mean, sd, int((~ok).sum())


# --- group bootstrap on a fixed test set -----------------------------------------------------------------
def draw_group_weights(groups, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Row weights for `reps` bootstrap resamples of whole profile groups (with replacement)."""
    uniq, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = len(uniq)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(reps, n_groups))
    counts = np.zeros((reps, n_groups))
    for r in range(reps):
        counts[r] = np.bincount(draws[r], minlength=n_groups)
    return counts[:, inverse]


def _ratio(a, b):
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def _pair_matrix(y, score):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    sp, sn = score[pos][:, None], score[neg][None, :]
    return pos, neg, (sp > sn).astype(float) + 0.5 * (sp == sn).astype(float)


def weighted_metrics(W, y, pred, score) -> dict:
    """Metric values for every weight row of W (each row = one bootstrap resample). NaN = undefined."""
    y, pred, score = np.asarray(y).astype(int), np.asarray(pred).astype(int), np.asarray(score, dtype=float)
    tp = W @ ((y == 1) & (pred == 1)).astype(float)
    tn = W @ ((y == 0) & (pred == 0)).astype(float)
    fp = W @ ((y == 0) & (pred == 1)).astype(float)
    fn = W @ ((y == 1) & (pred == 0)).astype(float)
    vals = {"accuracy": _ratio(tp + tn, tp + tn + fp + fn), "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}
    pos, neg, M = _pair_matrix(y, score)
    Wp, Wn = W[:, pos], W[:, neg]
    vals["roc_auc"] = _ratio(np.einsum("ri,ij,rj->r", Wp, M, Wn), Wp.sum(axis=1) * Wn.sum(axis=1))
    return vals


def bootstrap_summary(W, y, pred, score) -> dict:
    """2.5th and 97.5th percentiles over the defined resamples, plus the number of undefined resamples."""
    out = {}
    for key, v in weighted_metrics(W, y, pred, score).items():
        ok = ~np.isnan(v)
        lo, hi = np.percentile(v[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
        out[key] = {"lo": float(lo), "hi": float(hi), "n_undefined": int((~ok).sum())}
    return out


def roc_points(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return [round(float(x), 6) for x in fpr], [round(float(x), 6) for x in tpr]
