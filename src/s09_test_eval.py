"""One final test evaluation.

First verifies reports/model_selection.json against reports/model_selection.sha256 and aborts on a mismatch.
Only then are the test rows loaded. Every configuration was fixed by cross-validation in s08; no test result
changes a configuration, a model choice or the selection.

For each view and each of the five models (baseline included) the selected configuration is refitted on the
full training set of that view and evaluated once on that view's test set. Uncertainty comes from a group
bootstrap of whole profile groups on the fixed predictions (1,000 repetitions, seed 2026).

Run with:  <env python> -I -B src/s09_test_eval.py
Writes:    reports/model_test_results.json, reports/model_test_results.csv,
           models/primary_demo_pipeline.joblib, models/primary_demo_pipeline.txt
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from mlcommon import (BASELINE, BOOTSTRAP_REPS, BOOTSTRAP_SEED, FEATURES, METRICS, MODEL_ORDER, MODELS_DIR,  # noqa: E402
                      REPORTS, ROOT, VIEWS, bootstrap_summary, build_estimator, draw_group_weights,
                      evaluate_estimator, jsonable, load_training, positive_score, roc_points, sha256_file,
                      verify_inputs, view_frame, weighted_metrics)

SPLIT_ASSIGN = ROOT / "data" / "splits" / "split_assignment.csv"
CLEANED = ROOT / "data" / "processed" / "cleaned_all_rows.csv"
TEST_INPUT_SHA256 = {
    SPLIT_ASSIGN: "8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e",
    CLEANED: "2678581eedfbffae3d1d2a4668b3cbfffca78bd5369bcbc2a6981ac4cd810ff8",
}


def load_test(train_row_ids) -> dict:
    """The only place in s09 where test rows are read."""
    verify_inputs(TEST_INPUT_SHA256)
    assign = pd.read_csv(SPLIT_ASSIGN)
    cleaned = pd.read_csv(CLEANED)
    test_ids = set(assign.loc[assign["partition"] == "test", "row_id"])
    test = cleaned[cleaned["row_id"].isin(test_ids)].reset_index(drop=True)
    assert test.columns.tolist()[4:20] == FEATURES
    assert len(test) == 104 and test["profile_id"].nunique() == 49 and test["label"].nunique() == 2
    assert not (set(test["row_id"]) & set(train_row_ids)), "a test row id is also a training row id"
    sens = test[test["is_first_copy"] == 1].reset_index(drop=True)
    assert len(sens) == 49 and sens["profile_id"].nunique() == 49 and sens["label"].nunique() == 2
    return {"primary": test, "sensitivity": sens}


t0 = time.time()
sel_path = REPORTS / "model_selection.json"
recorded = (REPORTS / "model_selection.sha256").read_text().split()[0]
actual = sha256_file(sel_path)
if actual != recorded:
    raise SystemExit("ABORT: reports/model_selection.json does not match reports/model_selection.sha256")
print("selection file verified:", actual, flush=True)
selection = json.loads(sel_path.read_text())

frame = load_training()
tests = load_test(frame["row_id"])           # first read of any test row

results: dict = {}
fitted: dict = {}
for view in VIEWS:
    vf = view_frame(frame, view)
    Xtr, ytr = vf[FEATURES], vf["label"].to_numpy()
    te = tests[view]
    Xte, yte = te[FEATURES], te["label"].to_numpy()
    W = draw_group_weights(te["profile_id"].to_numpy(), BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    sv = selection["views"][view]
    out = {"n_train_rows": int(len(vf)), "n_test_rows": int(len(te)), "n_test_groups": int(te["profile_id"].nunique()),
           "test_positive": int(yte.sum()), "test_negative": int(len(yte) - yte.sum()), "models": {}}
    for model in [BASELINE] + MODEL_ORDER:
        params = {} if model == BASELINE else sv["models"][model]["params"]
        est = build_estimator(model, params).fit(Xtr, ytr)
        m, counts, score, pred = evaluate_estimator(est, Xte, yte)
        if model == BASELINE:
            assert abs(m["roc_auc"] - 0.5) < 1e-12
        wm = weighted_metrics(np.ones((1, len(yte))), yte, pred, score)    # self-check of the bootstrap code
        for key in METRICS:
            a, b = float(wm[key][0]), m[key]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12, (view, model, key, a, b)
        boot = bootstrap_summary(W, yte, pred, score)
        fpr, tpr = roc_points(yte, score)
        tn, fp, fn, tp = counts
        out["models"][model] = {
            "params": params, "metrics": m, "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "ci95": {k: [boot[k]["lo"], boot[k]["hi"]] for k in METRICS},
            "n_undefined_bootstrap": {k: boot[k]["n_undefined"] for k in METRICS},
            "roc": {"fpr": fpr, "tpr": tpr},
        }
        fitted[(view, model)] = (pred, score)
        print(f"[{view:11s}] {model:24s} acc {m['accuracy']:.3f}  prec {m['precision']:.3f}  rec {m['recall']:.3f}  "
              f"spec {m['specificity']:.3f}  F1 {m['f1']:.3f}  AUC {m['roc_auc']:.3f}  "
              f"AUC 95% [{boot['roc_auc']['lo']:.3f}, {boot['roc_auc']['hi']:.3f}]  CM {counts}", flush=True)
    results[view] = out

# Demo pipeline: the primary view's CV-selected model, refitted on the full primary training set.
demo_model = selection["views"]["primary"]["chosen_model"]
demo_params = selection["views"]["primary"]["models"][demo_model]["params"]
vf = view_frame(frame, "primary")
demo = build_estimator(demo_model, demo_params).fit(vf[FEATURES], vf["label"].to_numpy())
Xte = tests["primary"][FEATURES]
pred_a, score_a = np.asarray(demo.predict(Xte)).astype(int), positive_score(demo, Xte)
pred_prev, score_prev = fitted[("primary", demo_model)]
assert np.array_equal(pred_a, pred_prev) and np.array_equal(score_a, score_prev), "refit is not deterministic"
MODELS_DIR.mkdir(exist_ok=True)
joblib_path = MODELS_DIR / "primary_demo_pipeline.joblib"
joblib.dump(demo, joblib_path)
reloaded = joblib.load(joblib_path)
assert np.array_equal(np.asarray(reloaded.predict(Xte)).astype(int), pred_a), "reloaded pipeline predicts differently"
assert np.array_equal(positive_score(reloaded, Xte), score_a), "reloaded pipeline scores differ"
score_kind = "predicted probability of Positive" if hasattr(demo, "predict_proba") else "decision value"
note = ("Requires src/mlcommon.py to be importable (it defines the fixed age-band function used by this model)."
        if demo_model == "Categorical NB" else "Built only from scikit-learn components.")
(MODELS_DIR / "primary_demo_pipeline.txt").write_text("\n".join([
    f"Model: {demo_model} (primary view, chosen by cross-validation), parameters {json.dumps(demo_params)}",
    "Fitted on: data/processed/train_rows.csv, all 416 training rows, no test row",
    f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}, python {platform.python_version()}",
    "Input: a pandas DataFrame with exactly these 16 columns in this order (age in years; every other column 0/1):",
    "  " + ", ".join(FEATURES),
    "predict() returns the predicted dataset class: 1 = Positive, 0 = Negative.",
    f"Scores used for ROC-AUC: {score_kind}. They are not displayed in the app (plan section 3.13).",
    "Reload check: the reloaded pipeline gives identical test predictions and identical scores: True",
    f"joblib file SHA-256: {sha256_file(joblib_path)}",
    note, ""]))
print(f"demo pipeline: {demo_model} {json.dumps(demo_params)} saved; reload gives identical test predictions and scores", flush=True)

doc = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "selection_sha256": actual,
       "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
       "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "profile group",
                     "interval": "2.5th to 97.5th percentile of the defined resamples"},
       "demo_pipeline": {"model": demo_model, "params": demo_params, "file": "models/primary_demo_pipeline.joblib",
                         "reload_check_identical": True},
       "views": results}
(REPORTS / "model_test_results.json").write_text(json.dumps(jsonable(doc), indent=2, sort_keys=True) + "\n")
flat = []
for view, vres in results.items():
    for model, e in vres["models"].items():
        row = {"view": view, "model": model, "params": json.dumps(e["params"]), "n_test_rows": vres["n_test_rows"],
               "n_test_groups": vres["n_test_groups"], **{k: v for k, v in e["confusion"].items()}}
        for key in METRICS:
            row[key] = e["metrics"][key]
            row[f"{key}_ci_lo"], row[f"{key}_ci_hi"] = e["ci95"][key]
            row[f"{key}_n_undefined_bootstrap"] = e["n_undefined_bootstrap"][key]
        flat.append(row)
pd.DataFrame(flat).to_csv(REPORTS / "model_test_results.csv", index=False)
print(f"wrote reports/model_test_results.json and .csv; elapsed {time.time() - t0:.1f} s")
