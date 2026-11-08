"""Model selection by cross-validation on the training data only.

Reads only data/processed/train_rows.csv and data/splits/cv_folds.csv. It never opens the split
assignment, the full cleaned file or any test row.

Selection rule, declared before any result existed. It is applied separately inside each view, with
identical grids, folds and rule:
  1. Within a model: the configuration with the highest mean CV ROC-AUC (exact ties: the first one listed).
  2. Across the four models: best = the highest mean CV ROC-AUC; sd_best = the SD (ddof=1) of that model's
     15 fold ROC-AUC values. Every model with mean >= best - sd_best is eligible. The eligible model that is
     simplest in the fixed order Logistic Regression < Categorical NB < RBF SVM < Random Forest is chosen.
  3. The primary view's choice is the demo model. The sensitivity view's choice is for comparison only.

Run with:  <env python> -I -B src/s08_cv_select.py
Writes:    reports/model_cv_results.csv, reports/model_selection.json, reports/model_selection.sha256
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402

from mlcommon import (BASELINE, EXPECTED_SHA256, FEATURES, GRIDS, METRICS, MODEL_ORDER, REPORTS, VIEWS,  # noqa: E402
                      build_estimator, evaluate_estimator, jsonable, load_training, make_splits, sha256_file,
                      summarize, view_frame)

t0 = time.time()
frame = load_training()                      # verifies the SHA-256 of train_rows.csv and cv_folds.csv first
rows: list[dict] = []
views_out: dict = {}

for view in VIEWS:
    vf = view_frame(frame, view)
    X, y = vf[FEATURES], vf["label"].to_numpy()
    splits = make_splits(vf)                 # asserts both classes and group separation in all 15 splits
    chosen: dict = {}
    baseline_out = None
    for model in [BASELINE] + MODEL_ORDER:
        cfgs = []
        for cfg_id, params in enumerate(GRIDS[model]):
            est0 = build_estimator(model, params)
            vals = {m: [] for m in METRICS}
            for (_r, _k, tr, va) in splits:
                est = clone(est0).fit(X.iloc[tr], y[tr])
                m, _counts, _score, _pred = evaluate_estimator(est, X.iloc[va], y[va])
                for key in METRICS:
                    vals[key].append(m[key])
            if model == BASELINE:
                assert all(abs(v - 0.5) < 1e-12 for v in vals["roc_auc"]), "baseline ROC-AUC must be 0.5 in every fold"
            agg = {key: summarize(vals[key]) for key in METRICS}
            cfgs.append({"cfg_id": cfg_id, "params": params, "vals": vals, "agg": agg})
        best_i = int(np.argmax([c["agg"]["roc_auc"][0] for c in cfgs]))     # first maximum wins exact ties
        for i, c in enumerate(cfgs):
            row = {"view": view, "model": model, "config_id": c["cfg_id"], "params": json.dumps(c["params"]),
                   "n_folds": len(splits), "selected_within_model": i == best_i}
            for key in METRICS:
                mean, sd, n_und = c["agg"][key]
                row[f"{key}_mean"], row[f"{key}_sd"], row[f"{key}_n_undefined"] = mean, sd, n_und
            rows.append(row)
        pick = cfgs[best_i]
        entry = {"config_id": pick["cfg_id"], "params": pick["params"],
                 "roc_auc_mean": pick["agg"]["roc_auc"][0], "roc_auc_sd": pick["agg"]["roc_auc"][1],
                 "fold_roc_auc": pick["vals"]["roc_auc"],
                 "metrics_mean": {k: pick["agg"][k][0] for k in METRICS},
                 "metrics_sd": {k: pick["agg"][k][1] for k in METRICS},
                 "metrics_n_undefined": {k: pick["agg"][k][2] for k in METRICS}}
        if model == BASELINE:
            baseline_out = entry
        else:
            chosen[model] = entry
        print(f"[{view:11s}] {model:24s} best config {json.dumps(pick['params']):42s} "
              f"CV ROC-AUC {pick['agg']['roc_auc'][0]:.4f} +/- {pick['agg']['roc_auc'][1]:.4f}", flush=True)
    best_model = max(MODEL_ORDER, key=lambda m: chosen[m]["roc_auc_mean"])      # exact ties: simplest first
    best_mean, sd_best = chosen[best_model]["roc_auc_mean"], chosen[best_model]["roc_auc_sd"]
    threshold = best_mean - sd_best
    eligible = [m for m in MODEL_ORDER if chosen[m]["roc_auc_mean"] >= threshold]
    chosen_model = eligible[0]                                                  # MODEL_ORDER is simplest first
    views_out[view] = {"n_rows": int(len(vf)), "n_profiles": int(vf["profile_id"].nunique()), "n_splits": len(splits),
                       "models": chosen, "baseline": baseline_out, "best_model": best_model, "best_mean": best_mean,
                       "sd_best": sd_best, "threshold": threshold, "eligible": eligible, "chosen_model": chosen_model}
    print(f"[{view:11s}] best = {best_model} ({best_mean:.4f}), SD {sd_best:.4f}, threshold {threshold:.4f}, "
          f"eligible = {eligible}, CHOSEN = {chosen_model}\n", flush=True)

pd.DataFrame(rows).to_csv(REPORTS / "model_cv_results.csv", index=False)
selection = {
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scikit-learn": sklearn.__version__},
    "inputs_sha256": {p.name: h for p, h in EXPECTED_SHA256.items()},
    "rule": ["1. Within a model: highest mean CV ROC-AUC; exact ties go to the first configuration listed.",
             "2. Across the four models: best = highest mean CV ROC-AUC; sd_best = SD (ddof=1) of that model's 15 fold ROC-AUC "
             "values; every model with mean >= best - sd_best is eligible; choose the eligible model that is simplest in the "
             "fixed simplicity order.",
             "3. The primary view's choice is the demo model; the sensitivity view's choice is for comparison only."],
    "simplicity_order": MODEL_ORDER,
    "grids": GRIDS,
    "views": views_out,
}
sel_path = REPORTS / "model_selection.json"
sel_path.write_text(json.dumps(jsonable(selection), indent=2, sort_keys=True) + "\n")
digest = sha256_file(sel_path)
(REPORTS / "model_selection.sha256").write_text(f"{digest}  reports/model_selection.json\n")
print(f"wrote reports/model_cv_results.csv ({len(rows)} rows) and reports/model_selection.json")
print(f"SHA-256 of model_selection.json: {digest}")
print(f"elapsed {time.time() - t0:.1f} s")
