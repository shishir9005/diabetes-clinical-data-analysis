"""Build reports/09_model_results.md and the two figures from the saved result files. No model is fitted here.

Inputs: reports/model_cv_results.csv (not needed here), reports/model_selection.json, reports/model_test_results.json
and, if the optional FNN ran, reports/fnn_results.json.

Run with:  MPLCONFIGDIR=<temporary folder> <env python> -I -B src/s11_build_report.py
Writes:    reports/09_model_results.md, reports/figures/fig05_cv_roc_auc_by_model.png,
           reports/figures/fig06_test_roc_curves.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mlcommon import (BASELINE, FNN_NAME, METRIC_TITLE, METRICS, MODEL_ORDER, REPORTS, VIEW_TITLE, VIEWS)  # noqa: E402

FIG_DIR = REPORTS / "figures"
sel = json.loads((REPORTS / "model_selection.json").read_text())
tst = json.loads((REPORTS / "model_test_results.json").read_text())
fnn_path = REPORTS / "fnn_results.json"
fnn = json.loads(fnn_path.read_text()) if fnn_path.exists() else None
fnn_cv_csv = pd.read_csv(REPORTS / "fnn_cv_results.csv") if fnn else None
ALL_MODELS = [BASELINE] + MODEL_ORDER + ([FNN_NAME] if fnn else [])
VIEW_COLOR = {"primary": "#0072B2", "sensitivity": "#D55E00"}
MODEL_COLOR = {"Logistic Regression": "#0072B2", "Categorical NB": "#E69F00", "RBF SVM": "#009E73",
               "Random Forest": "#CC79A7", FNN_NAME: "#56B4E9"}
SHORT = {"primary": "primary", "sensitivity": "sensitivity"}


def isnan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def f3(v) -> str:
    return "undefined" if isnan(v) else f"{v:.3f}"


def pm(mean, sd) -> str:
    if isnan(mean):
        return "undefined"
    return f"{mean:.3f}" if isnan(sd) else f"{mean:.3f} ± {sd:.3f}"


def with_ci(v, ci) -> str:
    if isnan(v):
        return "undefined"
    lo, hi = ci
    return f"{v:.3f}" if isnan(lo) or isnan(hi) else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def diff(a, b) -> str:
    return "undefined" if isnan(a) or isnan(b) else f"{a - b:+.3f}"


def cfg(params) -> str:
    return "most frequent class" if not params else ", ".join(f"{k}={v}" for k, v in params.items())


def table(head, rows) -> list:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def cv_entry(view, name) -> dict:
    sv = sel["views"][view]
    if name == BASELINE:
        return sv["baseline"]
    if name == FNN_NAME:
        c = fnn["views"][view]["cv"]
        return {"metrics_mean": c["metrics_mean"], "metrics_sd": c["metrics_sd"], "params": None,
                "roc_auc_mean": c["metrics_mean"]["roc_auc"], "roc_auc_sd": c["metrics_sd"]["roc_auc"]}
    return sv["models"][name]


def test_entry(view, name) -> dict:
    if name == FNN_NAME:
        s = fnn["views"][view]["test"]["seed_averaged"]
        return {"metrics": s["metrics"], "ci95": s["ci95"], "confusion": s["confusion"], "roc": s["roc"],
                "n_undefined_bootstrap": s["n_undefined_bootstrap"]}
    return tst["views"][view]["models"][name]


# --- figures ---------------------------------------------------------------------------------------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 11})
FIG_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8.2, 0.8 * len(ALL_MODELS) + 1.9))
for i, name in enumerate(ALL_MODELS):
    for view, off in (("primary", -0.16), ("sensitivity", 0.16)):
        e = cv_entry(view, name)
        ax.errorbar(e["roc_auc_mean"], i + off, xerr=e["roc_auc_sd"], fmt="o", color=VIEW_COLOR[view], capsize=3,
                    markersize=6, label=VIEW_TITLE[view] if i == 0 else None)
ax.set_yticks(range(len(ALL_MODELS)))
ax.set_yticklabels(ALL_MODELS)
ax.invert_yaxis()
ax.set_xlim(0.45, 1.02)
ax.axvline(0.5, color="#9a9a96", lw=0.8, ls=":")
ax.set_xlabel("Cross-validated ROC-AUC (mean ± SD over the 15 frozen folds)")
ax.set_title("Cross-validation ROC-AUC by model (training data only)")
ax.grid(axis="x", color="#e4e3df")
ax.legend(loc="lower left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig05_cv_roc_auc_by_model.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), sharey=True)
for ax, view in zip(axes, VIEWS):
    for name in ALL_MODELS:
        e = test_entry(view, name)
        auc = e["metrics"]["roc_auc"]
        if name == BASELINE:
            ax.plot([0, 1], [0, 1], ls="--", color="#9a9a96", lw=1.2, label=f"{name} (AUC {auc:.3f})")
        else:
            ax.plot(e["roc"]["fpr"], e["roc"]["tpr"], color=MODEL_COLOR[name], lw=1.8, label=f"{name} (AUC {auc:.3f})")
    ax.set_title(VIEW_TITLE[view], fontsize=10)
    ax.set_xlabel("False positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#e4e3df")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
axes[0].set_ylabel("True positive rate (recall)")
fig.suptitle("Test ROC curves (one evaluation per view)", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig06_test_roc_curves.png", dpi=150)
plt.close(fig)

# --- tables ---------------------------------------------------------------------------------------------
def cv_table(view) -> list:
    head = ["Model", "Selected configuration"] + [METRIC_TITLE[m] for m in METRICS]
    rows = []
    for name in ALL_MODELS:
        e = cv_entry(view, name)
        conf = "most frequent class" if name == BASELINE else ("fixed architecture; epochs by inner early stopping" if name == FNN_NAME else cfg(e["params"]))
        rows.append([name, conf] + [pm(e["metrics_mean"][m], e["metrics_sd"][m]) for m in METRICS])
    return table(head, rows)


def selection_table() -> list:
    head = ["View", "Best model (mean CV ROC-AUC)", "SD of its 15 fold values", "Threshold (best - SD)", "Eligible models", "Chosen model"]
    rows = []
    for view in VIEWS:
        sv = sel["views"][view]
        rows.append([VIEW_TITLE[view], f"{sv['best_model']} ({sv['best_mean']:.4f})", f"{sv['sd_best']:.4f}",
                     f"{sv['threshold']:.4f}", ", ".join(sv["eligible"]), f"**{sv['chosen_model']}**"])
    return table(head, rows)


def test_table(view) -> list:
    head = ["Model"] + [f"{METRIC_TITLE[m]} [95% interval]" for m in METRICS]
    rows = []
    for name in ALL_MODELS:
        e = test_entry(view, name)
        rows.append([name] + [with_ci(e["metrics"][m], e["ci95"][m]) for m in METRICS])
    return table(head, rows)


def confusion_table(view) -> list:
    rows = []
    for name in ALL_MODELS:
        c = test_entry(view, name)["confusion"]
        rows.append([name, c["tn"], c["fp"], c["fn"], c["tp"]])
    return table(["Model", "True Negative", "False Positive", "False Negative", "True Positive"], rows)


def comparison_table() -> list:
    head = ["Model", "CV ROC-AUC primary", "CV ROC-AUC sensitivity", "Diff (S - P)", "Test ROC-AUC primary",
            "Test ROC-AUC sensitivity", "Diff (S - P)", "Test recall P / S", "Test specificity P / S"]
    rows = []
    for name in ALL_MODELS:
        cp, cs = cv_entry("primary", name)["roc_auc_mean"], cv_entry("sensitivity", name)["roc_auc_mean"]
        tp_, ts_ = test_entry("primary", name)["metrics"], test_entry("sensitivity", name)["metrics"]
        rows.append([name, f3(cp), f3(cs), diff(cs, cp), f3(tp_["roc_auc"]), f3(ts_["roc_auc"]),
                     diff(ts_["roc_auc"], tp_["roc_auc"]), f"{f3(tp_['recall'])} / {f3(ts_['recall'])}",
                     f"{f3(tp_['specificity'])} / {f3(ts_['specificity'])}"])
    return table(head, rows)


rank_cv = {v: sorted(MODEL_ORDER, key=lambda m: -sel["views"][v]["models"][m]["roc_auc_mean"]) for v in VIEWS}
rank_test = {v: sorted(MODEL_ORDER, key=lambda m: -tst["views"][v]["models"][m]["metrics"]["roc_auc"]) for v in VIEWS}
chosen = {v: sel["views"][v]["chosen_model"] for v in VIEWS}
undefined_notes = []
for v in VIEWS:
    for name in ALL_MODELS:
        e = test_entry(v, name)
        bad = {k: n for k, n in e["n_undefined_bootstrap"].items() if n}
        if bad:
            undefined_notes.append(f"{VIEW_TITLE[v]}, {name}: " + ", ".join(f"{METRIC_TITLE[k]} undefined in {n} of 1000 resamples" for k, n in bad.items()))
cv_undef = [f"{VIEW_TITLE[v]}, {name}: " + ", ".join(f"{METRIC_TITLE[k]} undefined in {n} folds" for k, n in cv_entry(v, name).get("metrics_n_undefined", {}).items() if n)
            for v in VIEWS for name in [BASELINE] + MODEL_ORDER if any(cv_entry(v, name).get("metrics_n_undefined", {}).values())]
demo = tst["demo_pipeline"]
sw = {**sel["software"], **{k: v for k, v in tst["software"].items() if k == "joblib"}}
fnn_summary = []
if fnn:
    sw["torch"] = fnn["meta"]["torch"]
    fp_, fs_ = test_entry("primary", FNN_NAME), test_entry("sensitivity", FNN_NAME)
    fnn_summary = [f"- **FNN (secondary experiment):** test ROC-AUC {with_ci(fp_['metrics']['roc_auc'], fp_['ci95']['roc_auc'])} (primary) and "
                   f"{with_ci(fs_['metrics']['roc_auc'], fs_['ci95']['roc_auc'])} (sensitivity), from seed-averaged probabilities. "
                   "It took no part in model selection and is not deployed."]

L = [
    "# 09 · Model results: primary and sensitivity analysis",
    "",
    "Generated by `src/s11_build_report.py` from the saved result files. Models were selected by cross-validation on the training data "
    "only (`src/s08_cv_select.py`) and evaluated once on the frozen test set (`src/s09_test_eval.py`)"
    + (", with the optional FNN in `src/s10_fnn.py`." if fnn else "."),
    "",
    "**Views.** Primary = all 520 records (416 training rows; 104 test rows from 49 profile groups). Sensitivity = the 251 unique "
    "questionnaire profiles (202 training rows; 49 test rows). Both use the frozen train/test split and the 15 frozen cross-validation "
    "folds (3 repeats x 5 folds, seeds 0, 1, 2), assigned per profile, so no profile is on both sides of any boundary. Positive = the "
    "dataset label Positive. Predictions use threshold 0.5 (SVM: decision value 0). F1 is 2TP / (2TP + FP + FN).",
    "",
    "## Summary",
    "",
    f"- **Chosen by the declared rule:** primary = **{chosen['primary']}**; sensitivity = **{chosen['sensitivity']}**"
    + (" (the two views agree)." if chosen["primary"] == chosen["sensitivity"] else " (the two views differ)."),
    f"- **Cross-validation ranking of the four models** (ROC-AUC, best first): primary = {' > '.join(rank_cv['primary'])}; "
    f"sensitivity = {' > '.join(rank_cv['sensitivity'])}"
    + (" (same order)." if rank_cv["primary"] == rank_cv["sensitivity"] else " (different order)."),
    f"- **Test ROC-AUC of the chosen demo model** ({chosen['primary']}, primary view): "
    f"{with_ci(tst['views']['primary']['models'][chosen['primary']]['metrics']['roc_auc'], tst['views']['primary']['models'][chosen['primary']]['ci95']['roc_auc'])}; "
    f"sensitivity view, {chosen['sensitivity']}: "
    f"{with_ci(tst['views']['sensitivity']['models'][chosen['sensitivity']]['metrics']['roc_auc'], tst['views']['sensitivity']['models'][chosen['sensitivity']]['ci95']['roc_auc'])}.",
    f"- **Demo pipeline saved:** `{demo['file']}` ({demo['model']}, {json.dumps(demo['params'])}); the reloaded pipeline gives identical test "
    "predictions and identical scores.",
    "- **Baseline:** the majority-class baseline predicts Positive for every row, so its ROC-AUC is exactly 0.5 in every fold and on both test sets.",
    *fnn_summary,
    "",
    "## 1. Cross-validation results (training data only)",
    "",
    "Mean ± SD over the 15 frozen validation folds, for the configuration chosen within each model. The SD is the spread across "
    "overlapping folds, not a confidence interval.",
    "",
    f"### {VIEW_TITLE['primary']}",
    "",
] + cv_table("primary") + [
    "",
    f"### {VIEW_TITLE['sensitivity']}",
    "",
] + cv_table("sensitivity") + [
    "",
    ("Undefined values in cross-validation: " + "; ".join(cv_undef) + ".") if cv_undef else "No metric was undefined in any cross-validation fold.",
    "",
    "![Cross-validation ROC-AUC by model, both views.](figures/fig05_cv_roc_auc_by_model.png)",
    "",
    "## 2. Selection record",
    "",
    "Rule, declared before any result existed and applied separately inside each view with identical grids, folds and rule: "
    "(1) within a model, the configuration with the highest mean CV ROC-AUC (exact ties: the first listed); (2) across the four models, "
    "every model whose mean CV ROC-AUC is at least best minus the SD of the best model's 15 fold values is eligible, and the eligible "
    "model that is simplest in the fixed order Logistic Regression < Categorical NB < RBF SVM < Random Forest is chosen; "
    "(3) the primary view's choice is the demo model, the sensitivity view's choice is for comparison only.",
    "",
] + selection_table() + [
    "",
    "Grids: Logistic Regression C in {0.01, 0.1, 1, 10}; Categorical NB alpha in {0.1, 0.5, 1.0} with fixed age bands; RBF SVM C in {0.1, 1, 10} x "
    "gamma in {scale, 0.01, 0.1}; Random Forest 300 trees, max_depth in {None, 5, 10} x min_samples_leaf in {1, 3}. The full grid results are in "
    "`reports/model_cv_results.csv`.",
    "",
    "## 3. Test results (one evaluation per view)",
    "",
    "Each model's configuration was fixed by cross-validation, refitted on the full training set of the view and evaluated once. No test result "
    "changed a configuration, a model choice or the selection. Intervals are 95% group-bootstrap intervals "
    "(1,000 resamples of whole profile groups, seed 2026) computed from the fixed predictions"
    + ("; the FNN row uses the seed-averaged probabilities." if fnn else "."),
    "",
    f"### {VIEW_TITLE['primary']}: {tst['views']['primary']['n_test_rows']} test rows from {tst['views']['primary']['n_test_groups']} profile groups "
    f"({tst['views']['primary']['test_positive']} Positive, {tst['views']['primary']['test_negative']} Negative)",
    "",
] + test_table("primary") + [
    "",
    "Confusion matrices (rows are the true class, columns the predicted class):",
    "",
] + confusion_table("primary") + [
    "",
    f"### {VIEW_TITLE['sensitivity']}: {tst['views']['sensitivity']['n_test_rows']} test rows "
    f"({tst['views']['sensitivity']['test_positive']} Positive, {tst['views']['sensitivity']['test_negative']} Negative)",
    "",
] + test_table("sensitivity") + [
    "",
    "Confusion matrices:",
    "",
] + confusion_table("sensitivity") + [
    "",
    ("Undefined bootstrap resamples: " + "; ".join(undefined_notes) + ".") if undefined_notes else "No metric was undefined in any bootstrap resample.",
    "",
    "![Test ROC curves for both views.](figures/fig06_test_roc_curves.png)",
    "",
    "## 4. Primary vs sensitivity",
    "",
] + comparison_table() + [
    "",
    f"- Ranking by CV ROC-AUC: primary = {' > '.join(rank_cv['primary'])}; sensitivity = {' > '.join(rank_cv['sensitivity'])}. "
    + ("The ranking is the same in both views." if rank_cv["primary"] == rank_cv["sensitivity"] else "The ranking differs between the views."),
    f"- Model chosen by the rule: primary = {chosen['primary']}; sensitivity = {chosen['sensitivity']}. "
    + ("The choice agrees." if chosen["primary"] == chosen["sensitivity"] else "The choice differs."),
    f"- Ranking by test ROC-AUC (information only, used for no choice): primary = {' > '.join(rank_test['primary'])}; "
    f"sensitivity = {' > '.join(rank_test['sensitivity'])}.",
    "- In the primary view every record counts, so profiles that occur many times weigh more; the sensitivity view counts each unique "
    "profile once. Differences between the two views reflect that weighting and the smaller, duplicate-free sample; neither view is "
    "'more correct', and the plan reports both.",
    "",
]
if fnn:
    L += ["## 5. FNN (secondary experiment)", ""]
    for view in VIEWS:
        t = fnn["views"][view]["test"]
        c = fnn["views"][view]["cv"]
        L += [f"**{VIEW_TITLE[view]}**", "",
              f"- Test metrics, mean ± SD across the 5 seeds: " + ", ".join(f"{METRIC_TITLE[m]} {pm(t['metrics_mean_across_seeds'][m], t['metrics_sd_across_seeds'][m])}" for m in METRICS) + ".",
              f"- Selected epoch count per seed in the final fit (seeds 0 to 4): {t['epochs_per_seed']}.",
              f"- Selected epoch count across the 15 CV folds x 5 seeds: min {c['epochs']['min']}, median {c['epochs']['median']}, max {c['epochs']['max']} "
              f"(n = {c['epochs']['n']}); per-fold values are in `reports/fnn_cv_results.csv`.",
              f"- Runs where the 200-epoch cap, not early stopping, ended phase 1 (best epoch 181 or later): "
              f"{int(((fnn_cv_csv['view'] == view) & (fnn_cv_csv['epochs'] >= 181)).sum())} of {c['epochs']['n']} in CV, "
              f"{sum(1 for e in t['epochs_per_seed'] if e >= 181)} of {len(t['epochs_per_seed'])} in the final fits.", ""]
    L += [f"- Protocol: {fnn['meta']['protocol']}", f"- Reproducibility: {fnn['meta']['reproducibility']}", ""]
L += [
    f"## {6 if fnn else 5}. Notes and limits",
    "",
    f"- Differences between the four fitted models are small relative to the intervals: the primary test set holds only {tst['views']['primary']['n_test_rows']} rows "
    f"from {tst['views']['primary']['n_test_groups']} profile groups, and the sensitivity test set holds {tst['views']['sensitivity']['n_test_rows']} rows "
    f"({tst['views']['sensitivity']['test_negative']} Negative). A single hold-out split is one draw of which profiles land in the test set.",
    "- High scores on one hospital's questionnaire sample are not evidence of clinical usefulness. They describe how well the questionnaire "
    "answers separate the dataset's recorded labels in this one sample; nothing here supports use outside this dataset.",
    "- Bootstrap intervals describe uncertainty conditional on the fitted models and the frozen test split. They do not capture training-split or "
    "model-selection uncertainty. Cross-validation scores are mildly optimistic because the same folds were used to choose configurations and the model.",
    "- Model outputs are predicted dataset classes (Positive or Negative) under a fixed threshold; no threshold tuning, class weighting or resampling was used.",
    "- Identical questionnaire records are reported as identical records. The file has no identifier, so nothing here says anything about who they came from.",
    "",
    "## Software",
    "",
    ", ".join(f"{k} {v}" for k, v in sw.items()) + ".",
    "",
    "Reproduce: `<env python> -I -B src/s08_cv_select.py`, then `src/s09_test_eval.py`" + (", then `src/s10_fnn.py`" if fnn else "")
    + ", then `src/s11_build_report.py` (with `MPLCONFIGDIR` set to a temporary folder). Frozen inputs are verified by SHA-256 before use.",
]
(REPORTS / "09_model_results.md").write_text("\n".join(L) + "\n")
print("wrote reports/09_model_results.md and 2 figures")
